// Package kubernetes provides Kubernetes operations using client-go SDK
package kubernetes

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"github.com/raja-aiml/sematic-cache/devops/pkg/devops/logger"
)

// Client provides Kubernetes operations using the client-go SDK
type Client struct {
	clientset *kubernetes.Clientset
	config    *rest.Config
	logger    *logger.Logger
}

// NewClient creates a new Kubernetes client using default kubeconfig
func NewClient() (*Client, error) {
	return NewClientWithConfig("")
}

// NewClientWithConfig creates a new Kubernetes client with specific kubeconfig
func NewClientWithConfig(kubeconfigPath string) (*Client, error) {
	var config *rest.Config
	var err error

	if kubeconfigPath == "" {
		// Try in-cluster config first
		config, err = rest.InClusterConfig()
		if err != nil {
			// Fall back to kubeconfig
			if home := homedir.HomeDir(); home != "" {
				kubeconfigPath = filepath.Join(home, ".kube", "config")
			}
		}
	}

	if config == nil && kubeconfigPath != "" {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfigPath)
		if err != nil {
			return nil, fmt.Errorf("failed to build config: %w", err)
		}
	}

	if config == nil {
		return nil, fmt.Errorf("no valid kubernetes config found")
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}

	return &Client{
		clientset: clientset,
		config:    config,
		logger:    logger.New(),
	}, nil
}

// NewClientWithLogger creates a new Kubernetes client with custom logger
func NewClientWithLogger(l *logger.Logger) (*Client, error) {
	client, err := NewClient()
	if err != nil {
		return nil, err
	}
	client.logger = l
	return client, nil
}

// GetClientset returns the underlying Kubernetes clientset
func (c *Client) GetClientset() kubernetes.Interface {
	return c.clientset
}

// ContextExists checks if a kubectl context exists
func (c *Client) ContextExists(contextName string) (bool, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	configAccess := clientcmd.NewDefaultPathOptions()
	
	config, err := loadingRules.Load()
	if err != nil {
		return false, fmt.Errorf("failed to load kubeconfig: %w", err)
	}
	
	_, exists := config.Contexts[contextName]
	_ = configAccess // Avoid unused variable
	return exists, nil
}

// GetCurrentContext returns the current kubectl context name
func (c *Client) GetCurrentContext() (string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	config, err := loadingRules.Load()
	if err != nil {
		return "", fmt.Errorf("failed to load kubeconfig: %w", err)
	}
	
	return config.CurrentContext, nil
}

// WaitForDeployment waits for a deployment to be ready
func (c *Client) WaitForDeployment(ctx context.Context, name, namespace string, timeout time.Duration) error {
	c.logger.Info("Waiting for deployment %s/%s to be ready...", namespace, name)
	
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		deployment, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		
		// Check if deployment is ready
		if deployment.Status.ObservedGeneration >= deployment.Generation &&
			deployment.Status.Replicas == *deployment.Spec.Replicas &&
			deployment.Status.UpdatedReplicas == *deployment.Spec.Replicas &&
			deployment.Status.ReadyReplicas == *deployment.Spec.Replicas {
			c.logger.Success("Deployment %s/%s is ready", namespace, name)
			return true, nil
		}
		
		c.logger.Debug("Deployment %s/%s status: %d/%d replicas ready", 
			namespace, name, deployment.Status.ReadyReplicas, *deployment.Spec.Replicas)
		return false, nil
	})
}

// ScaleDeployment scales a deployment to the specified number of replicas
func (c *Client) ScaleDeployment(ctx context.Context, name, namespace string, replicas int32) error {
	c.logger.Info("Scaling deployment %s/%s to %d replicas", namespace, name, replicas)
	
	// Get the deployment
	deployment, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment: %w", err)
	}
	
	// Update replicas
	deployment.Spec.Replicas = &replicas
	
	// Update the deployment
	_, err = c.clientset.AppsV1().Deployments(namespace).Update(ctx, deployment, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to scale deployment: %w", err)
	}
	
	c.logger.Success("Deployment %s/%s scaled to %d replicas", namespace, name, replicas)
	return nil
}

// GetDeploymentStatus returns the status of a deployment
func (c *Client) GetDeploymentStatus(ctx context.Context, name, namespace string) (*appsv1.DeploymentStatus, error) {
	deployment, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get deployment: %w", err)
	}
	
	return &deployment.Status, nil
}

// GetPods returns pods matching the given labels
func (c *Client) GetPods(ctx context.Context, namespace string, labels map[string]string) ([]corev1.Pod, error) {
	labelSelector := metav1.FormatLabelSelector(&metav1.LabelSelector{
		MatchLabels: labels,
	})
	
	podList, err := c.clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods: %w", err)
	}
	
	return podList.Items, nil
}

// GetPodLogs returns logs for a specific pod
func (c *Client) GetPodLogs(ctx context.Context, name, namespace string, opts *corev1.PodLogOptions) (string, error) {
	if opts == nil {
		opts = &corev1.PodLogOptions{}
	}
	
	req := c.clientset.CoreV1().Pods(namespace).GetLogs(name, opts)
	logs, err := req.Stream(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to get pod logs: %w", err)
	}
	defer logs.Close()
	
	buf := make([]byte, 4096)
	var result []byte
	for {
		n, err := logs.Read(buf)
		if n > 0 {
			result = append(result, buf[:n]...)
		}
		if err != nil {
			break
		}
	}
	
	return string(result), nil
}

// ExecInPod executes a command in a pod
func (c *Client) ExecInPod(ctx context.Context, name, namespace, container string, command []string) error {
	// Note: Full exec implementation requires additional setup with SPDY
	// This is a placeholder that shows the pattern
	c.logger.Info("Executing command in pod %s/%s: %v", namespace, name, command)
	
	// In a full implementation, you would:
	// 1. Create an exec request
	// 2. Set up SPDY connection
	// 3. Stream stdin/stdout/stderr
	// For now, we'll return a not implemented error
	
	return fmt.Errorf("exec in pod not fully implemented - use kubectl exec for now")
}

// CreateNamespace creates a new namespace
func (c *Client) CreateNamespace(ctx context.Context, name string) error {
	c.logger.Info("Creating namespace %s", name)
	
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	
	_, err := c.clientset.CoreV1().Namespaces().Create(ctx, namespace, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create namespace: %w", err)
	}
	
	c.logger.Success("Namespace %s created", name)
	return nil
}

// DeleteNamespace deletes a namespace
func (c *Client) DeleteNamespace(ctx context.Context, name string) error {
	c.logger.Info("Deleting namespace %s", name)
	
	err := c.clientset.CoreV1().Namespaces().Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		return fmt.Errorf("failed to delete namespace: %w", err)
	}
	
	c.logger.Success("Namespace %s deleted", name)
	return nil
}

// NamespaceExists checks if a namespace exists
func (c *Client) NamespaceExists(ctx context.Context, name string) (bool, error) {
	_, err := c.clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if err.Error() == "namespaces \""+name+"\" not found" {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// WaitForService waits for a service endpoint to be ready
func (c *Client) WaitForService(ctx context.Context, name, namespace string, timeout time.Duration) error {
	c.logger.Info("Waiting for service %s/%s to have endpoints...", namespace, name)
	
	return wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		endpoints, err := c.clientset.CoreV1().Endpoints(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		
		// Check if service has ready endpoints
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) > 0 {
				c.logger.Success("Service %s/%s has ready endpoints", namespace, name)
				return true, nil
			}
		}
		
		c.logger.Debug("Service %s/%s has no ready endpoints yet", namespace, name)
		return false, nil
	})
}

// RolloutRestart performs a rollout restart of a deployment
func (c *Client) RolloutRestart(ctx context.Context, name, namespace string) error {
	c.logger.Info("Restarting deployment %s/%s", namespace, name)
	
	// Get the deployment
	deployment, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment: %w", err)
	}
	
	// Update annotation to trigger restart
	if deployment.Spec.Template.Annotations == nil {
		deployment.Spec.Template.Annotations = make(map[string]string)
	}
	deployment.Spec.Template.Annotations["kubectl.kubernetes.io/restartedAt"] = time.Now().Format(time.RFC3339)
	
	// Update the deployment
	_, err = c.clientset.AppsV1().Deployments(namespace).Update(ctx, deployment, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to restart deployment: %w", err)
	}
	
	c.logger.Success("Deployment %s/%s restart initiated", namespace, name)
	return nil
}