// Package kubernetes provides Kubernetes operations using client-go SDK
package kubernetes

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Client implements the KubernetesClient interface
type Client struct {
	clientset kubernetes.Interface
	config    *rest.Config
	logger    interfaces.Logger
}

// NewClient creates a new Kubernetes client
func NewClient(logger interfaces.Logger, kubeconfigPath string) (interfaces.KubernetesClient, error) {
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
		logger:    logger,
	}, nil
}

// ContextExists checks if a kubectl context exists
func (c *Client) ContextExists(contextName string) (bool, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	config, err := loadingRules.Load()
	if err != nil {
		return false, fmt.Errorf("failed to load kubeconfig: %w", err)
	}

	_, exists := config.Contexts[contextName]
	return exists, nil
}

// GetCurrentContext returns the current kubectl context
func (c *Client) GetCurrentContext() (string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	configAccess := clientcmd.NewDefaultPathOptions()

	config, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		loadingRules,
		&clientcmd.ConfigOverrides{},
	).RawConfig()
	if err != nil {
		return "", fmt.Errorf("failed to load config: %w", err)
	}

	_ = configAccess // Avoid unused variable
	return config.CurrentContext, nil
}

// WaitForDeployment waits for a deployment to be ready
func (c *Client) WaitForDeployment(ctx context.Context, name, namespace string, timeout time.Duration) error {
	c.logger.Info("Waiting for deployment %s/%s to be ready...", namespace, name)

	return wait.PollImmediate(2*time.Second, timeout, func() (bool, error) {
		deployment, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		return deployment.Status.ReadyReplicas == *deployment.Spec.Replicas, nil
	})
}

// WaitForService waits for a service to be ready
func (c *Client) WaitForService(ctx context.Context, name, namespace string, timeout time.Duration) error {
	c.logger.Info("Waiting for service %s/%s to be ready...", namespace, name)

	return wait.PollImmediate(2*time.Second, timeout, func() (bool, error) {
		service, err := c.clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// Check if service has endpoints
		endpoints, err := c.clientset.CoreV1().Endpoints(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// Service is ready if it has at least one endpoint
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) > 0 {
				return true, nil
			}
		}

		_ = service // Avoid unused variable
		return false, nil
	})
}

// GetPods returns pods matching the label selector
func (c *Client) GetPods(ctx context.Context, namespace string, labelSelector string) ([]interfaces.PodInfo, error) {
	pods, err := c.clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods: %w", err)
	}

	var result []interfaces.PodInfo
	for _, pod := range pods.Items {
		info := interfaces.PodInfo{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Status:    string(pod.Status.Phase),
			Ready:     isPodReady(&pod),
			IP:        pod.Status.PodIP,
			Node:      pod.Spec.NodeName,
			Labels:    pod.Labels,
		}
		result = append(result, info)
	}

	return result, nil
}

// GetServices returns services in a namespace
func (c *Client) GetServices(ctx context.Context, namespace string) ([]interfaces.ServiceInfo, error) {
	services, err := c.clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list services: %w", err)
	}

	var result []interfaces.ServiceInfo
	for _, svc := range services.Items {
		info := interfaces.ServiceInfo{
			Name:      svc.Name,
			Namespace: svc.Namespace,
			Type:      string(svc.Spec.Type),
			ClusterIP: svc.Spec.ClusterIP,
			Selector:  svc.Spec.Selector,
		}

		for _, port := range svc.Spec.Ports {
			info.Ports = append(info.Ports, interfaces.ServicePort{
				Name:       port.Name,
				Protocol:   string(port.Protocol),
				Port:       port.Port,
				TargetPort: port.TargetPort.IntVal,
				NodePort:   port.NodePort,
			})
		}

		result = append(result, info)
	}

	return result, nil
}

// ApplyManifest applies a Kubernetes manifest
func (c *Client) ApplyManifest(ctx context.Context, manifest []byte) error {
	// This would use dynamic client to apply arbitrary manifests
	// For now, this is a placeholder
	return fmt.Errorf("not implemented")
}

// DeleteManifest deletes resources from a manifest
func (c *Client) DeleteManifest(ctx context.Context, manifest []byte) error {
	// This would use dynamic client to delete arbitrary manifests
	// For now, this is a placeholder
	return fmt.Errorf("not implemented")
}

// GetLogs returns logs for a container
func (c *Client) GetLogs(ctx context.Context, namespace, podName, containerName string, follow bool) (io.ReadCloser, error) {
	opts := &corev1.PodLogOptions{
		Follow:    follow,
		Container: containerName,
	}

	req := c.clientset.CoreV1().Pods(namespace).GetLogs(podName, opts)

	stream, err := req.Stream(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get log stream: %w", err)
	}

	return stream, nil
}

// Helper functions

func isPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
