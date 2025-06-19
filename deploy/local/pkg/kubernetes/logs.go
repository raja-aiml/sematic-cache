package kubernetes

import (
	"context"
	"fmt"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	corev1 "k8s.io/api/core/v1"
)

// LogRetriever provides unified log retrieval functionality
type LogRetriever struct {
	client *Client
	logger *utils.Logger
}

// NewLogRetriever creates a new log retriever
func NewLogRetriever(client *Client) *LogRetriever {
	return &LogRetriever{
		client: client,
		logger: utils.NewLogger("logs"),
	}
}

// LogOptions configures log retrieval
type LogOptions struct {
	Namespace     string
	LabelSelector string
	PodName       string
	TailLines     int64
	Follow        bool
	ShowPodName   bool
}

// GetLogs retrieves logs based on the provided options
func (lr *LogRetriever) GetLogs(ctx context.Context, opts LogOptions) (string, error) {
	if opts.PodName != "" {
		// Get logs for specific pod
		return lr.getPodLogs(ctx, opts.Namespace, opts.PodName, opts.TailLines)
	}

	// Get logs for all pods matching selector
	return lr.getPodsLogs(ctx, opts)
}

// getPodLogs retrieves logs for a specific pod
func (lr *LogRetriever) getPodLogs(ctx context.Context, namespace, podName string, tailLines int64) (string, error) {
	logs, err := lr.client.GetPodLogs(ctx, namespace, podName, tailLines)
	if err != nil {
		return "", fmt.Errorf("failed to get logs for pod %s/%s: %w", namespace, podName, err)
	}
	return logs, nil
}

// getPodsLogs retrieves logs for multiple pods
func (lr *LogRetriever) getPodsLogs(ctx context.Context, opts LogOptions) (string, error) {
	pods, err := lr.client.GetPods(ctx, opts.Namespace, opts.LabelSelector)
	if err != nil {
		return "", fmt.Errorf("failed to get pods: %w", err)
	}

	if len(pods) == 0 {
		return "", fmt.Errorf("no pods found with selector %s in namespace %s", opts.LabelSelector, opts.Namespace)
	}

	var allLogs strings.Builder

	for _, pod := range pods {
		if opts.ShowPodName {
			allLogs.WriteString(fmt.Sprintf("\n=== Pod: %s/%s ===\n", pod.Namespace, pod.Name))
		}

		logs, err := lr.client.GetPodLogs(ctx, pod.Namespace, pod.Name, opts.TailLines)
		if err != nil {
			lr.logger.Error("Failed to get logs for pod %s: %v", pod.Name, err)
			allLogs.WriteString(fmt.Sprintf("Error getting logs: %v\n", err))
			continue
		}

		allLogs.WriteString(logs)
		allLogs.WriteString("\n")
	}

	return allLogs.String(), nil
}

// StreamLogs streams logs from pods (placeholder for future implementation)
func (lr *LogRetriever) StreamLogs(ctx context.Context, opts LogOptions) error {
	if !opts.Follow {
		logs, err := lr.GetLogs(ctx, opts)
		if err != nil {
			return err
		}
		fmt.Print(logs)
		return nil
	}

	// For follow mode, we'd need to implement streaming
	lr.logger.Warn("Follow mode not implemented. Use: kubectl logs -f -n %s -l %s",
		opts.Namespace, opts.LabelSelector)
	return nil
}

// GetPodStatus returns a summary of pod statuses
func (lr *LogRetriever) GetPodStatus(ctx context.Context, namespace, labelSelector string) ([]PodStatus, error) {
	pods, err := lr.client.GetPods(ctx, namespace, labelSelector)
	if err != nil {
		return nil, fmt.Errorf("failed to get pods: %w", err)
	}

	statuses := make([]PodStatus, len(pods))
	for i, pod := range pods {
		statuses[i] = PodStatus{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Phase:     string(pod.Status.Phase),
			Ready:     isPodReady(pod),
			Restarts:  getPodRestarts(pod),
		}
	}

	return statuses, nil
}

// PodStatus represents simplified pod status information
type PodStatus struct {
	Name      string
	Namespace string
	Phase     string
	Ready     bool
	Restarts  int32
}

// Helper functions
func isPodReady(pod corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func getPodRestarts(pod corev1.Pod) int32 {
	var restarts int32
	for _, containerStatus := range pod.Status.ContainerStatuses {
		restarts += containerStatus.RestartCount
	}
	return restarts
}
