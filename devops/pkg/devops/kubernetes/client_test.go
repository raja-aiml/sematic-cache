package kubernetes

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestKubernetesClient(t *testing.T) {
	// These tests use a fake clientset for unit testing
	// Integration tests would require a real Kubernetes cluster
	
	t.Run("NewClient", func(t *testing.T) {
		// This will fail unless there's a valid kubeconfig
		// which is expected in unit tests
		_, err := NewClient()
		assert.Error(t, err)
	})
	
	t.Run("NamespaceOperations", func(t *testing.T) {
		// Create a fake clientset
		fakeClient := fake.NewSimpleClientset()
		
		// Create a client with the fake clientset
		client := &Client{
			clientset: fakeClient,
			logger:    nil, // Will be set to default
		}
		
		ctx := context.Background()
		
		// Test namespace creation
		err := client.CreateNamespace(ctx, "test-namespace")
		assert.NoError(t, err)
		
		// Verify namespace was created
		namespaces, err := fakeClient.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
		assert.NoError(t, err)
		assert.Len(t, namespaces.Items, 1)
		assert.Equal(t, "test-namespace", namespaces.Items[0].Name)
		
		// Test namespace exists
		exists, err := client.NamespaceExists(ctx, "test-namespace")
		assert.NoError(t, err)
		assert.True(t, exists)
		
		// Test namespace doesn't exist
		exists, err = client.NamespaceExists(ctx, "non-existent")
		assert.NoError(t, err)
		assert.False(t, exists)
		
		// Test namespace deletion
		err = client.DeleteNamespace(ctx, "test-namespace")
		assert.NoError(t, err)
	})
	
	t.Run("GetPods", func(t *testing.T) {
		// Create fake clientset with some pods
		pod1 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod1",
				Namespace: "default",
				Labels: map[string]string{
					"app": "test",
				},
			},
		}
		
		pod2 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod2",
				Namespace: "default",
				Labels: map[string]string{
					"app": "test",
				},
			},
		}
		
		pod3 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod3",
				Namespace: "default",
				Labels: map[string]string{
					"app": "other",
				},
			},
		}
		
		fakeClient := fake.NewSimpleClientset(pod1, pod2, pod3)
		client := &Client{
			clientset: fakeClient,
			logger:    nil,
		}
		
		ctx := context.Background()
		
		// Get pods with label app=test
		pods, err := client.GetPods(ctx, "default", map[string]string{"app": "test"})
		assert.NoError(t, err)
		assert.Len(t, pods, 2)
		
		// Get pods with label app=other
		pods, err = client.GetPods(ctx, "default", map[string]string{"app": "other"})
		assert.NoError(t, err)
		assert.Len(t, pods, 1)
		assert.Equal(t, "pod3", pods[0].Name)
	})
}

func TestContextExists(t *testing.T) {
	client := &Client{}
	
	// This test will depend on the local kubeconfig
	// In a unit test environment, it should handle the error gracefully
	_, err := client.ContextExists("test-context")
	
	// We expect an error if no kubeconfig is available
	if err != nil {
		assert.Contains(t, err.Error(), "kubeconfig")
	}
}

func TestGetCurrentContext(t *testing.T) {
	client := &Client{}
	
	// This test will depend on the local kubeconfig
	_, err := client.GetCurrentContext()
	
	// We expect an error if no kubeconfig is available
	if err != nil {
		assert.Contains(t, err.Error(), "kubeconfig")
	}
}