package kubernetes

import (
	"context"
	"fmt"
	"time"
)

// ClientFactory provides methods for creating Kubernetes clients
type ClientFactory struct {
	kubeconfigPath string
}

// NewClientFactory creates a new client factory
func NewClientFactory(kubeconfigPath string) *ClientFactory {
	return &ClientFactory{
		kubeconfigPath: kubeconfigPath,
	}
}

// GetClient creates a new Kubernetes client with retry logic
func (cf *ClientFactory) GetClient() (*Client, error) {
	return cf.GetClientWithRetry(3, 2*time.Second)
}

// GetClientWithRetry creates a new Kubernetes client with custom retry settings
func (cf *ClientFactory) GetClientWithRetry(maxRetries int, retryDelay time.Duration) (*Client, error) {
	var lastErr error

	for i := 0; i < maxRetries; i++ {
		client, err := NewClient(cf.kubeconfigPath)
		if err == nil {
			return client, nil
		}

		lastErr = err
		if i < maxRetries-1 {
			time.Sleep(retryDelay)
		}
	}

	return nil, fmt.Errorf("failed to create kubernetes client after %d attempts: %w", maxRetries, lastErr)
}

// DefaultClientFactory is a convenience instance with default settings
var DefaultClientFactory = NewClientFactory("")

// GetDefaultClient creates a client using the default factory
func GetDefaultClient() (*Client, error) {
	return DefaultClientFactory.GetClient()
}

// MustGetDefaultClient creates a client or panics
func MustGetDefaultClient() *Client {
	client, err := GetDefaultClient()
	if err != nil {
		panic(fmt.Sprintf("failed to create kubernetes client: %v", err))
	}
	return client
}

// WithContext creates a client that respects context cancellation
func (cf *ClientFactory) WithContext(ctx context.Context) (*Client, error) {
	done := make(chan struct{})
	var client *Client
	var err error

	go func() {
		client, err = cf.GetClient()
		close(done)
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-done:
		return client, err
	}
}
