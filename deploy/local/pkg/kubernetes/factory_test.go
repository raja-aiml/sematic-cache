package kubernetes

import (
	"context"
	"testing"
	"time"
)

func TestNewClientFactory(t *testing.T) {
	tests := []struct {
		name           string
		kubeconfigPath string
	}{
		{
			name:           "empty_path",
			kubeconfigPath: "",
		},
		{
			name:           "with_path",
			kubeconfigPath: "/tmp/kubeconfig",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cf := NewClientFactory(tt.kubeconfigPath)

			if cf == nil {
				t.Fatal("NewClientFactory returned nil")
			}

			if cf.kubeconfigPath != tt.kubeconfigPath {
				t.Errorf("NewClientFactory() kubeconfigPath = %v, want %v", cf.kubeconfigPath, tt.kubeconfigPath)
			}
		})
	}
}

func TestClientFactory_GetClient(t *testing.T) {
	cf := NewClientFactory("/nonexistent/path")

	// This will fail as kubeconfig doesn't exist
	_, err := cf.GetClient()
	if err == nil {
		t.Error("GetClient() expected error with invalid kubeconfig")
	}
}

func TestClientFactory_GetClientWithRetry(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		retryDelay time.Duration
		wantErr    bool
	}{
		{
			name:       "single_retry",
			maxRetries: 1,
			retryDelay: 10 * time.Millisecond,
			wantErr:    true,
		},
		{
			name:       "multiple_retries",
			maxRetries: 3,
			retryDelay: 10 * time.Millisecond,
			wantErr:    true,
		},
		{
			name:       "zero_retries",
			maxRetries: 0,
			retryDelay: 10 * time.Millisecond,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cf := NewClientFactory("/nonexistent/path")

			start := time.Now()
			_, err := cf.GetClientWithRetry(tt.maxRetries, tt.retryDelay)
			elapsed := time.Since(start)

			if (err != nil) != tt.wantErr {
				t.Errorf("GetClientWithRetry() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Check that retries actually happened
			expectedMinDuration := time.Duration(tt.maxRetries-1) * tt.retryDelay
			if tt.maxRetries > 1 && elapsed < expectedMinDuration {
				t.Errorf("GetClientWithRetry() took %v, expected at least %v", elapsed, expectedMinDuration)
			}
		})
	}
}

func TestGetDefaultClient(t *testing.T) {
	// This will fail as default kubeconfig doesn't exist
	_, err := GetDefaultClient()
	if err == nil {
		t.Error("GetDefaultClient() expected error without valid kubeconfig")
	}
}

func TestMustGetDefaultClient(t *testing.T) {
	// Test that it panics
	defer func() {
		if r := recover(); r == nil {
			t.Error("MustGetDefaultClient() should panic without valid kubeconfig")
		}
	}()

	_ = MustGetDefaultClient()
}

func TestClientFactory_WithContext(t *testing.T) {
	cf := NewClientFactory("/nonexistent/path")

	tests := []struct {
		name    string
		timeout time.Duration
		wantErr bool
	}{
		{
			name:    "context_timeout",
			timeout: 50 * time.Millisecond,
			wantErr: true,
		},
		{
			name:    "immediate_cancel",
			timeout: 0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			if tt.timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tt.timeout)
				defer cancel()
			} else {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel() // Cancel immediately
			}

			_, err := cf.WithContext(ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("WithContext() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDefaultClientFactory(t *testing.T) {
	// Test that DefaultClientFactory is initialized
	if DefaultClientFactory == nil {
		t.Fatal("DefaultClientFactory is nil")
	}

	if DefaultClientFactory.kubeconfigPath != "" {
		t.Errorf("DefaultClientFactory.kubeconfigPath = %v, want empty string", DefaultClientFactory.kubeconfigPath)
	}
}

// Benchmark tests
func BenchmarkNewClientFactory(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewClientFactory("/tmp/kubeconfig")
	}
}

func BenchmarkGetClientWithRetry(b *testing.B) {
	cf := NewClientFactory("/nonexistent/path")

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = cf.GetClientWithRetry(1, 0)
	}
}
