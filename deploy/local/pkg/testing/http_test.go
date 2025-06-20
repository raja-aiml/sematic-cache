package testing

import (
	"net/http"
	"testing"
	"time"
)

func TestNewHTTPClient(t *testing.T) {
	client := NewHTTPClient()

	if client == nil {
		t.Fatal("NewHTTPClient returned nil")
	}

	// Check that timeout is set
	if client.Timeout != 5*time.Second {
		t.Errorf("NewHTTPClient() Timeout = %v, want %v", client.Timeout, 5*time.Second)
	}

	// Check that transport is configured
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("NewHTTPClient() Transport is not *http.Transport")
	}

	// Check transport configuration
	if transport.MaxIdleConns != 10 {
		t.Errorf("Transport.MaxIdleConns = %v, want %v", transport.MaxIdleConns, 10)
	}

	if transport.IdleConnTimeout != 30*time.Second {
		t.Errorf("Transport.IdleConnTimeout = %v, want %v", transport.IdleConnTimeout, 30*time.Second)
	}
}

func TestNewHTTPClientWithTimeout(t *testing.T) {
	tests := []struct {
		name    string
		timeout time.Duration
	}{
		{
			name:    "1_second",
			timeout: 1 * time.Second,
		},
		{
			name:    "10_seconds",
			timeout: 10 * time.Second,
		},
		{
			name:    "30_seconds",
			timeout: 30 * time.Second,
		},
		{
			name:    "zero_timeout",
			timeout: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewHTTPClientWithTimeout(tt.timeout)

			if client == nil {
				t.Fatal("NewHTTPClientWithTimeout returned nil")
			}

			if client.Timeout != tt.timeout {
				t.Errorf("NewHTTPClientWithTimeout() Timeout = %v, want %v", client.Timeout, tt.timeout)
			}

			// Check that transport is configured
			transport, ok := client.Transport.(*http.Transport)
			if !ok {
				t.Fatal("NewHTTPClientWithTimeout() Transport is not *http.Transport")
			}

			// Check transport configuration remains the same
			if transport.MaxIdleConns != 10 {
				t.Errorf("Transport.MaxIdleConns = %v, want %v", transport.MaxIdleConns, 10)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewHTTPClient(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewHTTPClient()
	}
}

func BenchmarkNewHTTPClientWithTimeout(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewHTTPClientWithTimeout(10 * time.Second)
	}
}
