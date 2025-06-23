package http

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/raja-aiml/sematic-cache/devops/pkg/devops/logger"
)

func TestHTTPClient(t *testing.T) {
	t.Run("successful GET request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "GET", r.Method)
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		}))
		defer server.Close()

		client := NewClient()
		resp, err := client.Get(context.Background(), server.URL)
		require.NoError(t, err)
		defer resp.Body.Close()
		
		assert.Equal(t, http.StatusOK, resp.StatusCode)
	})

	t.Run("successful POST request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			w.WriteHeader(http.StatusCreated)
		}))
		defer server.Close()

		client := NewClient()
		resp, err := client.Post(context.Background(), server.URL, bytes.NewReader([]byte("test")))
		require.NoError(t, err)
		defer resp.Body.Close()
		
		assert.Equal(t, http.StatusCreated, resp.StatusCode)
	})

	t.Run("retry on server error", func(t *testing.T) {
		attempts := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			attempts++
			if attempts < 3 {
				w.WriteHeader(http.StatusInternalServerError)
			} else {
				w.WriteHeader(http.StatusOK)
			}
		}))
		defer server.Close()

		client := NewClientWithOptions(
			WithRetries(3),
			WithDelay(10*time.Millisecond),
		)
		
		resp, err := client.Get(context.Background(), server.URL)
		require.NoError(t, err)
		defer resp.Body.Close()
		
		assert.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, 3, attempts)
	})

	t.Run("no retry on client error", func(t *testing.T) {
		attempts := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			attempts++
			w.WriteHeader(http.StatusBadRequest)
		}))
		defer server.Close()

		client := NewClientWithOptions(WithRetries(3))
		resp, err := client.Get(context.Background(), server.URL)
		require.NoError(t, err)
		defer resp.Body.Close()
		
		assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
		assert.Equal(t, 1, attempts, "Should not retry on client error")
	})

	t.Run("context cancellation", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		client := NewClient()
		_, err := client.Get(ctx, server.URL)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "context canceled")
	})
}

func TestWaitForHTTP(t *testing.T) {
	t.Run("endpoint becomes ready", func(t *testing.T) {
		ready := make(chan bool)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-ready:
				w.WriteHeader(http.StatusOK)
			default:
				w.WriteHeader(http.StatusServiceUnavailable)
			}
		}))
		defer server.Close()

		// Make endpoint ready after 50ms
		go func() {
			time.Sleep(50 * time.Millisecond)
			close(ready)
		}()

		client := NewClient()
		err := client.WaitForHTTP(context.Background(), server.URL, http.StatusOK, 1*time.Second)
		assert.NoError(t, err)
	})

	t.Run("timeout waiting for endpoint", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer server.Close()

		client := NewClient()
		err := client.WaitForHTTP(context.Background(), server.URL, http.StatusOK, 100*time.Millisecond)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not ready after")
	})
}

func TestWaitForPort(t *testing.T) {
	t.Run("port becomes available", func(t *testing.T) {
		// Start a listener after a delay
		go func() {
			time.Sleep(50 * time.Millisecond)
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				return
			}
			defer listener.Close()
			time.Sleep(200 * time.Millisecond) // Keep it open
		}()

		// Get a free port
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		addr := listener.Addr().(*net.TCPAddr)
		port := addr.Port
		listener.Close()

		// Now start the delayed listener on that port
		go func() {
			time.Sleep(50 * time.Millisecond)
			listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
			if err != nil {
				return
			}
			defer listener.Close()
			time.Sleep(200 * time.Millisecond)
		}()

		err = WaitForPort(context.Background(), "127.0.0.1", port, 500*time.Millisecond)
		assert.NoError(t, err)
	})

	t.Run("timeout waiting for port", func(t *testing.T) {
		// Use a port that's definitely not in use
		err := WaitForPort(context.Background(), "127.0.0.1", 65534, 100*time.Millisecond)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not ready after")
	})
}

func TestHealthChecker(t *testing.T) {
	t.Run("HTTP endpoint healthy", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		checker := NewHealthChecker()
		err := checker.CheckHTTPEndpoint(context.Background(), server.URL, http.StatusOK)
		assert.NoError(t, err)
	})

	t.Run("HTTP endpoint unhealthy - wrong status", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer server.Close()

		checker := NewHealthChecker()
		err := checker.CheckHTTPEndpoint(context.Background(), server.URL, http.StatusOK)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "expected status 200, got 503")
	})

	t.Run("TCP port check - open", func(t *testing.T) {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		defer listener.Close()

		addr := listener.Addr().(*net.TCPAddr)
		
		checker := NewHealthChecker()
		err = checker.CheckTCPPort(context.Background(), "127.0.0.1", addr.Port)
		assert.NoError(t, err)
	})

	t.Run("TCP port check - closed", func(t *testing.T) {
		checker := NewHealthChecker()
		err := checker.CheckTCPPort(context.Background(), "127.0.0.1", 65533)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "port check failed")
	})
}

func TestClientOptions(t *testing.T) {
	var buf bytes.Buffer
	customLogger := logger.NewWithOptions(logger.DebugLevel, &buf)

	client := NewClientWithOptions(
		WithTimeout(5*time.Second),
		WithRetries(5),
		WithDelay(1*time.Second),
		WithLogger(customLogger),
	)

	assert.Equal(t, 5*time.Second, client.client.Timeout)
	assert.Equal(t, 5, client.retries)
	assert.Equal(t, 1*time.Second, client.delay)
	assert.Equal(t, customLogger, client.logger)
}