// Package httpclient provides HTTP client with retry logic
package httpclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Client implements the interfaces.HTTPClient interface
type Client struct {
	client     *http.Client
	maxRetries int
	retryDelay time.Duration
	logger     interfaces.Logger
}

// New creates a new HTTP client with default settings
func New(logger interfaces.Logger) interfaces.HTTPClient {
	return NewWithOptions(logger, Options{
		Timeout:    5 * time.Minute,
		MaxRetries: 3,
		RetryDelay: 2 * time.Second,
	})
}

// Options configures the HTTP client
type Options struct {
	Timeout    time.Duration
	MaxRetries int
	RetryDelay time.Duration
}

// NewWithOptions creates a new HTTP client with custom options
func NewWithOptions(logger interfaces.Logger, opts Options) interfaces.HTTPClient {
	return &Client{
		client: &http.Client{
			Timeout: opts.Timeout,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				MaxIdleConns:          100,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   10 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
			},
		},
		maxRetries: opts.MaxRetries,
		retryDelay: opts.RetryDelay,
		logger:     logger,
	}
}

// Get performs a GET request
func (c *Client) Get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	return c.Do(req)
}

// Do performs an HTTP request with retries
func (c *Client) Do(req *http.Request) (*http.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			c.logger.Warning("Retrying request (attempt %d/%d)...", attempt, c.maxRetries)
			time.Sleep(c.retryDelay)
		}

		resp, err := c.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		// Don't retry on success or client errors
		if resp.StatusCode < 500 {
			return resp, nil
		}

		// Server error, close response and retry
		resp.Body.Close()
		lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
		c.logger.Warning("Request failed with status: %d", resp.StatusCode)
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.maxRetries+1, lastErr)
}

// WaitForHTTP waits for an HTTP endpoint to be ready
func (c *Client) WaitForHTTP(ctx context.Context, url string, timeout time.Duration) error {
	c.logger.Info("Waiting for %s to be ready...", url)

	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		resp, err := c.Get(ctx, url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				c.logger.Success("%s is ready", url)
				return nil
			}
		}

		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("timeout waiting for %s after %v", url, timeout)
}

// WaitForPort waits for a TCP port to be ready
func (c *Client) WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error {
	address := fmt.Sprintf("%s:%d", host, port)
	c.logger.Info("Waiting for %s to be ready...", address)

	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		conn, err := net.DialTimeout("tcp", address, 1*time.Second)
		if err == nil {
			conn.Close()
			c.logger.Success("%s is ready", address)
			return nil
		}

		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("service %s not ready after %v", address, timeout)
}

// CheckHealth checks if an HTTP endpoint returns the expected status
func (c *Client) CheckHealth(ctx context.Context, url string, expectedStatus int) error {
	resp, err := c.Get(ctx, url)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != expectedStatus {
		return fmt.Errorf("health check failed: expected status %d, got %d", expectedStatus, resp.StatusCode)
	}

	return nil
}
