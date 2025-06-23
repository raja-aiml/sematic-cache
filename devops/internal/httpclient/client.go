// Package httpclient provides HTTP utilities with retry logic and health checking
package httpclient

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
)

// Client provides HTTP operations with retry and health check capabilities
type Client struct {
	client  *http.Client
	logger  *logger.Logger
	retries int
	delay   time.Duration
}

// NewClient creates a new HTTP client with default settings
func NewClient() *Client {
	return &Client{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger:  logger.New(),
		retries: 3,
		delay:   2 * time.Second,
	}
}

// Option is a configuration option for the HTTP client
type Option func(*Client)

// WithTimeout sets the HTTP client timeout
func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) {
		c.client.Timeout = timeout
	}
}

// WithRetries sets the number of retries
func WithRetries(retries int) Option {
	return func(c *Client) {
		c.retries = retries
	}
}

// WithDelay sets the delay between retries
func WithDelay(delay time.Duration) Option {
	return func(c *Client) {
		c.delay = delay
	}
}

// WithLogger sets a custom logger
func WithLogger(l *logger.Logger) Option {
	return func(c *Client) {
		c.logger = l
	}
}

// NewClientWithOptions creates a new HTTP client with custom options
func NewClientWithOptions(opts ...Option) *Client {
	c := NewClient()
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Get performs a GET request with retries
func (c *Client) Get(ctx context.Context, url string) (*http.Response, error) {
	return c.doWithRetry(ctx, "GET", url, nil)
}

// Post performs a POST request with retries
func (c *Client) Post(ctx context.Context, url string, body io.Reader) (*http.Response, error) {
	return c.doWithRetry(ctx, "POST", url, body)
}

// doWithRetry performs an HTTP request with retry logic
func (c *Client) doWithRetry(ctx context.Context, method, url string, body io.Reader) (*http.Response, error) {
	var lastErr error

	for attempt := 1; attempt <= c.retries; attempt++ {
		c.logger.Debug("HTTP %s request attempt %d/%d: %s", method, attempt, c.retries, url)

		req, err := http.NewRequestWithContext(ctx, method, url, body)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		resp, err := c.client.Do(req)
		if err == nil && resp.StatusCode < 500 {
			// Success or client error - don't retry
			return resp, nil
		}

		if err != nil {
			lastErr = err
			c.logger.Warn("Request failed: %v", err)
		} else {
			lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
			c.logger.Warn("Request failed with status: %d", resp.StatusCode)
			resp.Body.Close()
		}

		if attempt < c.retries {
			c.logger.Warn("Retrying in %v...", c.delay)
			select {
			case <-time.After(c.delay):
				// Continue to next attempt
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	return nil, fmt.Errorf("request failed after %d attempts: %w", c.retries, lastErr)
}

// WaitForHTTP waits for an HTTP endpoint to become available
func (c *Client) WaitForHTTP(ctx context.Context, url string, expectedCode int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	c.logger.Info("Waiting for HTTP endpoint %s...", url)

	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		resp, err := c.Get(ctx, url)
		cancel()

		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == expectedCode {
				c.logger.Success("HTTP endpoint %s is ready (%d)", url, resp.StatusCode)
				return nil
			}
			c.logger.Debug("Got status %d, expected %d", resp.StatusCode, expectedCode)
		}

		select {
		case <-time.After(time.Second):
			// Continue checking
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return fmt.Errorf("HTTP endpoint %s not ready after %v", url, timeout)
}

// WaitForPort waits for a TCP port to become available
func WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	address := fmt.Sprintf("%s:%d", host, port)

	log := logger.New()
	log.Info("Waiting for %s to be ready...", address)

	for time.Now().Before(deadline) {
		dialer := net.Dialer{
			Timeout: time.Second,
		}

		conn, err := dialer.DialContext(ctx, "tcp", address)
		if err == nil {
			conn.Close()
			log.Success("Service %s is ready", address)
			return nil
		}

		select {
		case <-time.After(time.Second):
			// Continue checking
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return fmt.Errorf("service %s not ready after %v", address, timeout)
}

// HealthChecker provides health checking functionality
type HealthChecker struct {
	client *Client
}

// NewHealthChecker creates a new health checker
func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		client: NewClient(),
	}
}

// CheckHTTPEndpoint checks if an HTTP endpoint is healthy
func (h *HealthChecker) CheckHTTPEndpoint(ctx context.Context, url string, expectedCode int) error {
	resp, err := h.client.Get(ctx, url)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != expectedCode {
		return fmt.Errorf("health check failed: expected status %d, got %d", expectedCode, resp.StatusCode)
	}

	return nil
}

// CheckTCPPort checks if a TCP port is open
func (h *HealthChecker) CheckTCPPort(ctx context.Context, host string, port int) error {
	address := fmt.Sprintf("%s:%d", host, port)

	dialer := net.Dialer{
		Timeout: 5 * time.Second,
	}

	conn, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return fmt.Errorf("port check failed: %w", err)
	}
	conn.Close()

	return nil
}
