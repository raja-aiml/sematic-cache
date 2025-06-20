package testing

import (
	"net/http"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
)

// NewHTTPClient creates a standard HTTP client with default timeout
func NewHTTPClient() *http.Client {
	return NewHTTPClientWithTimeout(constants.DefaultHTTPTimeout)
}

// NewHTTPClientWithTimeout creates an HTTP client with custom timeout
func NewHTTPClientWithTimeout(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			MaxIdleConns:       10,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
			DisableKeepAlives:  false,
		},
	}
}
