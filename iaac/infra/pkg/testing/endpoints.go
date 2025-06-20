package testing

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// EndpointTester provides utilities for testing HTTP endpoints
type EndpointTester struct {
	client  *http.Client
	logger  *utils.Logger
	baseURL string
}

// NewEndpointTester creates a new endpoint tester
func NewEndpointTester(baseURL string) *EndpointTester {
	return &EndpointTester{
		client:  NewHTTPClient(),
		logger:  utils.NewLogger("test"),
		baseURL: baseURL,
	}
}

// TestResult represents the result of an endpoint test
type TestResult struct {
	Endpoint   string
	Method     string
	StatusCode int
	Success    bool
	Error      error
	Body       string
}

// TestStandardEndpoints tests common health and metrics endpoints
func (et *EndpointTester) TestStandardEndpoints() ([]TestResult, error) {
	endpoints := []struct {
		path   string
		method string
	}{
		{"/health", "GET"},
		{"/metrics", "GET"},
	}

	var results []TestResult

	for _, ep := range endpoints {
		result := et.TestEndpoint(ep.method, ep.path, nil)
		results = append(results, result)

		if result.Success {
			et.logger.Info("✓ %s %s: %d", ep.method, ep.path, result.StatusCode)
		} else {
			et.logger.Error("✗ %s %s: %v", ep.method, ep.path, result.Error)
		}
	}

	return results, nil
}

// TestEndpoint tests a single endpoint
func (et *EndpointTester) TestEndpoint(method, path string, body interface{}) TestResult {
	url := et.baseURL + path
	result := TestResult{
		Endpoint: url,
		Method:   method,
	}

	var req *http.Request
	var err error

	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			result.Error = fmt.Errorf("failed to marshal body: %w", err)
			return result
		}
		req, err = http.NewRequest(method, url, bytes.NewBuffer(jsonData))
		if err != nil {
			result.Error = fmt.Errorf("failed to create request: %w", err)
			return result
		}
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, err = http.NewRequest(method, url, nil)
		if err != nil {
			result.Error = fmt.Errorf("failed to create request: %w", err)
			return result
		}
	}

	resp, err := et.client.Do(req)
	if err != nil {
		result.Error = err
		return result
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			// Log error but don't fail the operation
			et.logger.Debug("failed to close response body: %v", err)
		}
	}()

	result.StatusCode = resp.StatusCode
	result.Success = resp.StatusCode >= 200 && resp.StatusCode < 300

	bodyBytes, _ := io.ReadAll(resp.Body)
	result.Body = string(bodyBytes)

	return result
}

// TestCacheOperations tests cache-specific endpoints
func (et *EndpointTester) TestCacheOperations() error {
	et.logger.Info("Testing cache operations...")

	// Test set operation
	setBody := map[string]interface{}{
		"key":   "test-key",
		"value": "test-value",
		"ttl":   300,
	}

	setResult := et.TestEndpoint("POST", "/set", setBody)
	if !setResult.Success {
		return fmt.Errorf("set operation failed: %v", setResult.Error)
	}
	et.logger.Info("✓ Set operation: %d", setResult.StatusCode)

	// Test get operation
	getResult := et.TestEndpoint("GET", "/get?key=test-key", nil)
	if !getResult.Success {
		return fmt.Errorf("get operation failed: %v", getResult.Error)
	}
	et.logger.Info("✓ Get operation: %d - %s", getResult.StatusCode, getResult.Body)

	// Test query operation
	queryBody := map[string]interface{}{
		"text":      "test query",
		"threshold": 0.8,
	}

	queryResult := et.TestEndpoint("POST", "/query", queryBody)
	if !queryResult.Success {
		et.logger.Warn("Query operation returned: %d", queryResult.StatusCode)
	} else {
		et.logger.Info("✓ Query operation: %d", queryResult.StatusCode)
	}

	return nil
}
