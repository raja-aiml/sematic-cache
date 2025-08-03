package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/raja-aiml/sematic-cache/internal/cache"
)

// CacheClient provides HTTP client for cache server API
type CacheClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewCacheClient creates a new cache client
func NewCacheClient(baseURL string) *CacheClient {
	return &CacheClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// GetStats retrieves cache statistics from the server
func (c *CacheClient) GetStats() (*cache.StatsResponse, error) {
	url := fmt.Sprintf("%s/api/v1/stats", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get stats: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var stats cache.StatsResponse
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &stats, nil
}

// Get retrieves a cached value by prompt
func (c *CacheClient) Get(prompt string) (*cache.CacheResponse, error) {
	url := fmt.Sprintf("%s/api/v1/get", c.baseURL)

	reqBody := cache.GetRequest{
		Prompt: prompt,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to get cache entry: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var cacheResp cache.CacheResponse
	if err := json.NewDecoder(resp.Body).Decode(&cacheResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &cacheResp, nil
}

// Set stores a new cache entry
func (c *CacheClient) Set(prompt, answer, modelName, modelID string, embedding []float32) error {
	url := fmt.Sprintf("%s/api/v1/set", c.baseURL)

	reqBody := cache.SetRequest{
		Prompt:    prompt,
		Answer:    answer,
		ModelName: modelName,
		ModelID:   modelID,
		Embedding: embedding,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to set cache entry: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// SearchSimilar finds similar entries based on prompt or embedding
func (c *CacheClient) SearchSimilar(prompt string, embedding []float32, topK int, threshold float64) (*cache.SimilarResponse, error) {
	url := fmt.Sprintf("%s/api/v1/similar", c.baseURL)

	reqBody := cache.SimilarRequest{
		Prompt:    prompt,
		Embedding: embedding,
		TopK:      topK,
		Threshold: threshold,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to search similar entries: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var similarResp cache.SimilarResponse
	if err := json.NewDecoder(resp.Body).Decode(&similarResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &similarResp, nil
}

// Clear clears cache entries (requires server-side implementation)
func (c *CacheClient) Clear(all bool) error {
	url := fmt.Sprintf("%s/api/v1/clear", c.baseURL)
	if all {
		url += "?all=true"
	}

	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to clear cache: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// Health checks if the server is healthy
func (c *CacheClient) Health() error {
	url := fmt.Sprintf("%s/health", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server is not healthy: status %d", resp.StatusCode)
	}

	return nil
}

// Ready checks if the server is ready to serve requests
func (c *CacheClient) Ready() error {
	url := fmt.Sprintf("%s/ready", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("readiness check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server is not ready: status %d", resp.StatusCode)
	}

	return nil
}
