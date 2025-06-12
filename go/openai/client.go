// Package openai provides a minimal client wrapper for the OpenAI API.

package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

// Client wraps OpenAI API calls.
type Client struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient creates a new OpenAI client.
func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:     apiKey,
		BaseURL:    "https://api.openai.com/v1",
		HTTPClient: &http.Client{},
	}
}

// CompletionRequest holds the request body for completions.
type CompletionRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// CompletionResponse holds the response body for completions.
type CompletionResponse struct {
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
}

// EmbeddingRequest holds the request body for embeddings.
type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// EmbeddingResponse holds the response body for embeddings.
type EmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Complete calls OpenAI's completion API.
func (c *Client) Complete(prompt string) (string, error) {
	reqBody, err := json.Marshal(CompletionRequest{Model: "text-davinci-003", Prompt: prompt})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest("POST", c.BaseURL+"/completions", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("openai: unexpected status %s", resp.Status)
	}

	var res CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", err
	}
	if len(res.Choices) == 0 {
		return "", fmt.Errorf("openai: no choices returned")
	}
	return res.Choices[0].Text, nil
}

// Embedding calls OpenAI's embedding API.
func (c *Client) Embedding(text string) ([]float32, error) {
	reqBody, err := json.Marshal(EmbeddingRequest{Model: "text-embedding-ada-002", Input: text})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", c.BaseURL+"/embeddings", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: unexpected status %s", resp.Status)
	}

	var res EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}
	if len(res.Data) == 0 {
		return nil, fmt.Errorf("openai: no embedding returned")
	}
	return res.Data[0].Embedding, nil
}
