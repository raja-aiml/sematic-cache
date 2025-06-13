// Package openai provides a minimal client wrapper for the OpenAI API using the
// official Go SDK.

package openai

import (
	"context"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI SDK client.
type Client struct {
	apiKey  string
	BaseURL string
	client  *openai.Client
}

// NewClient creates a new OpenAI client.
func NewClient(apiKey string) *Client {
	c := &Client{apiKey: apiKey, BaseURL: openai.DefaultConfig(apiKey).BaseURL}
	c.configure()
	return c
}

func (c *Client) configure() {
	cfg := openai.DefaultConfig(c.apiKey)
	if c.BaseURL != "" {
		cfg.BaseURL = c.BaseURL
	}
	c.client = openai.NewClientWithConfig(cfg)
}

// SetBaseURL updates the API base URL and reinitializes the SDK client. This is
// mainly used in tests.
func (c *Client) SetBaseURL(url string) {
	c.BaseURL = url
	c.configure()
}

// Complete calls OpenAI's completion API.
func (c *Client) Complete(ctx context.Context, prompt string) (string, error) {
	req := openai.CompletionRequest{
		Model:  "text-davinci-003",
		Prompt: prompt,
	}
	resp, err := c.client.CreateCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai: no choices returned")
	}
	return resp.Choices[0].Text, nil
}

// Embedding calls OpenAI's embedding API.
func (c *Client) Embedding(ctx context.Context, text string) ([]float32, error) {
	req := openai.EmbeddingRequest{
		Model: "text-embedding-ada-002",
		Input: []string{text},
	}
	resp, err := c.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("openai: no embedding returned")
	}
	return resp.Data[0].Embedding, nil
}
