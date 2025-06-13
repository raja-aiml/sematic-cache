// Package openai provides a minimal client wrapper for the OpenAI API using the
// official Go SDK.

package openai

import (
	"context"
	"fmt"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

// Client wraps the OpenAI SDK client.
type Client struct {
	apiKey     string
	BaseURL    string
	APIVersion string
	client     openai.Client
}

// NewClient creates a new OpenAI client.
// NewClient creates a new OpenAI client.
func NewClient(apiKey string) *Client {
	c := &Client{apiKey: apiKey}
	c.configure()
	return c
}

// configure initializes the underlying openai.Client with options.
func (c *Client) configure() {
	opts := []option.RequestOption{option.WithAPIKey(c.apiKey)}
	if c.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(c.BaseURL))
	}
	if c.APIVersion != "" {
		opts = append(opts, option.WithQuery("api-version", c.APIVersion))
	}
	c.client = openai.NewClient(opts...)
}

// SetBaseURL updates the API base URL and reinitializes the SDK client. This is
// mainly used in tests.
func (c *Client) SetBaseURL(url string) {
	c.BaseURL = url
	c.configure()
}

// SetAPIKey updates the API key and reinitializes the SDK client.
func (c *Client) SetAPIKey(key string) {
	c.apiKey = key
	c.configure()
}

// ConfigureAzure sets Azure OpenAI specific parameters from environment.
func (c *Client) ConfigureAzure(key, baseURL, version string) {
	c.apiKey = key
	c.BaseURL = baseURL
	c.APIVersion = version
	c.configure()
}

// Complete calls OpenAI's completion API.
func (c *Client) Complete(ctx context.Context, prompt string) (string, error) {
	params := openai.CompletionNewParams{
		Model: "text-davinci-003",
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: param.NewOpt(prompt),
		},
	}
   resp, err := c.client.Completions.New(ctx, params)
   if err != nil {
       return "", fmt.Errorf("complete: %w", err)
   }
   if len(resp.Choices) == 0 {
       return "", fmt.Errorf("complete: no choices returned")
   }
   return resp.Choices[0].Text, nil
}

// Embedding calls OpenAI's embedding API.
func (c *Client) Embedding(ctx context.Context, text string) ([]float32, error) {
	params := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModelTextEmbeddingAda002,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: []string{text},
		},
	}
   resp, err := c.client.Embeddings.New(ctx, params)
   if err != nil {
       return nil, fmt.Errorf("embedding: %w", err)
   }
   if len(resp.Data) == 0 {
       return nil, fmt.Errorf("embedding: no embedding returned")
   }
	raw := resp.Data[0].Embedding
	vec := make([]float32, len(raw))
	for i, v := range raw {
		vec[i] = float32(v)
	}
	return vec, nil
}
