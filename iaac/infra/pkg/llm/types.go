package llm

import (
	"context"
	"time"
)

// Provider represents an LLM provider
type Provider interface {
	// Complete generates a completion for the given messages
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)

	// Name returns the provider name
	Name() string
}

// CompletionRequest represents a request to generate a completion
type CompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float32   `json:"temperature,omitempty"`
	TopP        float32   `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Format      string    `json:"format,omitempty"` // "json" for JSON mode
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"` // system, user, assistant
	Content string `json:"content"`
}

// CompletionResponse represents a response from the LLM
type CompletionResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
	Created int64    `json:"created"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Config represents LLM configuration
type Config struct {
	Provider    string        `json:"provider"`
	APIKey      string        `json:"api_key"`
	Model       string        `json:"model"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float32       `json:"temperature"`
	Timeout     time.Duration `json:"timeout"`
	BaseURL     string        `json:"base_url,omitempty"`
}
