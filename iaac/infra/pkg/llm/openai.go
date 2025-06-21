package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// OpenAIProvider implements the Provider interface for OpenAI
type OpenAIProvider struct {
	client *openai.Client
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(apiKey string, baseURL string) (*OpenAIProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(opts...)

	return &OpenAIProvider{
		client: &client,
	}, nil
}

// Name returns the provider name
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Complete generates a completion using OpenAI API
func (p *OpenAIProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	// Build messages for OpenAI
	messages := make([]openai.ChatCompletionMessageParamUnion, len(req.Messages))
	for i, msg := range req.Messages {
		switch msg.Role {
		case "system":
			messages[i] = openai.SystemMessage(msg.Content)
		case "user":
			messages[i] = openai.UserMessage(msg.Content)
		case "assistant":
			messages[i] = openai.AssistantMessage(msg.Content)
		default:
			return nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	// Build request parameters
	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    openai.ChatModel(req.Model),
	}

	if req.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(req.MaxTokens))
	}

	if req.Temperature > 0 {
		params.Temperature = openai.Float(float64(req.Temperature))
	}

	if req.TopP > 0 {
		params.TopP = openai.Float(float64(req.TopP))
	}

	// Note: JSON response format may require specific model support
	// For now, we'll handle JSON formatting in the prompt itself

	// Call OpenAI API
	completion, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API error: %w", err)
	}

	// Convert response to our format
	response := &CompletionResponse{
		ID:      completion.ID,
		Model:   completion.Model,
		Created: completion.Created,
		Choices: make([]Choice, len(completion.Choices)),
		Usage: Usage{
			PromptTokens:     int(completion.Usage.PromptTokens),
			CompletionTokens: int(completion.Usage.CompletionTokens),
			TotalTokens:      int(completion.Usage.TotalTokens),
		},
	}

	for i, choice := range completion.Choices {
		response.Choices[i] = Choice{
			Index: int(choice.Index),
			Message: Message{
				Role:    string(choice.Message.Role),
				Content: choice.Message.Content,
			},
			FinishReason: string(choice.FinishReason),
		}
	}

	return response, nil
}

// NewDefaultOpenAIProvider creates an OpenAI provider with default settings
func NewDefaultOpenAIProvider(apiKey string) (*OpenAIProvider, error) {
	return NewOpenAIProvider(apiKey, "")
}
