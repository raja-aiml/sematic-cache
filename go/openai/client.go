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

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
   Role    string // "system", "user", or "assistant"
   Content string
}

// ChatOptions holds optional parameters for chat completions.
type ChatOptions struct {
   Model       string   // e.g. "gpt-3.5-turbo"
   Temperature *float64 // sampling temperature
   MaxTokens   *int     // max tokens in completion
   Stop        []string // stop sequences
   TopP        *float64 // nucleus sampling parameter
   N           *int     // number of completions to generate
}

// Chat invokes OpenAI's chat completions API.
// It returns the content of the first choice.
func (c *Client) Chat(ctx context.Context, messages []ChatMessage, opts ChatOptions) (string, error) {
   // build request params
   params := openai.ChatCompletionNewParams{
       Model: openai.ChatModel(opts.Model),
   }
   // map messages
   msgs := make([]openai.ChatCompletionMessageParamUnion, len(messages))
   for i, m := range messages {
       var u openai.ChatCompletionMessageParamUnion
       switch m.Role {
       case "system":
           u.OfSystem = &openai.ChatCompletionSystemMessageParam{
               Content: openai.ChatCompletionSystemMessageParamContentUnion{OfString: param.NewOpt(m.Content)},
           }
       case "assistant":
           u.OfAssistant = &openai.ChatCompletionAssistantMessageParam{
               Content: openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(m.Content)},
           }
       default:
           u.OfUser = &openai.ChatCompletionUserMessageParam{
               Content: openai.ChatCompletionUserMessageParamContentUnion{OfString: param.NewOpt(m.Content)},
           }
       }
       msgs[i] = u
   }
   params.Messages = msgs
   // optional parameters
   if opts.Temperature != nil {
       params.Temperature = param.NewOpt(*opts.Temperature)
   }
   if opts.MaxTokens != nil {
       params.MaxTokens = param.NewOpt(int64(*opts.MaxTokens))
   }
   if opts.Stop != nil {
       params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: opts.Stop}
   }
   if opts.TopP != nil {
       params.TopP = param.NewOpt(*opts.TopP)
   }
   if opts.N != nil {
       params.N = param.NewOpt(int64(*opts.N))
   }
   // call API
   resp, err := c.client.Chat.Completions.New(ctx, params)
   if err != nil {
       return "", fmt.Errorf("chat completion: %w", err)
   }
   if len(resp.Choices) == 0 {
       return "", fmt.Errorf("chat completion: no choices returned")
   }
   return resp.Choices[0].Message.Content, nil
}

// ChatStream invokes the chat completion API with streaming enabled.
// It returns a channel streaming content deltas, and an error if initial request fails.
func (c *Client) ChatStream(ctx context.Context, messages []ChatMessage, opts ChatOptions) (<-chan string, error) {
   // build request params (same as Chat)
   params := openai.ChatCompletionNewParams{
       Model: openai.ChatModel(opts.Model),
   }
   msgs := make([]openai.ChatCompletionMessageParamUnion, len(messages))
   for i, m := range messages {
       var u openai.ChatCompletionMessageParamUnion
       switch m.Role {
       case "system":
           u.OfSystem = &openai.ChatCompletionSystemMessageParam{Content: openai.ChatCompletionSystemMessageParamContentUnion{OfString: param.NewOpt(m.Content)}}
       case "assistant":
           u.OfAssistant = &openai.ChatCompletionAssistantMessageParam{Content: openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(m.Content)}}
       default:
           u.OfUser = &openai.ChatCompletionUserMessageParam{Content: openai.ChatCompletionUserMessageParamContentUnion{OfString: param.NewOpt(m.Content)}}
       }
       msgs[i] = u
   }
   params.Messages = msgs
   if opts.Temperature != nil {
       params.Temperature = param.NewOpt(*opts.Temperature)
   }
   if opts.MaxTokens != nil {
       params.MaxTokens = param.NewOpt(int64(*opts.MaxTokens))
   }
   if opts.Stop != nil {
       params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: opts.Stop}
   }
   if opts.TopP != nil {
       params.TopP = param.NewOpt(*opts.TopP)
   }
   if opts.N != nil {
       params.N = param.NewOpt(int64(*opts.N))
   }
   // initiate streaming
   stream := c.client.Chat.Completions.NewStreaming(ctx, params)
   out := make(chan string)
   go func() {
       defer close(out)
       for stream.Next() {
           chunk := stream.Current()
           if len(chunk.Choices) > 0 {
               delta := chunk.Choices[0].Delta.Content
               if delta != "" {
                   out <- delta
               }
           }
       }
   }()
   return out, nil
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
