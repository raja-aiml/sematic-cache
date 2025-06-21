package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/llm"
)

// OpenAINLPEngine implements NLP using OpenAI
type OpenAINLPEngine struct {
	provider    llm.Provider
	model       string
	maxTokens   int
	temperature float32
}

// NewOpenAINLPEngine creates a new OpenAI-based NLP engine
func NewOpenAINLPEngine(apiKey, model string, maxTokens int) (*OpenAINLPEngine, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	if model == "" {
		model = "gpt-4-turbo-preview" // Use GPT-4 Turbo for better understanding
	}

	if maxTokens == 0 {
		maxTokens = 1000
	}

	provider, err := llm.NewDefaultOpenAIProvider(apiKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
	}

	return &OpenAINLPEngine{
		provider:    provider,
		model:       model,
		maxTokens:   maxTokens,
		temperature: 0.3, // Lower temperature for more deterministic responses
	}, nil
}

// Interpret converts natural language to structured command
func (e *OpenAINLPEngine) Interpret(ctx context.Context, query string, registry *CommandRegistry) (*InterpretedCommand, error) {
	// Build system prompt with command registry
	systemPrompt := e.buildSystemPrompt(registry)

	// Create the completion request
	req := llm.CompletionRequest{
		Model: e.model,
		Messages: []llm.Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Convert this request to a CLI command: %s", query),
			},
		},
		Temperature: e.temperature,
		MaxTokens:   e.maxTokens,
		Format:      "json",
	}

	resp, err := e.provider.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from LLM")
	}

	// Parse the JSON response
	var interpreted InterpretedCommand
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &interpreted); err != nil {
		return nil, fmt.Errorf("failed to parse LLM response: %w", err)
	}

	// Set the original query
	interpreted.Query = query

	// Validate the interpreted command against registry
	if err := e.validateCommand(&interpreted, registry); err != nil {
		return nil, fmt.Errorf("command validation failed: %w", err)
	}

	return &interpreted, nil
}

// buildSystemPrompt creates a detailed system prompt with command information
func (e *OpenAINLPEngine) buildSystemPrompt(registry *CommandRegistry) string {
	var sb strings.Builder

	sb.WriteString("You are a CLI command interpreter for an infrastructure management tool.\n")
	sb.WriteString("Your task is to convert natural language queries into executable CLI commands.\n\n")

	sb.WriteString("IMPORTANT RULES:\n")
	sb.WriteString("1. Only use commands that exist in the registry below\n")
	sb.WriteString("2. Always validate required options are provided\n")
	sb.WriteString("3. Mark dangerous commands appropriately\n")
	sb.WriteString("4. Provide clear explanations for your interpretation\n")
	sb.WriteString("5. Return confidence score between 0 and 1\n\n")

	sb.WriteString("OUTPUT FORMAT:\n")
	sb.WriteString("Return a JSON object with this structure:\n")
	sb.WriteString(`{
  "command": "main command",
  "subcommand": "subcommand if any",
  "args": ["positional", "arguments"],
  "options": {"option-name": "value"},
  "confidence": 0.95,
  "explanation": "Brief explanation of the interpretation",
  "dangerous": false
}` + "\n\n")

	sb.WriteString("AVAILABLE COMMANDS:\n")
	sb.WriteString(e.formatCommandsForPrompt(registry.Commands, ""))

	return sb.String()
}

// formatCommandsForPrompt formats commands for the system prompt
func (e *OpenAINLPEngine) formatCommandsForPrompt(commands []Command, prefix string) string {
	var sb strings.Builder

	for _, cmd := range commands {
		cmdPath := prefix
		if cmdPath != "" {
			cmdPath += " "
		}
		cmdPath += cmd.Name

		sb.WriteString(fmt.Sprintf("\n%s - %s\n", cmdPath, cmd.Description))

		if cmd.Dangerous {
			sb.WriteString("  [DANGEROUS] This command can cause data loss\n")
		}

		// List options
		if len(cmd.Options) > 0 {
			sb.WriteString("  Options:\n")
			for _, opt := range cmd.Options {
				required := ""
				if opt.Required {
					required = " [REQUIRED]"
				}
				sb.WriteString(fmt.Sprintf("    --%s (%s): %s%s",
					opt.Name, opt.Type, opt.Description, required))
				if opt.Default != "" {
					sb.WriteString(fmt.Sprintf(" (default: %s)", opt.Default))
				}
				sb.WriteString("\n")
			}
		}

		// Process subcommands
		if len(cmd.Subcommands) > 0 {
			sb.WriteString(e.formatCommandsForPrompt(cmd.Subcommands, cmdPath))
		}
	}

	return sb.String()
}

// validateCommand validates the interpreted command against the registry
func (e *OpenAINLPEngine) validateCommand(cmd *InterpretedCommand, registry *CommandRegistry) error {
	// Build command path
	path := []string{cmd.Command}
	if cmd.Subcommand != "" {
		path = append(path, cmd.Subcommand)
	}

	// Find command in registry
	registryCmd, err := registry.FindCommand(path)
	if err != nil {
		return err
	}

	// Set dangerous flag from registry
	cmd.Dangerous = registryCmd.Dangerous

	// Validate required options
	for _, opt := range registryCmd.Options {
		if opt.Required {
			if _, exists := cmd.Options[opt.Name]; !exists {
				return fmt.Errorf("required option --%s is missing", opt.Name)
			}
		}
	}

	// Validate option values against choices if specified
	for optName, optValue := range cmd.Options {
		for _, opt := range registryCmd.Options {
			if opt.Name == optName && len(opt.Choices) > 0 {
				valid := false
				for _, choice := range opt.Choices {
					if choice == optValue {
						valid = true
						break
					}
				}
				if !valid {
					return fmt.Errorf("invalid value for --%s: %s (valid choices: %v)",
						optName, optValue, opt.Choices)
				}
			}
		}
	}

	return nil
}

// GenerateDocumentation creates human-readable documentation
func (e *OpenAINLPEngine) GenerateDocumentation(commands []Command) (string, error) {
	prompt := "Generate a user-friendly guide for using natural language queries with these commands:\n\n"

	// Add command summaries
	for _, cmd := range commands {
		prompt += fmt.Sprintf("- %s: %s\n", cmd.Name, cmd.Description)
	}

	prompt += "\nProvide examples of natural language queries and tips for users."

	req := llm.CompletionRequest{
		Model: e.model,
		Messages: []llm.Message{
			{
				Role:    "system",
				Content: "You are a technical documentation writer. Create clear, concise documentation.",
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Temperature: 0.7,
		MaxTokens:   2000,
	}

	resp, err := e.provider.Complete(context.Background(), req)
	if err != nil {
		return "", fmt.Errorf("failed to generate documentation: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	return resp.Choices[0].Message.Content, nil
}

// InterpretWithExamples provides example-based interpretation
func (e *OpenAINLPEngine) InterpretWithExamples(ctx context.Context, query string, examples []Example) (*InterpretedCommand, error) {
	// Build few-shot prompt with examples
	var examplePrompt strings.Builder
	examplePrompt.WriteString("Here are some example conversions:\n\n")

	for _, ex := range examples {
		examplePrompt.WriteString(fmt.Sprintf("User: %s\nCommand: %s\n\n", ex.Description, ex.Command))
	}

	req := llm.CompletionRequest{
		Model: e.model,
		Messages: []llm.Message{
			{
				Role:    "system",
				Content: "Convert natural language to CLI commands based on the examples provided.",
			},
			{
				Role:    "user",
				Content: examplePrompt.String() + fmt.Sprintf("User: %s\nCommand: ", query),
			},
		},
		Temperature: 0.3,
		MaxTokens:   200,
	}

	resp, err := e.provider.Complete(ctx, req)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from LLM")
	}

	// Parse the command string into InterpretedCommand
	cmdStr := strings.TrimSpace(resp.Choices[0].Message.Content)
	return e.parseCommandString(cmdStr, query)
}

// parseCommandString parses a command string into InterpretedCommand
func (e *OpenAINLPEngine) parseCommandString(cmdStr, query string) (*InterpretedCommand, error) {
	parts := strings.Fields(cmdStr)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty command")
	}

	interpreted := &InterpretedCommand{
		Query:       query,
		Command:     parts[0],
		Args:        []string{},
		Options:     make(map[string]string),
		Confidence:  0.8, // Default confidence for string parsing
		Explanation: fmt.Sprintf("Parsed from command: %s", cmdStr),
	}

	// Parse the rest of the command
	i := 1
	if i < len(parts) && !strings.HasPrefix(parts[i], "-") {
		interpreted.Subcommand = parts[i]
		i++
	}

	// Parse options and arguments
	for i < len(parts) {
		part := parts[i]
		if strings.HasPrefix(part, "--") {
			// Long option
			if strings.Contains(part, "=") {
				kv := strings.SplitN(part[2:], "=", 2)
				interpreted.Options[kv[0]] = kv[1]
			} else if i+1 < len(parts) && !strings.HasPrefix(parts[i+1], "-") {
				interpreted.Options[part[2:]] = parts[i+1]
				i++
			} else {
				interpreted.Options[part[2:]] = "true"
			}
		} else if strings.HasPrefix(part, "-") && len(part) > 1 {
			// Short option
			if i+1 < len(parts) && !strings.HasPrefix(parts[i+1], "-") {
				interpreted.Options[part[1:]] = parts[i+1]
				i++
			} else {
				interpreted.Options[part[1:]] = "true"
			}
		} else {
			// Positional argument
			interpreted.Args = append(interpreted.Args, part)
		}
		i++
	}

	return interpreted, nil
}
