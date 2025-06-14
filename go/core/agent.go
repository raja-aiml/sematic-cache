package core

import (
   "context"

   "github.com/raja-aiml/sematic-cache/go/openai"
)

// ContextChain holds an ordered list of chat messages with a maximum length.
type ContextChain struct {
   messages []openai.ChatMessage
   maxLen   int
}

// NewContextChain creates a ContextChain with the given maximum length (0 = unlimited).
func NewContextChain(maxLen int) *ContextChain {
   return &ContextChain{maxLen: maxLen, messages: make([]openai.ChatMessage, 0, maxLen)}
}

// Add appends a message, evicting the oldest if over capacity.
func (cc *ContextChain) Add(msg openai.ChatMessage) {
   if cc.maxLen > 0 && len(cc.messages) >= cc.maxLen {
       cc.messages = cc.messages[1:]
   }
   cc.messages = append(cc.messages, msg)
}

// Messages returns a copy of the current message chain.
func (cc *ContextChain) Messages() []openai.ChatMessage {
   out := make([]openai.ChatMessage, len(cc.messages))
   copy(out, cc.messages)
   return out
}

// Clear resets the context chain to empty.
func (cc *ContextChain) Clear() {
   cc.messages = cc.messages[:0]
}

// Agent represents an AI agent with its own context chain and expert ID.
type Agent struct {
   ExpertID string
   Chain    *ContextChain
}

// NewAgent creates an Agent with the given expertID and context chain length.
func NewAgent(expertID string, maxChain int) *Agent {
   return &Agent{ExpertID: expertID, Chain: NewContextChain(maxChain)}
}

// AddUser appends a user message to the context.
func (a *Agent) AddUser(text string) {
   a.Chain.Add(openai.ChatMessage{Role: "user", Content: text})
}

// AddAssistant appends an assistant message to the context.
func (a *Agent) AddAssistant(text string) {
   a.Chain.Add(openai.ChatMessage{Role: "assistant", Content: text})
}

// Chat sends the current context to the provided OpenAI client and
// records the assistant's response in the context chain.
func (a *Agent) Chat(ctx context.Context, client *openai.Client, opts openai.ChatOptions) (string, error) {
   // Prepend a system message for expert ID if set
   messages := a.Chain.Messages()
   if a.ExpertID != "" {
       systemMsg := openai.ChatMessage{Role: "system", Content: a.ExpertID}
       messages = append([]openai.ChatMessage{systemMsg}, messages...)
   }
   resp, err := client.Chat(ctx, messages, opts)
   if err != nil {
       return "", err
   }
   a.AddAssistant(resp)
   return resp, nil
}