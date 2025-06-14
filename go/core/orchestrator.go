package core

import (
   "context"
   "fmt"

   "github.com/raja-aiml/sematic-cache/go/openai"
)

// Orchestrator routes input messages to registered agents based on a routing function.
type Orchestrator struct {
   agents map[string]*Agent
   router func(input string) string
   client *openai.Client
   chatOptions openai.ChatOptions
}

// NewOrchestrator creates an Orchestrator with the given routing function, OpenAI client, and chat options.
func NewOrchestrator(router func(string) string, client *openai.Client, opts openai.ChatOptions) *Orchestrator {
   return &Orchestrator{agents: make(map[string]*Agent), router: router, client: client, chatOptions: opts}
}

// RegisterAgent adds an agent under the given ID.
func (o *Orchestrator) RegisterAgent(id string, agent *Agent) {
   o.agents[id] = agent
}

// Route receives an input string, chooses an agent via the router, and returns the agent's response.
func (o *Orchestrator) Route(ctx context.Context, input string) (string, error) {
   agentID := o.router(input)
   agent, ok := o.agents[agentID]
   if !ok {
       return "", fmt.Errorf("agent '%s' not found", agentID)
   }
   agent.AddUser(input)
   return agent.Chat(ctx, o.client, o.chatOptions)
}