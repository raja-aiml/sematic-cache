package core

import (
   "context"
   "strings"
   "testing"

   "github.com/raja-aiml/sematic-cache/go/openai"
)

func TestContextChainEviction(t *testing.T) {
   cc := NewContextChain(2)
   cc.Add(openai.ChatMessage{Role: "user", Content: "m1"})
   cc.Add(openai.ChatMessage{Role: "user", Content: "m2"})
   cc.Add(openai.ChatMessage{Role: "user", Content: "m3"})
   msgs := cc.Messages()
   if len(msgs) != 2 {
       t.Fatalf("expected 2 messages, got %d", len(msgs))
   }
   if msgs[0].Content != "m2" || msgs[1].Content != "m3" {
       t.Errorf("unexpected messages %+v", msgs)
   }
}

func TestContextChainClear(t *testing.T) {
   cc := NewContextChain(0)
   cc.Add(openai.ChatMessage{Role: "user", Content: "m1"})
   cc.Clear()
   if len(cc.Messages()) != 0 {
       t.Errorf("expected chain to be empty after Clear")
   }
}

func TestAgentMessages(t *testing.T) {
   a := NewAgent("expert-1", 2)
   a.AddUser("u1")
   a.AddAssistant("a1")
   a.AddUser("u2")
   a.AddAssistant("a2")
   msgs := a.Chain.Messages()
   // maxLen=2, should keep only last two assistant and user
   if len(msgs) != 2 {
       t.Fatalf("expected 2 messages, got %d", len(msgs))
   }
   if msgs[0].Role != "user" || msgs[0].Content != "u2" {
       t.Errorf("unexpected first message %+v", msgs[0])
   }
   if msgs[1].Role != "assistant" || msgs[1].Content != "a2" {
       t.Errorf("unexpected second message %+v", msgs[1])
   }
}

func TestOrchestratorRouteNotFound(t *testing.T) {
   orc := NewOrchestrator(func(s string) string { return "x" }, &openai.Client{}, openai.ChatOptions{Model: "gpt-3.5-turbo"})
   _, err := orc.Route(context.Background(), "hello")
   if err == nil || !strings.Contains(err.Error(), "not found") {
       t.Fatalf("expected not found error, got %v", err)
   }
}