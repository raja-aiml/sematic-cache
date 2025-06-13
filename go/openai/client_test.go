package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestComplete(t *testing.T) {
   handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
       w.Header().Set("Content-Type", "application/json")
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("bad request: %v", err)
		}
		if req["prompt"] != "hello" {
			t.Fatalf("expected prompt hello, got %v", req["prompt"])
		}
		resp := map[string]any{
			"choices": []map[string]string{{"text": "world"}},
		}
		json.NewEncoder(w).Encode(resp)
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	c := NewClient("test")
	c.SetBaseURL(server.URL)
	got, err := c.Complete(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if got != "world" {
		t.Fatalf("expected world, got %s", got)
	}
}

func TestEmbedding(t *testing.T) {
   handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
       w.Header().Set("Content-Type", "application/json")
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("bad request: %v", err)
		}
		input, ok := req["input"].([]interface{})
		if !ok || len(input) == 0 || input[0] != "hello" {
			t.Fatalf("expected input hello, got %v", req["input"])
		}
		resp := map[string]any{
			"data": []map[string]any{{"embedding": []float32{1, 2}}},
		}
		json.NewEncoder(w).Encode(resp)
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	c := NewClient("test")
	c.SetBaseURL(server.URL)
	got, err := c.Embedding(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embedding returned error: %v", err)
	}
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("unexpected embedding %v", got)
	}
}
