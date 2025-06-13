package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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

// TestCompleteNoChoices ensures Complete returns an error when no choices are returned.
func TestCompleteNoChoices(t *testing.T) {
   handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
       w.Header().Set("Content-Type", "application/json")
       json.NewEncoder(w).Encode(map[string]any{"choices": []any{}})
   })
   server := httptest.NewServer(handler)
   defer server.Close()

   c := NewClient("test")
   c.SetBaseURL(server.URL)
   _, err := c.Complete(context.Background(), "hello")
   if err == nil || !strings.Contains(err.Error(), "no choices returned") {
       t.Fatalf("expected no choices error, got %v", err)
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

// TestEmbeddingNoData ensures Embedding returns an error when no data is returned.
func TestEmbeddingNoData(t *testing.T) {
   handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
       w.Header().Set("Content-Type", "application/json")
       json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
   })
   server := httptest.NewServer(handler)
   defer server.Close()

   c := NewClient("test")
   c.SetBaseURL(server.URL)
   _, err := c.Embedding(context.Background(), "hello")
   if err == nil || !strings.Contains(err.Error(), "no embedding returned") {
       t.Fatalf("expected no embedding error, got %v", err)
   }
}

func TestClientConfig(t *testing.T) {
	c := NewClient("k1")
	if c.apiKey != "k1" {
		t.Fatalf("expected k1")
	}
	c.SetAPIKey("k2")
	if c.apiKey != "k2" {
		t.Fatalf("expected k2")
	}
	c.ConfigureAzure("k3", "http://x", "2023-09-01")
	if c.BaseURL != "http://x" || c.APIVersion != "2023-09-01" || c.apiKey != "k3" {
		t.Fatalf("azure config not applied")
	}
}
