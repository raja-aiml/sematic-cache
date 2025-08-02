package embedding

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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

// TestChatStreamClose verifies that the streaming connection is closed after
// processing completes.
type mockBody struct {
	io.Reader
	closed bool
}

func (m *mockBody) Close() error {
	m.closed = true
	return nil
}

type mockRoundTripper struct {
	body *mockBody
}

func (m *mockRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	data := "data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\ndata: [DONE]\n\n"
	body := &mockBody{Reader: strings.NewReader(data)}
	m.body = body
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       body,
	}, nil
}

// TestChatStreamClose verifies that Close is called on the underlying stream.
func TestChatStreamClose(t *testing.T) {
	rt := &mockRoundTripper{}
	httpClient := &http.Client{Transport: rt}
	ai := openai.NewClient(option.WithAPIKey("test"), option.WithHTTPClient(httpClient))
	c := &Client{apiKey: "test", client: ai}

	out, errs, err := c.ChatStream(context.Background(), []ChatMessage{{Role: "user", Content: "hi"}}, ChatOptions{Model: "test"})
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if msg := <-out; msg != "hi" {
		t.Fatalf("unexpected message %q", msg)
	}
	for range out {
	}
	if err := <-errs; err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if !rt.body.closed {
		t.Fatalf("expected stream to be closed")
	}
}
