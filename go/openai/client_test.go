package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestComplete(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("bad request: %v", err)
		}
		if req.Prompt != "hello" {
			t.Fatalf("expected prompt hello, got %s", req.Prompt)
		}
		resp := CompletionResponse{Choices: []struct {
			Text string `json:"text"`
		}{
			{Text: "world"},
		}}
		json.NewEncoder(w).Encode(resp)
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	c := NewClient("test")
	c.BaseURL = server.URL
	got, err := c.Complete("hello")
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if got != "world" {
		t.Fatalf("expected world, got %s", got)
	}
}
