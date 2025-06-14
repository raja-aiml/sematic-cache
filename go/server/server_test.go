package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/raja-aiml/sematic-cache/go/core"
)

func TestServerSetGet(t *testing.T) {
	cache := core.NewCache(10)
	srv := New(cache)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	// set value
	body, _ := json.Marshal(map[string]string{"prompt": "p", "answer": "a"})
	resp, err := http.Post(ts.URL+"/set", "application/json", bytes.NewBuffer(body))
	if err != nil || resp.StatusCode != http.StatusCreated {
		t.Fatalf("set failed: %v %v", err, resp.Status)
	}

	// get value
	body, _ = json.Marshal(map[string]string{"prompt": "p"})
	resp, err = http.Post(ts.URL+"/get", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	var data map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if data["answer"] != "a" {
		t.Fatalf("expected a, got %s", data["answer"])
	}
}

func TestServerFlush(t *testing.T) {
	cache := core.NewCache(10)
	srv := New(cache)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	body, _ := json.Marshal(map[string]string{"prompt": "p", "answer": "a"})
	http.Post(ts.URL+"/set", "application/json", bytes.NewBuffer(body))
	http.Post(ts.URL+"/flush", "application/json", nil)

	body, _ = json.Marshal(map[string]string{"prompt": "p"})
	resp, _ := http.Post(ts.URL+"/get", "application/json", bytes.NewBuffer(body))
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected not found after flush, got %v", resp.StatusCode)
	}
}

// Test that model metadata is stored and returned by the server.
func TestServerModelMetadata(t *testing.T) {
	cache := core.NewCache(10)
	srv := New(cache)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	// set with model metadata
	body, _ := json.Marshal(map[string]string{
		"prompt":    "p",
		"answer":    "a",
		"modelName": "gpt-3.5-turbo",
		"modelID":   "model-123",
	})
	resp, err := http.Post(ts.URL+"/set", "application/json", bytes.NewBuffer(body))
	if err != nil || resp.StatusCode != http.StatusCreated {
		t.Fatalf("set with metadata failed: %v %v", err, resp.Status)
	}

	// get and verify metadata
	body, _ = json.Marshal(map[string]string{"prompt": "p"})
	resp, err = http.Post(ts.URL+"/get", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	var data map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if data["answer"] != "a" {
		t.Errorf("expected answer a, got %s", data["answer"])
	}
	if data["modelName"] != "gpt-3.5-turbo" {
		t.Errorf("expected modelName gpt-3.5-turbo, got %s", data["modelName"])
	}
	if data["modelID"] != "model-123" {
		t.Errorf("expected modelID model-123, got %s", data["modelID"])
	}
}
