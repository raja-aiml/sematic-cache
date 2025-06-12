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
	cache := core.NewCache()
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
