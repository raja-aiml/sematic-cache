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

// TestServerQuery ensures /query returns the correct answer for a given embedding.
func TestServerQuery(t *testing.T) {
   // build cache with two entries
   cache := core.NewCache(2,
       core.WithSimilarityFunc(core.InnerProduct),
   )
   cache.Set("x", []float32{1, 0}, "AnswerX")
   cache.Set("y", []float32{0, 1}, "AnswerY")
   srv := New(cache)
   ts := httptest.NewServer(srv)
   defer ts.Close()

   // query near x
   reqBody, _ := json.Marshal(map[string]interface{}{
       "embedding": []float32{0.9, 0.1},
   })
   resp, err := http.Post(ts.URL+"/query", "application/json", bytes.NewBuffer(reqBody))
   if err != nil || resp.StatusCode != http.StatusOK {
       t.Fatalf("query failed: %v %v", err, resp.Status)
   }
   var qr map[string]interface{}
   if err := json.NewDecoder(resp.Body).Decode(&qr); err != nil {
       t.Fatalf("decoding query response: %v", err)
   }
   if qr["answer"] != "AnswerX" {
       t.Errorf("expected AnswerX, got %v", qr["answer"])
   }
}

// TestServerTopK ensures /topk returns the top K results.
func TestServerTopK(t *testing.T) {
   cache := core.NewCache(3,
       core.WithSimilarityFunc(core.InnerProduct),
   )
   data := map[string][]float32{
       "a": {1, 0},
       "b": {0, 1},
       "c": {1, 1},
   }
   for k, v := range data {
       cache.Set(k, v, "Ans"+k)
   }
   srv := New(cache)
   ts := httptest.NewServer(srv)
   defer ts.Close()

   reqBody, _ := json.Marshal(map[string]interface{}{
       "embedding": []float32{1, 1},
       "k": 2,
   })
   resp, err := http.Post(ts.URL+"/topk", "application/json", bytes.NewBuffer(reqBody))
   if err != nil || resp.StatusCode != http.StatusOK {
       t.Fatalf("topk failed: %v %v", err, resp.Status)
   }
   var results []map[string]interface{}
   if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
       t.Fatalf("decode topk: %v", err)
   }
   if len(results) != 2 {
       t.Fatalf("expected 2 results, got %d", len(results))
   }
   // first should be 'c'
   if results[0]["prompt"] != "c" {
       t.Errorf("expected first prompt c, got %v", results[0]["prompt"])
   }
}

// TestServerHealth checks the /health endpoint.
func TestServerHealth(t *testing.T) {
   srv := New(core.NewCache(1))
   ts := httptest.NewServer(srv)
   defer ts.Close()
   resp, err := http.Get(ts.URL + "/health")
   if err != nil || resp.StatusCode != http.StatusOK {
       t.Fatalf("health failed: %v %v", err, resp.Status)
   }
}

// TestServerMetrics checks the /metrics endpoint.
func TestServerMetrics(t *testing.T) {
   cache := core.NewCache(1)
   // generate one hit and one miss
   cache.Set("p", []float32{1}, "a")
   cache.Get("p")
   cache.Get("x")
   srv := New(cache)
   ts := httptest.NewServer(srv)
   defer ts.Close()
   resp, err := http.Get(ts.URL + "/metrics")
   if err != nil || resp.StatusCode != http.StatusOK {
       t.Fatalf("metrics failed: %v %v", err, resp.Status)
   }
   var m map[string]float64
   if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
       t.Fatalf("decode metrics: %v", err)
   }
   if m["hits"] != 1 || m["misses"] != 1 {
       t.Errorf("expected hits=1 misses=1, got %v", m)
   }
}

func TestServerFlush(t *testing.T) {
	cache := core.NewCache(10)
	srv := New(cache)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	body, _ := json.Marshal(map[string]string{"prompt": "p", "answer": "a"})
	http.Post(ts.URL+"/set", "application/json", bytes.NewBuffer(body))
		http.Post(ts.URL+"/admin/flush", "application/json", nil)

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
