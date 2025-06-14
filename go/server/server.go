// Package server exposes HTTP handlers for cache operations.

package server

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/raja-aiml/sematic-cache/go/core"
)

// setRequest represents the JSON payload for /set.
type setRequest struct {
	Prompt    string `json:"prompt"`
	Answer    string `json:"answer"`
	ModelName string `json:"modelName,omitempty"`
	ModelID   string `json:"modelID,omitempty"`
}

// getRequest represents the JSON payload for /get.
type getRequest struct {
	Prompt string `json:"prompt"`
}

// getResponse is the JSON response for /get.
type getResponse struct {
	Answer    string `json:"answer"`
	ModelName string `json:"modelName,omitempty"`
	ModelID   string `json:"modelID,omitempty"`
}

// Server provides HTTP access to the cache.
type Server struct {
	Cache *core.Cache
	mux   *http.ServeMux
}

// New returns a new Server instance with default routes.
func New(c *core.Cache) *Server {
	s := &Server{Cache: c, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("/get", s.handleGet)
	s.mux.HandleFunc("/set", s.handleSet)
   // existing endpoints
   s.mux.HandleFunc("/flush", s.handleFlush)
   // embedding-based lookup
   s.mux.HandleFunc("/query", s.handleQuery)
   // top-K similarity search
   s.mux.HandleFunc("/topk", s.handleTopK)
   // health check
   s.mux.HandleFunc("/health", s.handleHealth)
   // metrics (cache stats)
   s.mux.HandleFunc("/metrics", s.handleMetrics)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req getRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// retrieve answer and model metadata
	answer, ok := s.Cache.Get(req.Prompt)
	if !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	modelName, modelID, _ := s.Cache.GetModelInfo(req.Prompt)
	// respond with structured JSON
	resp := getResponse{
		Answer:    answer,
		ModelName: modelName,
		ModelID:   modelID,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// decode request payload
	var req setRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// store entry with optional model metadata
	s.Cache.SetWithModel(req.Prompt, nil, req.Answer, req.ModelName, req.ModelID)
	w.WriteHeader(http.StatusCreated)
}

func (s *Server) handleFlush(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.Cache.Flush()
	w.WriteHeader(http.StatusOK)
}

// queryRequest represents the JSON payload for /query
type queryRequest struct {
   Embedding []float32 `json:"embedding"`
}

// queryResponse is the JSON response for /query.
type queryResponse struct {
   Answer     string  `json:"answer"`
   ModelName  string  `json:"modelName,omitempty"`
   ModelID    string  `json:"modelID,omitempty"`
   Similarity float64 `json:"similarity,omitempty"`
}

// handleQuery performs a similarity lookup by embedding.
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
   if r.Method != http.MethodPost {
       http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
       return
   }
   var req queryRequest
   if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
       http.Error(w, err.Error(), http.StatusBadRequest)
       return
   }
   answer, ok := s.Cache.GetByEmbedding(req.Embedding)
   if !ok {
       http.Error(w, "not found", http.StatusNotFound)
       return
   }
   modelName, modelID, _ := s.Cache.GetModelInfo(answer)
   resp := queryResponse{Answer: answer, ModelName: modelName, ModelID: modelID}
   w.Header().Set("Content-Type", "application/json")
   json.NewEncoder(w).Encode(resp)
}

// topKRequest represents the JSON payload for /topk
type topKRequest struct {
   Embedding []float32 `json:"embedding"`
   K         int       `json:"k"`
}

// topKResponseItem is a single result in /topk
type topKResponseItem struct {
   Prompt     string  `json:"prompt"`
   Answer     string  `json:"answer"`
   Similarity float64 `json:"similarity"`
   ModelName  string  `json:"modelName,omitempty"`
   ModelID    string  `json:"modelID,omitempty"`
}

// handleTopK performs a top-K similarity search by embedding.
func (s *Server) handleTopK(w http.ResponseWriter, r *http.Request) {
   if r.Method != http.MethodPost {
       http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
       return
   }
   var req topKRequest
   if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
       http.Error(w, err.Error(), http.StatusBadRequest)
       return
   }
   results := s.Cache.GetTopKByEmbedding(req.Embedding, req.K)
   resp := make([]topKResponseItem, 0, len(results))
   for _, r0 := range results {
       resp = append(resp, topKResponseItem{
           Prompt:     r0.Prompt,
           Answer:     r0.Answer,
           Similarity: r0.Similarity,
           ModelName:  r0.ModelName,
           ModelID:    r0.ModelID,
       })
   }
   w.Header().Set("Content-Type", "application/json")
   json.NewEncoder(w).Encode(resp)
}

// handleHealth responds with a simple OK status.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
   if r.Method != http.MethodGet {
       http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
       return
   }
   w.WriteHeader(http.StatusOK)
   w.Write([]byte(`{"status":"ok"}`))
}

// handleMetrics returns cache hit/miss statistics as JSON.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
   if r.Method != http.MethodGet {
       http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
       return
   }
   hits, misses, rate := s.Cache.Stats()
   m := map[string]interface{}{"hits": hits, "misses": misses, "hitRate": rate}
   w.Header().Set("Content-Type", "application/json")
   json.NewEncoder(w).Encode(m)
}

// Run starts the HTTP server.
func (s *Server) Run(addr string) error {
	log.Printf("server listening on %s", addr)
	return http.ListenAndServe(addr, s)
}
