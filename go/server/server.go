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
	s.mux.HandleFunc("/flush", s.handleFlush)
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

// Run starts the HTTP server.
func (s *Server) Run(addr string) error {
	log.Printf("server listening on %s", addr)
	return http.ListenAndServe(addr, s)
}
