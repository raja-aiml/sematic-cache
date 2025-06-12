package server

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/raja-aiml/sematic-cache/go/core"
)

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
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	var req struct{ Prompt string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	val, ok := s.Cache.Get(req.Prompt)
	if !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(map[string]string{"answer": val})
}

func (s *Server) handleSet(w http.ResponseWriter, r *http.Request) {
	var req struct{ Prompt, Answer string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.Cache.Set(req.Prompt, req.Answer)
	w.WriteHeader(http.StatusCreated)
}

// Run starts the HTTP server.
func (s *Server) Run(addr string) error {
	log.Printf("server listening on %s", addr)
	return http.ListenAndServe(addr, s)
}
