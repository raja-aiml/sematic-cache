// Binary server runs the cache server.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raja-aiml/sematic-cache/go/config"
	"github.com/raja-aiml/sematic-cache/go/core"
	"github.com/raja-aiml/sematic-cache/go/observability"
	"github.com/raja-aiml/sematic-cache/go/openai"
	"github.com/raja-aiml/sematic-cache/go/server"
)

func main() {
	configPath := flag.String("config", "cache_config.yml", "path to YAML configuration file")
	addr := flag.String("address", ":8080", "server address (overrides config)")
	flag.Parse()

	// Load configuration if provided
	var cfg *config.Config
	if *configPath != "" {
		var err error
		cfg, err = config.LoadConfig(*configPath)
		if err != nil {
			log.Fatalf("failed to load config: %v", err)
		}
	}
	// Override address from config
	if cfg != nil && cfg.Server.Address != "" {
		*addr = cfg.Server.Address
	}

	shutdown, err := observability.Init(context.Background(), "cache-server", os.Getenv("JAEGER_ENDPOINT"))
	if err != nil {
		log.Fatalf("otel init: %v", err)
	}
	defer shutdown(context.Background())

	// Build OpenAI client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if cfg != nil && cfg.OpenAI.APIKey != "" {
		apiKey = cfg.OpenAI.APIKey
	}
	openaiClient := openai.NewClient(apiKey)
	if cfg != nil && cfg.OpenAI.BaseURL != "" {
		openaiClient.SetBaseURL(cfg.OpenAI.BaseURL)
	}
	if cfg != nil && cfg.OpenAI.APIVersion != "" {
		openaiClient.APIVersion = cfg.OpenAI.APIVersion
	}
	// Configure cache
	cap := 100
	if cfg != nil && cfg.Cache.Capacity > 0 {
		cap = cfg.Cache.Capacity
	}
	opts := []core.Option{
		core.WithEmbeddingFunc(func(p string) ([]float32, error) {
			return openaiClient.Embedding(context.Background(), p)
		}),
	}
	if cfg != nil {
		if cfg.Cache.EvictionPolicy != "" {
			opts = append(opts, core.WithEvictionPolicy(cfg.Cache.EvictionPolicy))
		}
		if ttl := cfg.TTLDuration(); ttl > 0 {
			opts = append(opts, core.WithTTL(ttl))
		}
		if cfg.Cache.MinSimilarity != 0 {
			opts = append(opts, core.WithMinSimilarity(cfg.Cache.MinSimilarity))
		}
	}
	cache := core.NewCache(cap, opts...)
	srv := server.New(cache)
	httpSrv := &http.Server{
		Addr:         *addr,
		Handler:      srv,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in background
	go func() {
		log.Printf("server listening on %s", *addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}
}
