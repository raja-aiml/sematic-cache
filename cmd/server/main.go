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

	"github.com/raja-aiml/sematic-cache/config"
	"github.com/raja-aiml/sematic-cache/observability"
	"github.com/raja-aiml/sematic-cache/openai"
	"github.com/raja-aiml/sematic-cache/server"
	"github.com/raja-aiml/sematic-cache/storage"
)

func main() {
	// Path to YAML configuration file; empty disables config loading
	configPath := flag.String("config", "", "path to YAML configuration file (empty to skip)")
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

	// Initialize OpenTelemetry/Jaeger if endpoint is provided; otherwise no-op
	var shutdown func(context.Context) error
	jaegerEndpoint := os.Getenv("JAEGER_ENDPOINT")
	if jaegerEndpoint != "" {
		var err error
		shutdown, err = observability.Init(context.Background(), "cache-server", jaegerEndpoint)
		if err != nil {
			log.Fatalf("otel init: %v", err)
		}
	} else {
		shutdown = func(context.Context) error { return nil }
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
	// Create cache backend using factory
	cache, err := storage.NewBackend(cfg, func(p string) ([]float32, error) {
		return openaiClient.Embedding(context.Background(), p)
	})
	if err != nil {
		log.Fatalf("failed to create cache backend: %v", err)
	}
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
