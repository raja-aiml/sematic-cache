// Package cmd contains the cache server implementation.
package cmd

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/config"
	"github.com/raja-aiml/sematic-cache/observability"
	"github.com/raja-aiml/sematic-cache/openai"
	"github.com/raja-aiml/sematic-cache/server"
	"github.com/raja-aiml/sematic-cache/storage"
)

// Run starts the cache server with the provided context.
// It returns when the context is cancelled or an error occurs.
func Run(ctx context.Context) error {
	// Set Gin to release mode to avoid debug warnings
	gin.SetMode(gin.ReleaseMode)

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
			return fmt.Errorf("failed to load config: %w", err)
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
		shutdown, err = observability.Init(ctx, "cache-server", jaegerEndpoint)
		if err != nil {
			return fmt.Errorf("otel init: %w", err)
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
		return openaiClient.Embedding(ctx, p)
	})
	if err != nil {
		return fmt.Errorf("failed to create cache backend: %w", err)
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
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("server listening on %s", *addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- fmt.Errorf("HTTP server error: %w", err)
		}
		close(serverErr)
	}()

	// Wait for context cancellation or server error
	select {
	case <-ctx.Done():
		log.Println("Shutting down server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("server forced to shutdown: %w", err)
		}
		return nil
	case err := <-serverErr:
		return err
	}
}