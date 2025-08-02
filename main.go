package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raja-aiml/sematic-cache/cmd"
)

func main() {
	// Create context that listens for interrupt signals
	ctx, stop := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Run server with context
	errCh := make(chan error, 1)
	go func() {
		// Always run instrumented version - it's production ready
		// If OTEL_EXPORTER_OTLP_ENDPOINT is not set, tracing will be no-op
		errCh <- cmd.RunInstrumented(ctx)
	}()

	select {
	case <-ctx.Done():
		log.Println("Shutdown signal received")
		// Graceful shutdown with timeout
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		// Wait for server to shutdown or timeout
		select {
		case err := <-errCh:
			if err != nil {
				log.Fatalf("Server error during shutdown: %v", err)
			}
			log.Println("Server shutdown gracefully")
		case <-shutdownCtx.Done():
			log.Fatal("Forced shutdown after timeout")
		}
	case err := <-errCh:
		if err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}
}
