package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/raja-aiml/sematic-cache/internal/database"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"github.com/raja-aiml/sematic-cache/internal/observability"
	"github.com/raja-aiml/sematic-cache/internal/server"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// Run starts the semantic cache server with all orchestration
func Run() error {
	observability.SetupLogging()

	cfg := config.LoadFromEnv()
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	server.LogServerConfig(cfg)

	if err := database.WaitForDatabase(cfg.DatabaseURL); err != nil {
		return fmt.Errorf("database connection failed: %w", err)
	}

	store, err := server.CreateStorage(cfg)
	if err != nil {
		return fmt.Errorf("failed to create storage: %w", err)
	}
	defer store.Close()

	adapter := storage.NewCacheAdapter(store)
	router := server.SetupRouter(adapter, cfg)

	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return runWithGracefulShutdown(srv, cfg.ShutdownTimeout)
}

// runWithGracefulShutdown handles application lifecycle and shutdown
func runWithGracefulShutdown(srv *http.Server, shutdownTimeout int) error {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	serverErrors := make(chan error, 1)
	go func() {
		logger.Info("Server starting", logger.Fields{"address": srv.Addr})
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErrors <- err
		}
	}()

	select {
	case err := <-serverErrors:
		return fmt.Errorf("server failed to start: %w", err)

	case sig := <-sigChan:
		logger.Info("Shutdown signal received", logger.Fields{"signal": sig.String()})

		ctx, cancel := context.WithTimeout(
			context.Background(),
			time.Duration(shutdownTimeout)*time.Second,
		)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			return fmt.Errorf("server shutdown failed: %w", err)
		}

		logger.Info("Server shutdown complete")
		return nil
	}
}
