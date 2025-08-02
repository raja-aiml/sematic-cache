// Twelve-Factor App compliant server implementation
package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/raja-aiml/sematic-cache/internal/api/handlers"
	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/raja-aiml/sematic-cache/internal/embedding"
	"github.com/raja-aiml/sematic-cache/internal/storage"
	"github.com/raja-aiml/sematic-cache/pkg/agent"
)

func main() {
	// Factor XI: Logs - Structured logging to stdout
	setupLogging()

	// Factor III: Config - Load from environment
	cfg := config.LoadFromEnv()
	if err := cfg.Validate(); err != nil {
		slog.Error("Configuration validation failed", "error", err)
		os.Exit(1)
	}

	// Log configuration (redacted sensitive values)
	slog.Info("Starting semantic-cache server",
		"port", cfg.Port,
		"database_configured", cfg.DatabaseURL != "",
		"openai_configured", cfg.OpenAIAPIKey != "",
		"otel_configured", cfg.OTELEndpoint != "",
	)

	// Factor IX: Disposability - Handle signals for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Factor IV: Backing services - Database as attached resource
	if err := waitForDatabase(cfg.DatabaseURL); err != nil {
		slog.Error("Database connection failed", "error", err)
		os.Exit(1)
	}

	// Create embedding function if OpenAI is configured
	var embedFunc func(string) ([]float32, error)
	if cfg.OpenAIAPIKey != "" {
		client := embedding.NewClient(cfg.OpenAIAPIKey)
		if cfg.OpenAIBaseURL != "" {
			client.SetBaseURL(cfg.OpenAIBaseURL)
		}
		embedFunc = func(text string) ([]float32, error) {
			return client.Embedding(context.Background(), text)
		}
		slog.Info("OpenAI embeddings enabled", "model", cfg.OpenAIModel)
	} else {
		slog.Warn("OpenAI API key not configured, embeddings disabled")
	}

	// Create storage with twelve-factor config
	storeCfg := &config.Config{
		Storage: config.StorageConfig{
			DSN:                 cfg.DatabaseURL,
			SimilarityThreshold: cfg.SimilarityThreshold,
			PoolSize:            cfg.DatabaseMaxConnections,
			IndexLists:          cfg.VectorIndexLists,
		},
	}

	store, err := storage.NewVectorStore(storeCfg, embedFunc)
	if err != nil {
		slog.Error("Failed to create vector store", "error", err)
		os.Exit(1)
	}
	defer store.Close()

	// Create adapter for backward compatibility
	adapter := storage.NewCacheAdapter(store)

	// Factor VII: Port binding - Self-contained web server
	router := setupRouter(adapter, store, embedFunc, cfg)

	// Create HTTP server with timeouts
	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	serverErrors := make(chan error, 1)
	go func() {
		slog.Info("Server starting", "address", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErrors <- err
		}
	}()

	// Factor IX: Disposability - Graceful shutdown
	select {
	case err := <-serverErrors:
		slog.Error("Server failed to start", "error", err)
		os.Exit(1)

	case sig := <-sigChan:
		slog.Info("Shutdown signal received", "signal", sig)

		// Create shutdown context with timeout
		shutdownCtx, shutdownCancel := context.WithTimeout(
			context.Background(),
			time.Duration(cfg.ShutdownTimeout)*time.Second,
		)
		defer shutdownCancel()

		// Gracefully shutdown server
		if err := srv.Shutdown(shutdownCtx); err != nil {
			slog.Error("Server shutdown failed", "error", err)
			os.Exit(1)
		}

		slog.Info("Server shutdown complete")
	}
}

// setupLogging configures structured logging to stdout (Factor XI)
func setupLogging() {
	logLevel := slog.LevelInfo
	switch os.Getenv("LOG_LEVEL") {
	case "debug":
		logLevel = slog.LevelDebug
	case "warn":
		logLevel = slog.LevelWarn
	case "error":
		logLevel = slog.LevelError
	}

	var handler slog.Handler
	opts := &slog.HandlerOptions{Level: logLevel}

	if os.Getenv("LOG_FORMAT") == "json" {
		handler = slog.NewJSONHandler(os.Stdout, opts)
	} else {
		handler = slog.NewTextHandler(os.Stdout, opts)
	}

	logger := slog.New(handler)
	slog.SetDefault(logger)
}

// setupRouter configures HTTP routes with health checks
func setupRouter(cache storage.CacheBackend, store *storage.VectorStore, embedFunc func(string) ([]float32, error), cfg *config.EnvConfig) *gin.Engine {
	// Set Gin mode based on environment
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(structuredLoggingMiddleware())

	// Health checks (Factor IX: Disposability)
	router.GET(cfg.HealthCheckPath, healthCheck)
	router.GET(cfg.ReadinessCheckPath, readinessCheck(cache))

	// API routes - KISS principle: direct, simple routing
	cacheHandler := handlers.NewCacheHandler(cache)

	api := router.Group("/api/v1")
	{
		// Cache operations
		api.POST("/get", cacheHandler.HandleGet)
		api.POST("/set", cacheHandler.HandleSet)
		api.POST("/similar", cacheHandler.HandleSimilar)
		api.GET("/stats", cacheHandler.HandleStats)
		api.DELETE("/clear", cacheHandler.HandleClear)

		// Agent operations (if embedding is available)
		if embedFunc != nil {
			agentHandler := handlers.NewAgentHandler(store, embedFunc)
			api.POST("/agent/process", agentHandler.HandleAgentRequest)

			// A2A protocol endpoints
			a2aAdapter := agent.NewA2AMemoryAdapter(cache, cache, embedFunc)
			a2aHandler := handlers.NewA2AHandler(a2aAdapter)
			a2aHandler.RegisterRoutes(api)
		}
	}

	return router
}

// waitForDatabase implements retry logic for database connection
func waitForDatabase(dsn string) error {
	maxRetries := 30
	retryInterval := 2 * time.Second

	for i := 0; i < maxRetries; i++ {
		db, err := sql.Open("postgres", dsn)
		if err != nil {
			slog.Warn("Database connection attempt failed",
				"attempt", i+1,
				"max_retries", maxRetries,
				"error", err,
			)
			time.Sleep(retryInterval)
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err = db.PingContext(ctx)
		cancel()
		db.Close()

		if err == nil {
			slog.Info("Database connection established")
			return nil
		}

		slog.Warn("Database ping failed",
			"attempt", i+1,
			"max_retries", maxRetries,
			"error", err,
		)
		time.Sleep(retryInterval)
	}

	return fmt.Errorf("failed to connect to database after %d attempts", maxRetries)
}

// healthCheck returns basic health status
func healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy",
		"time":   time.Now().Unix(),
	})
}

// readinessCheck verifies all dependencies are ready
func readinessCheck(cache storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check cache/database connectivity
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		// Try a simple operation to verify database connection
		if _, err := cache.GetTopKByText(ctx, "health_check", 1); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not_ready",
				"error":  "database not accessible",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status": "ready",
			"time":   time.Now().Unix(),
		})
	}
}

// structuredLoggingMiddleware logs requests in structured format
func structuredLoggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Log request (Factor XI: Logs as event streams)
		latency := time.Since(start)
		clientIP := c.ClientIP()
		method := c.Request.Method
		statusCode := c.Writer.Status()

		if raw != "" {
			path = path + "?" + raw
		}

		// Structured log entry
		slog.Info("request",
			"client_ip", clientIP,
			"method", method,
			"path", path,
			"status", statusCode,
			"latency_ms", latency.Milliseconds(),
			"user_agent", c.Request.UserAgent(),
			"error", c.Errors.String(),
		)
	}
}
