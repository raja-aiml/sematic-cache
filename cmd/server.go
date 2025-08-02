// Package cmd contains the instrumented cache server implementation with full observability.
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
	
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

// RunInstrumented starts the cache server with full observability instrumentation.
func RunInstrumented(ctx context.Context) error {
	// Set Gin to release mode to avoid debug warnings
	gin.SetMode(gin.ReleaseMode)

	// Path to YAML configuration file
	configPath := flag.String("config", "", "path to YAML configuration file")
	addr := flag.String("address", ":8080", "server address")
	flag.Parse()

	// Load configuration
	var cfg *config.Config
	if *configPath != "" {
		var err error
		cfg, err = config.LoadConfig(*configPath)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}
	}
	if cfg != nil && cfg.Server.Address != "" {
		*addr = cfg.Server.Address
	}

	// Initialize OpenTelemetry
	shutdown, err := initOTLP(ctx)
	if err != nil {
		return fmt.Errorf("failed to initialize OpenTelemetry: %w", err)
	}
	defer shutdown(context.Background())

	// Initialize custom metrics
	if err := observability.InitMetrics(); err != nil {
		return fmt.Errorf("failed to initialize metrics: %w", err)
	}

	// Build OpenAI client with tracing
	apiKey := os.Getenv("OPENAI_API_KEY")
	if cfg != nil && cfg.OpenAI.APIKey != "" {
		apiKey = cfg.OpenAI.APIKey
	}
	openaiClient := openai.NewClient(apiKey)
	if cfg != nil && cfg.OpenAI.BaseURL != "" {
		openaiClient.SetBaseURL(cfg.OpenAI.BaseURL)
	}

	// Wrap embedding function with tracing
	embedFunc := func(prompt string) ([]float32, error) {
		ctx, span := observability.TraceEmbeddingGeneration(context.Background(), prompt, "text-embedding-3-small")
		defer span.End()

		start := time.Now()
		embedding, err := openaiClient.Embedding(ctx, prompt)
		duration := time.Since(start)

		observability.RecordEmbeddingGeneration(ctx, duration, "text-embedding-3-small", len(embedding))
		
		if err != nil {
			span.RecordError(err)
		}
		return embedding, err
	}

	// Create cache backend with instrumentation
	cache, err := storage.NewBackend(cfg, embedFunc)
	if err != nil {
		return fmt.Errorf("failed to create cache backend: %w", err)
	}

	// Create instrumented server
	srv := server.New(cache)
	
	// Add OpenTelemetry middleware to Gin
	router := srv.Router()
	router.Use(otelgin.Middleware("semantic-cache",
		otelgin.WithTracerProvider(otel.GetTracerProvider()),
		otelgin.WithPropagators(propagation.TraceContext{}),
	))

	// Add custom middleware for cache metrics
	router.Use(cacheMetricsMiddleware())

	httpSrv := &http.Server{
		Addr:         *addr,
		Handler:      router,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("Instrumented server listening on %s", *addr)
		log.Printf("SigNoz UI available at: http://localhost:3301")
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- fmt.Errorf("HTTP server error: %w", err)
		}
		close(serverErr)
	}()

	// Wait for shutdown
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

// initOTLP initializes OpenTelemetry with OTLP exporters
func initOTLP(ctx context.Context) (func(context.Context) error, error) {
	// Get OTLP endpoint from environment
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// No endpoint configured - use no-op provider
		log.Println("OTEL_EXPORTER_OTLP_ENDPOINT not set, running without telemetry export")
		// Just return early - the default global providers are no-op
		return func(context.Context) error { return nil }, nil
	}

	// Create resource
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String("semantic-cache"),
			semconv.ServiceVersionKey.String("1.0.0"),
			attribute.String("environment", "development"),
		),
		resource.WithHost(),
		resource.WithProcess(),
		resource.WithOS(),
		resource.WithContainer(),
	)
	if err != nil {
		return nil, err
	}

	// Parse endpoint to remove scheme if present
	if len(endpoint) > 7 && endpoint[:7] == "http://" {
		endpoint = endpoint[7:]
	} else if len(endpoint) > 8 && endpoint[:8] == "https://" {
		endpoint = endpoint[8:]
	}
	
	// Create trace exporter
	traceExporter, err := otlptrace.New(ctx,
		otlptracehttp.NewClient(
			otlptracehttp.WithEndpoint(endpoint),
			otlptracehttp.WithInsecure(),
		),
	)
	if err != nil {
		return nil, err
	}

	// Create trace provider
	tp := trace.NewTracerProvider(
		trace.WithBatcher(traceExporter),
		trace.WithResource(res),
		trace.WithSampler(trace.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})

	// Metrics are optional - Jaeger doesn't support them
	// Only create metric exporter if explicitly enabled
	var mp *metric.MeterProvider
	if os.Getenv("ENABLE_METRICS") == "true" {
		// Create metric exporter
		metricExporter, err := otlpmetrichttp.New(ctx,
			otlpmetrichttp.WithEndpoint(endpoint),
			otlpmetrichttp.WithInsecure(),
		)
		if err != nil {
			log.Printf("Warning: Failed to create metric exporter: %v", err)
		} else {
			// Create metric provider
			mp = metric.NewMeterProvider(
				metric.WithReader(
					metric.NewPeriodicReader(metricExporter,
						metric.WithInterval(10*time.Second),
					),
				),
				metric.WithResource(res),
			)
			otel.SetMeterProvider(mp)
		}
	}

	// Return shutdown function
	return func(ctx context.Context) error {
		if err := tp.Shutdown(ctx); err != nil {
			return err
		}
		if mp != nil {
			return mp.Shutdown(ctx)
		}
		return nil
	}, nil
}

// cacheMetricsMiddleware adds cache-specific metrics to requests
func cacheMetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		// Process request
		c.Next()
		
		// Record metrics
		duration := time.Since(start)
		observability.RecordCacheOperation(c.Request.Context(),
			c.Request.Method+" "+c.Request.URL.Path,
			duration,
			c.Writer.Status() < 400,
		)
	}
}