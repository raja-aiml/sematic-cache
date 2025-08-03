package observability

import (
	"context"
	"fmt"
	"time"

	"github.com/raja-aiml/sematic-cache/internal/observability"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// Telemetry holds OpenTelemetry components
type Telemetry struct {
	Tracer   trace.Tracer
	Meter    metric.Meter
	Logger   *zap.Logger
	Shutdown func(context.Context) error

	// Metrics
	commandCounter  metric.Int64Counter
	commandDuration metric.Float64Histogram
	errorCounter    metric.Int64Counter
	dbConnections   metric.Int64UpDownCounter
	cacheHits       metric.Int64Counter
	cacheMisses     metric.Int64Counter
}

// InitTelemetry initializes OpenTelemetry
func InitTelemetry(ctx context.Context, serviceName, serviceVersion string, endpoint string, logger *zap.Logger) (*Telemetry, error) {
	// Use the existing observability setup from internal package
	shutdown, err := observability.Init(ctx, serviceName, endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to setup OpenTelemetry: %w", err)
	}

	// Get tracer and meter
	tracer := otel.Tracer(serviceName)
	meter := otel.Meter(serviceName)

	// Initialize metrics
	t := &Telemetry{
		Tracer:   tracer,
		Meter:    meter,
		Logger:   logger,
		Shutdown: shutdown,
	}

	// Create metrics
	if err := t.createMetrics(); err != nil {
		return nil, fmt.Errorf("failed to create metrics: %w", err)
	}

	return t, nil
}

// createMetrics creates all the metrics
func (t *Telemetry) createMetrics() error {
	var err error

	// Command execution counter
	t.commandCounter, err = t.Meter.Int64Counter(
		"cli.command.executions",
		metric.WithDescription("Number of CLI command executions"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Command execution duration
	t.commandDuration, err = t.Meter.Float64Histogram(
		"cli.command.duration",
		metric.WithDescription("Duration of CLI command execution"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return err
	}

	// Error counter
	t.errorCounter, err = t.Meter.Int64Counter(
		"cli.errors",
		metric.WithDescription("Number of errors encountered"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Database connections
	t.dbConnections, err = t.Meter.Int64UpDownCounter(
		"cli.db.connections",
		metric.WithDescription("Number of active database connections"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Cache hits
	t.cacheHits, err = t.Meter.Int64Counter(
		"cli.cache.hits",
		metric.WithDescription("Number of cache hits"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	// Cache misses
	t.cacheMisses, err = t.Meter.Int64Counter(
		"cli.cache.misses",
		metric.WithDescription("Number of cache misses"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return err
	}

	return nil
}

// StartCommand starts a trace for a command
func (t *Telemetry) StartCommand(ctx context.Context, command string, args []string) (context.Context, trace.Span) {
	ctx, span := t.Tracer.Start(ctx, fmt.Sprintf("cli.command.%s", command),
		trace.WithAttributes(
			attribute.String("command", command),
			attribute.StringSlice("args", args),
		),
	)

	// Record command execution
	t.commandCounter.Add(ctx, 1, metric.WithAttributes(
		attribute.String("command", command),
	))

	return ctx, span
}

// EndCommand ends a command trace
func (t *Telemetry) EndCommand(ctx context.Context, span trace.Span, start time.Time, command string, err error) {
	duration := time.Since(start).Milliseconds()

	// Record duration
	t.commandDuration.Record(ctx, float64(duration), metric.WithAttributes(
		attribute.String("command", command),
	))

	// Record error if any
	if err != nil {
		span.RecordError(err)
		t.errorCounter.Add(ctx, 1, metric.WithAttributes(
			attribute.String("command", command),
			attribute.String("error", err.Error()),
		))
	}

	span.End()
}

// RecordCacheHit records a cache hit
func (t *Telemetry) RecordCacheHit(ctx context.Context) {
	t.cacheHits.Add(ctx, 1)
}

// RecordCacheMiss records a cache miss
func (t *Telemetry) RecordCacheMiss(ctx context.Context) {
	t.cacheMisses.Add(ctx, 1)
}

// RecordDBConnection records database connection change
func (t *Telemetry) RecordDBConnection(ctx context.Context, delta int64) {
	t.dbConnections.Add(ctx, delta)
}
