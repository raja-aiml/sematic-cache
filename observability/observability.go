package observability

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	api "go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Init configures OpenTelemetry with OTLP exporters for traces and metrics.
// It sends telemetry data to the OpenTelemetry Collector.
// Returns shutdown functions for traces and metrics.
func Init(ctx context.Context, service, endpoint string) (func(context.Context) error, error) {
	// Create resource with service information
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(service),
			semconv.ServiceVersionKey.String("1.0.0"),
		),
		resource.WithHost(),
		resource.WithProcess(),
		resource.WithTelemetrySDK(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Initialize trace provider
	traceShutdown, err := initTraceProvider(ctx, res, endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize trace provider: %w", err)
	}

	// Initialize metric provider
	metricShutdown, err := initMetricProvider(ctx, res, endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize metric provider: %w", err)
	}

	// Set global propagator
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// Combined shutdown function
	shutdown := func(ctx context.Context) error {
		var err error
		if traceErr := traceShutdown(ctx); traceErr != nil {
			err = fmt.Errorf("trace shutdown error: %w", traceErr)
		}
		if metricErr := metricShutdown(ctx); metricErr != nil {
			if err != nil {
				err = fmt.Errorf("%v; metric shutdown error: %w", err, metricErr)
			} else {
				err = fmt.Errorf("metric shutdown error: %w", metricErr)
			}
		}
		return err
	}

	return shutdown, nil
}

// initTraceProvider initializes the trace provider with OTLP exporter
func initTraceProvider(ctx context.Context, res *resource.Resource, endpoint string) (func(context.Context) error, error) {
	// Create gRPC connection to OTel Collector
	conn, err := grpc.DialContext(ctx, endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC connection: %w", err)
	}

	// Create OTLP trace exporter
	traceExporter, err := otlptrace.New(ctx, otlptracegrpc.NewClient(
		otlptracegrpc.WithGRPCConn(conn),
	))
	if err != nil {
		return nil, fmt.Errorf("failed to create trace exporter: %w", err)
	}

	// Create trace provider
	bsp := sdktrace.NewBatchSpanProcessor(traceExporter,
		sdktrace.WithBatchTimeout(time.Second),
	)
	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(bsp),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()), // Sample all traces in dev
	)

	otel.SetTracerProvider(tracerProvider)

	return tracerProvider.Shutdown, nil
}

// initMetricProvider initializes the metric provider with OTLP exporter
func initMetricProvider(ctx context.Context, res *resource.Resource, endpoint string) (func(context.Context) error, error) {
	// Create OTLP metric exporter
	metricExporter, err := otlpmetricgrpc.New(ctx,
		otlpmetricgrpc.WithEndpoint(endpoint),
		otlpmetricgrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create metric exporter: %w", err)
	}

	// Create metric provider
	meterProvider := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(
			metricExporter,
			sdkmetric.WithInterval(10*time.Second), // Export metrics every 10 seconds
		)),
	)

	otel.SetMeterProvider(meterProvider)

	return meterProvider.Shutdown, nil
}

// GetMeter returns a configured OpenTelemetry meter
func GetMeter(name string) api.Meter {
	return otel.Meter(name)
}

// GetTracer returns a configured OpenTelemetry tracer
func GetTracer(name string) trace.Tracer {
	return otel.Tracer(name)
}
