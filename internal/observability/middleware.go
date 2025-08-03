package observability

import (
	"fmt"
	"strconv"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
)

// TracingMiddleware creates a Gin middleware for distributed tracing
func TracingMiddleware(serviceName string) gin.HandlerFunc {
	tracer := otel.Tracer(serviceName)
	propagator := otel.GetTextMapPropagator()

	return func(c *gin.Context) {
		// Extract trace context from incoming request headers
		ctx := propagator.Extract(c.Request.Context(), propagation.HeaderCarrier(c.Request.Header))

		// Start a new span for this request
		spanName := fmt.Sprintf("%s %s", c.Request.Method, c.FullPath())
		if spanName == " " {
			spanName = fmt.Sprintf("%s %s", c.Request.Method, c.Request.URL.Path)
		}

		// Parse port
		port := 80
		if c.Request.URL.Port() != "" {
			if p, err := strconv.Atoi(c.Request.URL.Port()); err == nil {
				port = p
			}
		}

		ctx, span := tracer.Start(ctx, spanName,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				semconv.HTTPMethodKey.String(c.Request.Method),
				semconv.HTTPTargetKey.String(c.Request.URL.Path),
				semconv.HTTPRouteKey.String(c.FullPath()),
				semconv.HTTPURLKey.String(c.Request.URL.String()),
				attribute.String("http.host", c.Request.Host),
				semconv.HTTPSchemeKey.String(c.Request.URL.Scheme),
				semconv.HTTPRequestContentLengthKey.Int64(c.Request.ContentLength),
				semconv.NetHostNameKey.String(c.Request.Host),
				semconv.NetHostPortKey.Int(port),
				semconv.HTTPUserAgentKey.String(c.Request.UserAgent()),
				attribute.String("net.peer.ip", c.ClientIP()),
			),
		)
		defer span.End()

		// Store trace and span IDs in context for logging
		if spanCtx := span.SpanContext(); spanCtx.IsValid() {
			c.Set("trace_id", spanCtx.TraceID().String())
			c.Set("span_id", spanCtx.SpanID().String())
		}

		// Update request context with the new span context
		c.Request = c.Request.WithContext(ctx)

		// Process request
		c.Next()

		// Record response attributes
		span.SetAttributes(
			semconv.HTTPStatusCodeKey.Int(c.Writer.Status()),
			attribute.Int("http.response_size", c.Writer.Size()),
		)

		// Set span status based on HTTP status code
		if c.Writer.Status() >= 400 {
			span.SetStatus(codes.Error, fmt.Sprintf("HTTP %d", c.Writer.Status()))
		}

		// Record any errors
		if len(c.Errors) > 0 {
			for _, err := range c.Errors {
				span.RecordError(err.Err)
			}
		}
	}
}

// GetTraceID extracts the trace ID from the Gin context
func GetTraceID(c *gin.Context) string {
	if traceID, exists := c.Get("trace_id"); exists {
		if id, ok := traceID.(string); ok {
			return id
		}
	}
	return ""
}

// GetSpanID extracts the span ID from the Gin context
func GetSpanID(c *gin.Context) string {
	if spanID, exists := c.Get("span_id"); exists {
		if id, ok := spanID.(string); ok {
			return id
		}
	}
	return ""
}