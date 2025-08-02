package logger

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// WithTrace adds trace context to log fields
func WithTrace(ctx context.Context) Fields {
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() {
		return Fields{}
	}

	return Fields{
		"trace_id": span.SpanContext().TraceID().String(),
		"span_id":  span.SpanContext().SpanID().String(),
	}
}

// LogSpanError logs an error and records it in the span
func LogSpanError(ctx context.Context, err error, msg string, fields ...Fields) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.RecordError(err)
		span.SetAttributes(fieldsToAttributes(mergeFields(fields))...)
	}

	// Add trace context to log
	allFields := mergeFields(append(fields, WithTrace(ctx)))
	allFields["error"] = err.Error()
	Error(msg, allFields)
}

// LogSpanEvent logs an event to both logger and span
func LogSpanEvent(ctx context.Context, msg string, fields ...Fields) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.AddEvent(msg, trace.WithAttributes(fieldsToAttributes(mergeFields(fields))...))
	}

	// Add trace context to log
	allFields := mergeFields(append(fields, WithTrace(ctx)))
	Info(msg, allFields)
}

// Helper to convert fields to OpenTelemetry attributes
func fieldsToAttributes(fields Fields) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, len(fields))
	for k, v := range fields {
		switch val := v.(type) {
		case string:
			attrs = append(attrs, attribute.String(k, val))
		case int:
			attrs = append(attrs, attribute.Int(k, val))
		case int64:
			attrs = append(attrs, attribute.Int64(k, val))
		case float64:
			attrs = append(attrs, attribute.Float64(k, val))
		case bool:
			attrs = append(attrs, attribute.Bool(k, val))
		default:
			attrs = append(attrs, attribute.String(k, fmt.Sprintf("%v", val)))
		}
	}
	return attrs
}
