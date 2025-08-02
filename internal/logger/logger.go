package logger

import (
	"context"
	"os"
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	instance *zap.Logger
	once     sync.Once
)

// Fields type for structured logging
type Fields map[string]interface{}

// Merge combines this Fields with another, with other taking precedence
func (f Fields) Merge(other Fields) Fields {
	result := make(Fields, len(f)+len(other))
	for k, v := range f {
		result[k] = v
	}
	for k, v := range other {
		result[k] = v
	}
	return result
}

// Initialize sets up the logger (called automatically on first use)
func Initialize() {
	once.Do(func() {
		instance = buildLogger()
	})
}

// buildLogger creates the zap logger based on environment
func buildLogger() *zap.Logger {
	var config zap.Config

	// Choose base config
	if os.Getenv("ENVIRONMENT") == "production" {
		config = zap.NewProductionConfig()
		config.EncoderConfig.TimeKey = "timestamp"
		config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	} else {
		config = zap.NewDevelopmentConfig()
		config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}

	// Override with environment settings
	setLogLevel(&config)
	setLogFormat(&config)

	logger, err := config.Build(
		zap.AddCaller(),
		zap.AddCallerSkip(1),
		zap.AddStacktrace(zapcore.ErrorLevel),
	)
	if err != nil {
		// Fallback to no-op logger if config fails
		return zap.NewNop()
	}

	return logger
}

// setLogLevel configures the log level from environment
func setLogLevel(config *zap.Config) {
	levelMap := map[string]zapcore.Level{
		"debug": zap.DebugLevel,
		"info":  zap.InfoLevel,
		"warn":  zap.WarnLevel,
		"error": zap.ErrorLevel,
	}

	if level, ok := levelMap[os.Getenv("LOG_LEVEL")]; ok {
		config.Level.SetLevel(level)
	}
}

// setLogFormat configures the output format from environment
func setLogFormat(config *zap.Config) {
	if os.Getenv("LOG_FORMAT") == "json" {
		config.Encoding = "json"
	}
}

// get returns the logger instance
func get() *zap.Logger {
	if instance == nil {
		Initialize()
	}
	return instance
}

// WithContext returns a logger with context fields
func WithContext(ctx context.Context) *zap.Logger {
	logger := get()

	// Extract standard context values
	if requestID := ctx.Value("request_id"); requestID != nil {
		logger = logger.With(zap.Any("request_id", requestID))
	}
	if userID := ctx.Value("user_id"); userID != nil {
		logger = logger.With(zap.Any("user_id", userID))
	}

	return logger
}

// WithFields returns a logger with additional fields
func WithFields(fields Fields) *zap.Logger {
	return get().With(toZapFields(fields)...)
}

// WithError returns a logger with an error field
func WithError(err error) *zap.Logger {
	return get().With(zap.Error(err))
}

// Package-level logging functions

// Debug logs a debug message
func Debug(msg string, fields ...Fields) {
	get().Debug(msg, toZapFields(mergeFields(fields))...)
}

// Info logs an info message
func Info(msg string, fields ...Fields) {
	get().Info(msg, toZapFields(mergeFields(fields))...)
}

// Warn logs a warning message
func Warn(msg string, fields ...Fields) {
	get().Warn(msg, toZapFields(mergeFields(fields))...)
}

// Error logs an error message
func Error(msg string, fields ...Fields) {
	get().Error(msg, toZapFields(mergeFields(fields))...)
}

// Fatal logs a fatal message and exits
func Fatal(msg string, fields ...Fields) {
	get().Fatal(msg, toZapFields(mergeFields(fields))...)
}

// Formatted logging functions

// Debugf logs a formatted debug message
func Debugf(format string, args ...interface{}) {
	get().Sugar().Debugf(format, args...)
}

// Infof logs a formatted info message
func Infof(format string, args ...interface{}) {
	get().Sugar().Infof(format, args...)
}

// Warnf logs a formatted warning message
func Warnf(format string, args ...interface{}) {
	get().Sugar().Warnf(format, args...)
}

// Errorf logs a formatted error message
func Errorf(format string, args ...interface{}) {
	get().Sugar().Errorf(format, args...)
}

// Fatalf logs a formatted fatal message and exits
func Fatalf(format string, args ...interface{}) {
	get().Sugar().Fatalf(format, args...)
}

// Sync flushes any buffered log entries
func Sync() error {
	if instance != nil {
		return instance.Sync()
	}
	return nil
}

// Helper functions

func toZapFields(fields Fields) []zap.Field {
	result := make([]zap.Field, 0, len(fields))
	for k, v := range fields {
		result = append(result, zap.Any(k, v))
	}
	return result
}

func mergeFields(fieldsList []Fields) Fields {
	if len(fieldsList) == 0 {
		return nil
	}

	merged := make(Fields)
	for _, fields := range fieldsList {
		for k, v := range fields {
			merged[k] = v
		}
	}
	return merged
}
