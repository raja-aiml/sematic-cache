// Package composite provides logging utilities for composite storage backends.
package composite

import (
	"log"
	"os"
)

// Logger provides structured logging for storage operations.
// It wraps the standard logger with operation context.
type Logger struct {
	backend string
	logger  *log.Logger
}

// NewLogger creates a logger for a specific storage backend.
func NewLogger(backend string) *Logger {
	return &Logger{
		backend: backend,
		logger:  log.New(os.Stderr, "", log.LstdFlags),
	}
}

// LogError logs an error with operation context.
func (l *Logger) LogError(operation string, key string, err error) {
	if err != nil {
		l.logger.Printf("[%s] %s failed for key %q: %v", l.backend, operation, key, err)
	}
}

// LogInfo logs informational messages.
func (l *Logger) LogInfo(operation string, message string) {
	l.logger.Printf("[%s] %s: %s", l.backend, operation, message)
}
