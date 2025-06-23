package logger

import (
	"bytes"
	"io"
)

// LogWriter wraps a Logger to implement io.Writer
type LogWriter struct {
	logger *Logger
	level  Level
}

// NewLogWriter creates a new LogWriter
func NewLogWriter(logger *Logger, level Level) io.Writer {
	return &LogWriter{
		logger: logger,
		level:  level,
	}
}

// Write implements io.Writer
func (w *LogWriter) Write(p []byte) (n int, err error) {
	// Remove trailing newline if present
	msg := string(bytes.TrimRight(p, "\n"))
	
	if msg == "" {
		return len(p), nil
	}
	
	switch w.level {
	case DebugLevel:
		w.logger.Debug("%s", msg)
	case InfoLevel:
		w.logger.Info("%s", msg)
	case WarnLevel:
		w.logger.Warn("%s", msg)
	case ErrorLevel:
		w.logger.Error("%s", msg)
	}
	
	return len(p), nil
}