// Package logger provides a colored logger implementation
package logger

import (
	"fmt"
	"log"
	"os"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Logger implements the interfaces.Logger interface
type Logger struct {
	level    LogLevel
	useColor bool
	logger   *log.Logger
}

// LogLevel represents the logging level
type LogLevel int

const (
	DebugLevel LogLevel = iota
	InfoLevel
	WarningLevel
	ErrorLevel
)

// Color codes for terminal output
const (
	Reset   = "\033[0m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Blue    = "\033[34m"
	Magenta = "\033[35m"
	Cyan    = "\033[36m"
	White   = "\033[37m"
	Gray    = "\033[90m"
)

// New creates a new logger instance
func New() interfaces.Logger {
	return NewWithOptions(InfoLevel, true)
}

// NewWithOptions creates a new logger with custom options
func NewWithOptions(level LogLevel, useColor bool) interfaces.Logger {
	return &Logger{
		level:    level,
		useColor: useColor,
		logger:   log.New(os.Stderr, "", 0),
	}
}

// Info logs an info message
func (l *Logger) Info(format string, args ...interface{}) {
	if l.level <= InfoLevel {
		l.log(Blue, "[INFO]", format, args...)
	}
}

// Success logs a success message
func (l *Logger) Success(format string, args ...interface{}) {
	if l.level <= InfoLevel {
		l.log(Green, "[SUCCESS]", format, args...)
	}
}

// Warning logs a warning message
func (l *Logger) Warning(format string, args ...interface{}) {
	if l.level <= WarningLevel {
		l.log(Yellow, "[WARN]", format, args...)
	}
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	if l.level <= ErrorLevel {
		l.log(Red, "[ERROR]", format, args...)
	}
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	if l.level <= DebugLevel {
		l.log(Gray, "[DEBUG]", format, args...)
	}
}

// log formats and prints the log message
func (l *Logger) log(color, prefix, format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)

	if l.useColor && isTerminal() {
		l.logger.Printf("%s%s%s %s", color, prefix, Reset, message)
	} else {
		l.logger.Printf("%s %s", prefix, message)
	}
}

// isTerminal checks if output is a terminal
func isTerminal() bool {
	fileInfo, err := os.Stderr.Stat()
	if err != nil {
		return false
	}
	return (fileInfo.Mode() & os.ModeCharDevice) != 0
}

// ParseLevel parses a string log level
func ParseLevel(level string) LogLevel {
	switch level {
	case "debug":
		return DebugLevel
	case "info":
		return InfoLevel
	case "warning", "warn":
		return WarningLevel
	case "error":
		return ErrorLevel
	default:
		return InfoLevel
	}
}
