// Package logger provides colored logging functionality for DevOps tools
package logger

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/fatih/color"
)

// Level represents the logging level
type Level int

const (
	// DebugLevel logs everything
	DebugLevel Level = iota
	// InfoLevel logs info, warnings, and errors
	InfoLevel
	// WarnLevel logs warnings and errors
	WarnLevel
	// ErrorLevel logs only errors
	ErrorLevel
)

// Logger provides logging functionality with color support
type Logger struct {
	level  Level
	output io.Writer
	mu     sync.Mutex

	// Color functions
	infoColor    *color.Color
	warnColor    *color.Color
	errorColor   *color.Color
	debugColor   *color.Color
	successColor *color.Color
	failureColor *color.Color
}

// New creates a new logger instance
func New() *Logger {
	return NewWithOptions(InfoLevel, os.Stderr)
}

// NewWithOptions creates a new logger with custom options
func NewWithOptions(level Level, output io.Writer) *Logger {
	// Disable color if NO_COLOR is set or not a terminal
	if os.Getenv("NO_COLOR") != "" || !isTerminal() {
		color.NoColor = true
	}

	return &Logger{
		level:        level,
		output:       output,
		infoColor:    color.New(color.FgGreen),
		warnColor:    color.New(color.FgYellow),
		errorColor:   color.New(color.FgRed),
		debugColor:   color.New(color.FgCyan),
		successColor: color.New(color.FgGreen),
		failureColor: color.New(color.FgRed),
	}
}

// SetLevel sets the logging level
func (l *Logger) SetLevel(level Level) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// SetOutput sets the output writer
func (l *Logger) SetOutput(w io.Writer) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.output = w
}

// Info logs an info message
func (l *Logger) Info(format string, args ...interface{}) {
	if l.level <= InfoLevel {
		l.log(l.infoColor, "[INFO]", format, args...)
	}
}

// Warn logs a warning message
func (l *Logger) Warn(format string, args ...interface{}) {
	if l.level <= WarnLevel {
		l.log(l.warnColor, "[WARN]", format, args...)
	}
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	if l.level <= ErrorLevel {
		l.log(l.errorColor, "[ERROR]", format, args...)
	}
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	if l.level <= DebugLevel {
		l.log(l.debugColor, "[DEBUG]", format, args...)
	}
}

// Success logs a success message with checkmark
func (l *Logger) Success(format string, args ...interface{}) {
	if l.level <= InfoLevel {
		msg := fmt.Sprintf(format, args...)
		l.mu.Lock()
		defer l.mu.Unlock()
		fmt.Fprintf(l.output, "%s %s\n", l.successColor.Sprint("✅"), msg)
	}
}

// Failure logs a failure message with X mark
func (l *Logger) Failure(format string, args ...interface{}) {
	if l.level <= ErrorLevel {
		msg := fmt.Sprintf(format, args...)
		l.mu.Lock()
		defer l.mu.Unlock()
		fmt.Fprintf(l.output, "%s %s\n", l.failureColor.Sprint("❌"), msg)
	}
}

// Fatal logs an error message and exits
func (l *Logger) Fatal(format string, args ...interface{}) {
	l.Error(format, args...)
	os.Exit(1)
}

// log is the internal logging function
func (l *Logger) log(c *color.Color, prefix, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	l.mu.Lock()
	defer l.mu.Unlock()
	fmt.Fprintf(l.output, "%s %s\n", c.Sprint(prefix), msg)
}

// ParseLevel parses a string into a logging level
func ParseLevel(s string) (Level, error) {
	switch strings.ToLower(s) {
	case "debug":
		return DebugLevel, nil
	case "info":
		return InfoLevel, nil
	case "warn", "warning":
		return WarnLevel, nil
	case "error":
		return ErrorLevel, nil
	default:
		return InfoLevel, fmt.Errorf("unknown log level: %s", s)
	}
}

// isTerminal checks if the output is a terminal
func isTerminal() bool {
	if fileInfo, _ := os.Stderr.Stat(); (fileInfo.Mode() & os.ModeCharDevice) != 0 {
		return true
	}
	return false
}

// Default logger instance
var defaultLogger = New()

// Package-level functions using the default logger

// SetLevel sets the default logger's level
func SetLevel(level Level) {
	defaultLogger.SetLevel(level)
}

// Info logs an info message using the default logger
func Info(format string, args ...interface{}) {
	defaultLogger.Info(format, args...)
}

// Warn logs a warning message using the default logger
func Warn(format string, args ...interface{}) {
	defaultLogger.Warn(format, args...)
}

// Error logs an error message using the default logger
func Error(format string, args ...interface{}) {
	defaultLogger.Error(format, args...)
}

// Debug logs a debug message using the default logger
func Debug(format string, args ...interface{}) {
	defaultLogger.Debug(format, args...)
}

// Success logs a success message using the default logger
func Success(format string, args ...interface{}) {
	defaultLogger.Success(format, args...)
}

// Failure logs a failure message using the default logger
func Failure(format string, args ...interface{}) {
	defaultLogger.Failure(format, args...)
}

// Fatal logs an error message and exits using the default logger
func Fatal(format string, args ...interface{}) {
	defaultLogger.Fatal(format, args...)
}
