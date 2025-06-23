package logger

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLogger(t *testing.T) {
	tests := []struct {
		name     string
		level    Level
		logFunc  func(*Logger, string, ...interface{})
		message  string
		expected string
		shouldLog bool
	}{
		{
			name:     "info at info level",
			level:    InfoLevel,
			logFunc:  (*Logger).Info,
			message:  "test info",
			expected: "[INFO] test info",
			shouldLog: true,
		},
		{
			name:     "debug at info level",
			level:    InfoLevel,
			logFunc:  (*Logger).Debug,
			message:  "test debug",
			expected: "[DEBUG] test debug",
			shouldLog: false,
		},
		{
			name:     "debug at debug level",
			level:    DebugLevel,
			logFunc:  (*Logger).Debug,
			message:  "test debug",
			expected: "[DEBUG] test debug",
			shouldLog: true,
		},
		{
			name:     "warn at warn level",
			level:    WarnLevel,
			logFunc:  (*Logger).Warn,
			message:  "test warn",
			expected: "[WARN] test warn",
			shouldLog: true,
		},
		{
			name:     "error at error level",
			level:    ErrorLevel,
			logFunc:  (*Logger).Error,
			message:  "test error",
			expected: "[ERROR] test error",
			shouldLog: true,
		},
		{
			name:     "info at error level",
			level:    ErrorLevel,
			logFunc:  (*Logger).Info,
			message:  "test info",
			expected: "[INFO] test info",
			shouldLog: false,
		},
	}

	// Force disable colors for testing
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			logger := NewWithOptions(tt.level, &buf)
			
			tt.logFunc(logger, tt.message)
			
			output := buf.String()
			if tt.shouldLog {
				assert.Contains(t, output, tt.expected)
			} else {
				assert.Empty(t, output)
			}
		})
	}
}

func TestLoggerSpecialMessages(t *testing.T) {
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	var buf bytes.Buffer
	logger := NewWithOptions(InfoLevel, &buf)

	t.Run("success message", func(t *testing.T) {
		buf.Reset()
		logger.Success("Operation completed")
		output := buf.String()
		assert.Contains(t, output, "✅")
		assert.Contains(t, output, "Operation completed")
	})

	t.Run("failure message", func(t *testing.T) {
		buf.Reset()
		logger.Failure("Operation failed")
		output := buf.String()
		assert.Contains(t, output, "❌")
		assert.Contains(t, output, "Operation failed")
	})
}

func TestLoggerFormatting(t *testing.T) {
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	var buf bytes.Buffer
	logger := NewWithOptions(InfoLevel, &buf)

	logger.Info("Hello %s, you have %d messages", "World", 5)
	output := buf.String()
	assert.Contains(t, output, "Hello World, you have 5 messages")
}

func TestParseLevel(t *testing.T) {
	tests := []struct {
		input    string
		expected Level
		hasError bool
	}{
		{"debug", DebugLevel, false},
		{"DEBUG", DebugLevel, false},
		{"info", InfoLevel, false},
		{"INFO", InfoLevel, false},
		{"warn", WarnLevel, false},
		{"warning", WarnLevel, false},
		{"error", ErrorLevel, false},
		{"ERROR", ErrorLevel, false},
		{"invalid", InfoLevel, true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			level, err := ParseLevel(tt.input)
			if tt.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, level)
			}
		})
	}
}

func TestPackageLevelFunctions(t *testing.T) {
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	// Create a buffer to capture output
	var buf bytes.Buffer
	defaultLogger.SetOutput(&buf)
	defaultLogger.SetLevel(DebugLevel)

	tests := []struct {
		name     string
		logFunc  func(string, ...interface{})
		message  string
		expected string
	}{
		{"Info", Info, "test info", "[INFO] test info"},
		{"Warn", Warn, "test warn", "[WARN] test warn"},
		{"Error", Error, "test error", "[ERROR] test error"},
		{"Debug", Debug, "test debug", "[DEBUG] test debug"},
		{"Success", Success, "test success", "✅ test success"},
		{"Failure", Failure, "test failure", "❌ test failure"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf.Reset()
			tt.logFunc(tt.message)
			output := buf.String()
			assert.True(t, strings.Contains(output, tt.expected))
		})
	}
}

func TestSetLevel(t *testing.T) {
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	var buf bytes.Buffer
	logger := NewWithOptions(InfoLevel, &buf)

	// Should not log debug at info level
	logger.Debug("should not appear")
	assert.Empty(t, buf.String())

	// Change to debug level
	logger.SetLevel(DebugLevel)
	logger.Debug("should appear")
	assert.Contains(t, buf.String(), "should appear")
}

func TestConcurrency(t *testing.T) {
	os.Setenv("NO_COLOR", "true")
	defer os.Unsetenv("NO_COLOR")

	var buf bytes.Buffer
	logger := NewWithOptions(InfoLevel, &buf)

	// Test concurrent logging
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			logger.Info("Message from goroutine %d", id)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Check that all messages were logged
	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	require.Equal(t, 10, len(lines))
}