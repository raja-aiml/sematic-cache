package framework

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewSimpleLogger(t *testing.T) {
	tests := []struct {
		name    string
		prefix  string
		verbose bool
	}{
		{
			name:    "with_prefix_verbose",
			prefix:  "TEST",
			verbose: true,
		},
		{
			name:    "without_prefix",
			prefix:  "",
			verbose: false,
		},
		{
			name:    "custom_prefix_non_verbose",
			prefix:  "CUSTOM",
			verbose: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := NewSimpleLogger(tt.prefix, tt.verbose)
			assert.NotNil(t, logger)
			assert.Equal(t, tt.prefix, logger.prefix)
			assert.Equal(t, tt.verbose, logger.verbose)
		})
	}
}

func TestSimpleLogger_Info(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewSimpleLogger("TEST", true)
	logger.Info("test info message", "key1", "value1", "key2", 42)

	output := buf.String()
	assert.Contains(t, output, "[TEST]")
	assert.Contains(t, output, "[INFO]")
	assert.Contains(t, output, "test info message")
	assert.Contains(t, output, "key1=value1")
	assert.Contains(t, output, "key2=42")
}

func TestSimpleLogger_Debug(t *testing.T) {
	tests := []struct {
		name         string
		verbose      bool
		expectOutput bool
	}{
		{
			name:         "verbose_mode",
			verbose:      true,
			expectOutput: true,
		},
		{
			name:         "non_verbose_mode",
			verbose:      false,
			expectOutput: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			log.SetOutput(&buf)
			defer log.SetOutput(nil)

			logger := NewSimpleLogger("DEBUG_TEST", tt.verbose)
			logger.Debug("debug message", "debug", true)

			output := buf.String()
			if tt.expectOutput {
				assert.Contains(t, output, "[DEBUG]")
				assert.Contains(t, output, "debug message")
				assert.Contains(t, output, "debug=true")
			} else {
				assert.Empty(t, output)
			}
		})
	}
}

func TestSimpleLogger_Error(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewSimpleLogger("ERROR_TEST", false)
	logger.Error("error occurred", "error", "file not found", "code", 404)

	output := buf.String()
	assert.Contains(t, output, "[ERROR_TEST]")
	assert.Contains(t, output, "[ERROR]")
	assert.Contains(t, output, "error occurred")
	assert.Contains(t, output, "error=file not found")
	assert.Contains(t, output, "code=404")
}

func TestSimpleLogger_Warn(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewSimpleLogger("", true)
	logger.Warn("warning message", "level", "high", "retry", 3)

	output := buf.String()
	assert.Contains(t, output, "[WARN]")
	assert.Contains(t, output, "warning message")
	assert.Contains(t, output, "level=high")
	assert.Contains(t, output, "retry=3")
	assert.NotContains(t, output, "[]") // No empty prefix brackets
}

func TestSimpleLogger_log(t *testing.T) {
	tests := []struct {
		name     string
		prefix   string
		level    string
		msg      string
		fields   []interface{}
		expected []string
	}{
		{
			name:     "basic_log",
			prefix:   "PREFIX",
			level:    "INFO",
			msg:      "basic message",
			fields:   []interface{}{},
			expected: []string{"[PREFIX]", "[INFO]", "basic message"},
		},
		{
			name:     "log_with_fields",
			prefix:   "TEST",
			level:    "DEBUG",
			msg:      "debug info",
			fields:   []interface{}{"key", "value", "number", 123},
			expected: []string{"[TEST]", "[DEBUG]", "debug info", "key=value", "number=123"},
		},
		{
			name:     "log_with_odd_fields",
			prefix:   "ODD",
			level:    "WARN",
			msg:      "odd fields",
			fields:   []interface{}{"key1", "value1", "orphan"},
			expected: []string{"[ODD]", "[WARN]", "odd fields", "key1=value1"},
		},
		{
			name:     "no_prefix",
			prefix:   "",
			level:    "ERROR",
			msg:      "no prefix message",
			fields:   []interface{}{"error", "critical"},
			expected: []string{"[ERROR]", "no prefix message", "error=critical"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			log.SetOutput(&buf)
			defer log.SetOutput(nil)

			logger := &SimpleLogger{prefix: tt.prefix, verbose: true}
			logger.log(tt.level, tt.msg, tt.fields...)

			output := buf.String()
			for _, exp := range tt.expected {
				assert.Contains(t, output, exp)
			}
		})
	}
}

func TestSimpleLogger_MultipleMessages(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewSimpleLogger("MULTI", true)

	// Log multiple messages of different types
	logger.Info("info message")
	logger.Debug("debug message")
	logger.Warn("warning message")
	logger.Error("error message")

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")

	assert.Len(t, lines, 4)
	assert.Contains(t, lines[0], "[INFO]")
	assert.Contains(t, lines[1], "[DEBUG]")
	assert.Contains(t, lines[2], "[WARN]")
	assert.Contains(t, lines[3], "[ERROR]")
}

func TestSimpleLogger_ComplexFields(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(nil)

	logger := NewSimpleLogger("COMPLEX", true)

	// Test with various field types
	logger.Info("complex fields",
		"string", "value",
		"int", 42,
		"float", 3.14,
		"bool", true,
		"nil", nil,
		"slice", []int{1, 2, 3},
		"map", map[string]int{"a": 1},
	)

	output := buf.String()
	assert.Contains(t, output, "string=value")
	assert.Contains(t, output, "int=42")
	assert.Contains(t, output, "float=3.14")
	assert.Contains(t, output, "bool=true")
	assert.Contains(t, output, "nil=<nil>")
	assert.Contains(t, output, "slice=[1 2 3]")
	assert.Contains(t, output, "map=map[a:1]")
}
