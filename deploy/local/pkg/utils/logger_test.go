package utils

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestNewLogger(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
		want   string
	}{
		{
			name:   "creates logger with prefix",
			prefix: "test",
			want:   "test",
		},
		{
			name:   "creates logger with empty prefix",
			prefix: "",
			want:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := NewLogger(tt.prefix)
			if logger.prefix != tt.want {
				t.Errorf("NewLogger() prefix = %v, want %v", logger.prefix, tt.want)
			}
		})
	}
}

func TestLogger_Info(t *testing.T) {
	// Capture stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	logger := NewLogger("test")
	logger.Info("test message %s", "arg")

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "[test]") {
		t.Errorf("Info() output missing prefix")
	}
	if !strings.Contains(output, "test message arg") {
		t.Errorf("Info() output missing formatted message")
	}
}

func TestLogger_Debug(t *testing.T) {
	tests := []struct {
		name      string
		debugMode bool
		wantLog   bool
	}{
		{
			name:      "logs when DEBUG is set",
			debugMode: true,
			wantLog:   true,
		},
		{
			name:      "no logs when DEBUG is not set",
			debugMode: false,
			wantLog:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set or unset DEBUG env
			if tt.debugMode {
				os.Setenv("DEBUG", "1")
			} else {
				os.Unsetenv("DEBUG")
			}

			// Capture stdout
			old := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			logger := NewLogger("test")
			logger.Debug("debug message")

			w.Close()
			os.Stdout = old

			var buf bytes.Buffer
			buf.ReadFrom(r)
			output := buf.String()

			hasOutput := len(output) > 0
			if hasOutput != tt.wantLog {
				t.Errorf("Debug() logged = %v, want %v", hasOutput, tt.wantLog)
			}
		})
	}
}

func BenchmarkLogger_Info(b *testing.B) {
	logger := NewLogger("bench")
	// Disable output during benchmark
	old := os.Stdout
	os.Stdout = nil
	defer func() { os.Stdout = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logger.Info("benchmark message %d", i)
	}
}