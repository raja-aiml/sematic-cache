package utils

import (
	"os"
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
	// Just ensure no panic
	logger := NewLogger("test")
	logger.Info("test message %s", "arg")
	// If we get here without panic, test passes
}

func TestLogger_Debug(t *testing.T) {
	// Test with DEBUG set
	os.Setenv("DEBUG", "1")
	defer os.Unsetenv("DEBUG")
	
	logger := NewLogger("test")
	logger.Debug("debug message")
	
	// Test without DEBUG
	os.Unsetenv("DEBUG")
	logger.Debug("should not log")
	
	// If we get here without panic, test passes
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