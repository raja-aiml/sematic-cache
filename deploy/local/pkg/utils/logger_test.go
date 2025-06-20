package utils

import (
	"testing"
)

func TestLogger_Methods(t *testing.T) {
	logger := NewLogger("test")
	
	// Test that all methods can be called without panic
	t.Run("Info", func(t *testing.T) {
		logger.Info("test message")
		logger.Info("formatted %s %d", "string", 42)
	})
	
	t.Run("Warn", func(t *testing.T) {
		logger.Warn("warning message")
		logger.Warn("warning: %s", "be careful")
	})
	
	t.Run("Error", func(t *testing.T) {
		logger.Error("error message")
		logger.Error("error code %d", 500)
	})
	
	t.Run("Debug", func(t *testing.T) {
		logger.Debug("debug message")
		logger.Debug("debug: %v", struct{}{})
	})
	
	t.Run("Step", func(t *testing.T) {
		logger.Step("step message")
		logger.Step("step %d of %d", 1, 5)
	})
}

func TestLogger_Prefix(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
	}{
		{"empty", ""},
		{"simple", "test"},
		{"complex", "test-123_ABC"},
		{"with_spaces", "test logger"},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := NewLogger(tt.prefix)
			if logger.prefix != tt.prefix {
				t.Errorf("Logger prefix = %v, want %v", logger.prefix, tt.prefix)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewLogger(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewLogger("bench")
	}
}

func BenchmarkLogger_Info(b *testing.B) {
	logger := NewLogger("bench")
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		logger.Info("benchmark message %d", i)
	}
}

func BenchmarkLogger_Error(b *testing.B) {
	logger := NewLogger("bench")
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		logger.Error("error %d", i)
	}
}