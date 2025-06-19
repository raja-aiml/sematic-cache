package utils

import (
	"fmt"
	"os"

	"github.com/fatih/color"
)

type Logger struct {
	prefix string
}

func NewLogger(prefix string) *Logger {
	return &Logger{prefix: prefix}
}

func (l *Logger) Info(format string, args ...interface{}) {
	color.Green("✓ [%s] %s", l.prefix, fmt.Sprintf(format, args...))
}

func (l *Logger) Warn(format string, args ...interface{}) {
	color.Yellow("⚠ [%s] %s", l.prefix, fmt.Sprintf(format, args...))
}

func (l *Logger) Error(format string, args ...interface{}) {
	color.Red("✗ [%s] %s", l.prefix, fmt.Sprintf(format, args...))
}

func (l *Logger) Debug(format string, args ...interface{}) {
	if os.Getenv("DEBUG") != "" {
		color.Blue("→ [%s] %s", l.prefix, fmt.Sprintf(format, args...))
	}
}

func (l *Logger) Step(format string, args ...interface{}) {
	fmt.Printf("→ [%s] %s\n", l.prefix, fmt.Sprintf(format, args...))
}
