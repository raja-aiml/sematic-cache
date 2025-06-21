package framework

import (
	"fmt"
	"log"
	"strings"
)

// SimpleLogger provides basic logging for tests
type SimpleLogger struct {
	prefix  string
	verbose bool
}

// NewSimpleLogger creates a new simple logger
func NewSimpleLogger(prefix string, verbose bool) *SimpleLogger {
	return &SimpleLogger{
		prefix:  prefix,
		verbose: verbose,
	}
}

// Info logs an info message
func (l *SimpleLogger) Info(msg string, fields ...interface{}) {
	l.log("INFO", msg, fields...)
}

// Debug logs a debug message
func (l *SimpleLogger) Debug(msg string, fields ...interface{}) {
	if l.verbose {
		l.log("DEBUG", msg, fields...)
	}
}

// Error logs an error message
func (l *SimpleLogger) Error(msg string, fields ...interface{}) {
	l.log("ERROR", msg, fields...)
}

// Warn logs a warning message
func (l *SimpleLogger) Warn(msg string, fields ...interface{}) {
	l.log("WARN", msg, fields...)
}

func (l *SimpleLogger) log(level, msg string, fields ...interface{}) {
	// Format fields as key=value pairs
	var fieldStrs []string
	for i := 0; i < len(fields); i += 2 {
		if i+1 < len(fields) {
			fieldStrs = append(fieldStrs, fmt.Sprintf("%v=%v", fields[i], fields[i+1]))
		}
	}
	
	logMsg := fmt.Sprintf("[%s] %s", level, msg)
	if l.prefix != "" {
		logMsg = fmt.Sprintf("[%s] %s", l.prefix, logMsg)
	}
	
	if len(fieldStrs) > 0 {
		logMsg = fmt.Sprintf("%s %s", logMsg, strings.Join(fieldStrs, " "))
	}
	
	log.Println(logMsg)
}