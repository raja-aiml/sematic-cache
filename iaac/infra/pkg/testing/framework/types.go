package framework

import (
	"context"
	"time"
)

// TestCase represents a single test case
type TestCase struct {
	Name        string
	Description string
	Timeout     time.Duration
	Fn          TestFunc
}

// TestFunc is the function signature for test cases
type TestFunc func(ctx context.Context, env *TestEnvironment) TestResult

// TestResult represents the result of a test
type TestResult struct {
	Name      string
	Passed    bool
	Duration  time.Duration
	Message   string
	Error     error
	Details   map[string]interface{}
}

// TestSuite represents a collection of related tests
type TestSuite struct {
	Name        string
	Description string
	Tests       []TestCase
	Setup       func(ctx context.Context, env *TestEnvironment) error
	Teardown    func(ctx context.Context, env *TestEnvironment) error
}

// TestEnvironment provides access to test resources
type TestEnvironment struct {
	KubeClient interface{} // Will be replaced with actual kubernetes client
	Config     *TestConfig
	Logger     Logger
	Context    map[string]interface{} // Shared context between tests
}

// TestConfig holds test configuration
type TestConfig struct {
	Namespace       string
	Scenario        string
	Timeout         time.Duration
	Parallel        bool
	FailFast        bool
	VerboseLogging  bool
}

// Logger interface for test logging
type Logger interface {
	Info(msg string, fields ...interface{})
	Debug(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
}

// Reporter interface for test reporting
type Reporter interface {
	StartSuite(suite *TestSuite)
	EndSuite(suite *TestSuite, results []TestResult)
	StartTest(test *TestCase)
	EndTest(result TestResult)
	GenerateReport() ([]byte, error)
}