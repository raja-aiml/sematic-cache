package framework

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestTestCase(t *testing.T) {
	testCase := TestCase{
		Name:        "test-case-1",
		Description: "Test case description",
		Timeout:     5 * time.Second,
		Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
			return TestResult{
				Name:   "test-case-1",
				Passed: true,
			}
		},
	}

	assert.Equal(t, "test-case-1", testCase.Name)
	assert.Equal(t, "Test case description", testCase.Description)
	assert.Equal(t, 5*time.Second, testCase.Timeout)
	assert.NotNil(t, testCase.Fn)
}

func TestTestResult(t *testing.T) {
	result := TestResult{
		Name:     "test-result",
		Passed:   true,
		Duration: 100 * time.Millisecond,
		Message:  "Test passed successfully",
		Error:    nil,
		Details: map[string]interface{}{
			"key1": "value1",
			"key2": 42,
		},
	}

	assert.Equal(t, "test-result", result.Name)
	assert.True(t, result.Passed)
	assert.Equal(t, 100*time.Millisecond, result.Duration)
	assert.Equal(t, "Test passed successfully", result.Message)
	assert.Nil(t, result.Error)
	assert.Equal(t, "value1", result.Details["key1"])
	assert.Equal(t, 42, result.Details["key2"])
}

func TestTestSuite(t *testing.T) {
	setupCalled := false
	teardownCalled := false

	suite := TestSuite{
		Name:        "test-suite",
		Description: "Test suite description",
		Tests: []TestCase{
			{
				Name: "test-1",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					return TestResult{Name: "test-1", Passed: true}
				},
			},
			{
				Name: "test-2",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					return TestResult{Name: "test-2", Passed: true}
				},
			},
		},
		Setup: func(ctx context.Context, env *TestEnvironment) error {
			setupCalled = true
			return nil
		},
		Teardown: func(ctx context.Context, env *TestEnvironment) error {
			teardownCalled = true
			return nil
		},
	}

	assert.Equal(t, "test-suite", suite.Name)
	assert.Equal(t, "Test suite description", suite.Description)
	assert.Len(t, suite.Tests, 2)
	assert.NotNil(t, suite.Setup)
	assert.NotNil(t, suite.Teardown)

	// Test setup/teardown functions
	env := &TestEnvironment{}
	err := suite.Setup(context.Background(), env)
	assert.NoError(t, err)
	assert.True(t, setupCalled)

	err = suite.Teardown(context.Background(), env)
	assert.NoError(t, err)
	assert.True(t, teardownCalled)
}

func TestTestEnvironment(t *testing.T) {
	logger := &MockLogger{}
	config := &TestConfig{
		Namespace: "test-namespace",
		Scenario:  "test-scenario",
	}

	env := TestEnvironment{
		KubeClient: nil,
		Config:     config,
		Logger:     logger,
		Context: map[string]interface{}{
			"key1": "value1",
			"key2": 42,
		},
	}

	assert.Nil(t, env.KubeClient)
	assert.Equal(t, config, env.Config)
	assert.Equal(t, logger, env.Logger)
	assert.Equal(t, "value1", env.Context["key1"])
	assert.Equal(t, 42, env.Context["key2"])
}

func TestTestConfig(t *testing.T) {
	tests := []struct {
		name   string
		config TestConfig
	}{
		{
			name: "full_config",
			config: TestConfig{
				Namespace:      "test-ns",
				Scenario:       "minimal",
				Timeout:        30 * time.Second,
				Parallel:       true,
				FailFast:       true,
				VerboseLogging: true,
			},
		},
		{
			name: "minimal_config",
			config: TestConfig{
				Namespace: "default",
				Scenario:  "default",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.NotEmpty(t, tt.config.Namespace)
			assert.NotEmpty(t, tt.config.Scenario)
		})
	}
}

// MockLogger implements the Logger interface for testing
type MockLogger struct {
	InfoCalled  bool
	DebugCalled bool
	ErrorCalled bool
	WarnCalled  bool
	LastMessage string
	LastFields  []interface{}
}

func (m *MockLogger) Info(msg string, fields ...interface{}) {
	m.InfoCalled = true
	m.LastMessage = msg
	m.LastFields = fields
}

func (m *MockLogger) Debug(msg string, fields ...interface{}) {
	m.DebugCalled = true
	m.LastMessage = msg
	m.LastFields = fields
}

func (m *MockLogger) Error(msg string, fields ...interface{}) {
	m.ErrorCalled = true
	m.LastMessage = msg
	m.LastFields = fields
}

func (m *MockLogger) Warn(msg string, fields ...interface{}) {
	m.WarnCalled = true
	m.LastMessage = msg
	m.LastFields = fields
}

func TestLoggerInterface(t *testing.T) {
	logger := &MockLogger{}

	logger.Info("info message", "key", "value")
	assert.True(t, logger.InfoCalled)
	assert.Equal(t, "info message", logger.LastMessage)
	assert.Equal(t, []interface{}{"key", "value"}, logger.LastFields)

	logger.Debug("debug message", "debug", true)
	assert.True(t, logger.DebugCalled)
	assert.Equal(t, "debug message", logger.LastMessage)

	logger.Error("error message", "error", "critical")
	assert.True(t, logger.ErrorCalled)
	assert.Equal(t, "error message", logger.LastMessage)

	logger.Warn("warning message", "level", "high")
	assert.True(t, logger.WarnCalled)
	assert.Equal(t, "warning message", logger.LastMessage)
}

// MockReporter implements the Reporter interface for testing
type MockReporter struct {
	StartSuiteCalled bool
	EndSuiteCalled   bool
	StartTestCalled  bool
	EndTestCalled    bool
	LastSuite        *TestSuite
	LastTest         *TestCase
	LastResults      []TestResult
	LastResult       TestResult
}

func (m *MockReporter) StartSuite(suite *TestSuite) {
	m.StartSuiteCalled = true
	m.LastSuite = suite
}

func (m *MockReporter) EndSuite(suite *TestSuite, results []TestResult) {
	m.EndSuiteCalled = true
	m.LastSuite = suite
	m.LastResults = results
}

func (m *MockReporter) StartTest(test *TestCase) {
	m.StartTestCalled = true
	m.LastTest = test
}

func (m *MockReporter) EndTest(result TestResult) {
	m.EndTestCalled = true
	m.LastResult = result
}

func (m *MockReporter) GenerateReport() ([]byte, error) {
	return []byte("mock report"), nil
}

func TestReporterInterface(t *testing.T) {
	reporter := &MockReporter{}

	suite := &TestSuite{Name: "test-suite"}
	reporter.StartSuite(suite)
	assert.True(t, reporter.StartSuiteCalled)
	assert.Equal(t, suite, reporter.LastSuite)

	test := &TestCase{Name: "test-case"}
	reporter.StartTest(test)
	assert.True(t, reporter.StartTestCalled)
	assert.Equal(t, test, reporter.LastTest)

	result := TestResult{Name: "test-case", Passed: true}
	reporter.EndTest(result)
	assert.True(t, reporter.EndTestCalled)
	assert.Equal(t, result, reporter.LastResult)

	results := []TestResult{result}
	reporter.EndSuite(suite, results)
	assert.True(t, reporter.EndSuiteCalled)
	assert.Equal(t, results, reporter.LastResults)

	report, err := reporter.GenerateReport()
	assert.NoError(t, err)
	assert.Equal(t, []byte("mock report"), report)
}
