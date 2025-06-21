package framework

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRunner(t *testing.T) {
	config := &TestConfig{
		Namespace: "test",
		Scenario:  "minimal",
	}
	reporter := &MockReporter{}
	env := &TestEnvironment{
		Logger: &MockLogger{},
	}

	runner := NewRunner(config, reporter, env)
	assert.NotNil(t, runner)
	assert.Equal(t, config, runner.config)
	assert.Equal(t, reporter, runner.reporter)
	assert.Equal(t, env, runner.env)
}

func TestRunner_RunSuite(t *testing.T) {
	tests := []struct {
		name           string
		suite          *TestSuite
		setupError     error
		teardownError  error
		expectedPassed int
		expectedFailed int
	}{
		{
			name: "successful_suite",
			suite: &TestSuite{
				Name: "success-suite",
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
			},
			expectedPassed: 2,
			expectedFailed: 0,
		},
		{
			name: "suite_with_failures",
			suite: &TestSuite{
				Name: "failure-suite",
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
							return TestResult{Name: "test-2", Passed: false, Message: "Test failed"}
						},
					},
				},
			},
			expectedPassed: 1,
			expectedFailed: 1,
		},
		{
			name: "suite_with_setup_teardown",
			suite: &TestSuite{
				Name: "setup-teardown-suite",
				Setup: func(ctx context.Context, env *TestEnvironment) error {
					env.Context["setup"] = true
					return nil
				},
				Teardown: func(ctx context.Context, env *TestEnvironment) error {
					env.Context["teardown"] = true
					return nil
				},
				Tests: []TestCase{
					{
						Name: "test-1",
						Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
							// Check setup was called
							if env.Context["setup"] == true {
								return TestResult{Name: "test-1", Passed: true}
							}
							return TestResult{Name: "test-1", Passed: false, Message: "Setup not called"}
						},
					},
				},
			},
			expectedPassed: 1,
			expectedFailed: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &TestConfig{
				Namespace: "test",
				Scenario:  "minimal",
			}
			reporter := &MockReporter{}
			env := &TestEnvironment{
				Logger:  &MockLogger{},
				Context: make(map[string]interface{}),
			}

			runner := NewRunner(config, reporter, env)

			results, err := runner.RunSuite(context.Background(), tt.suite)
			assert.NoError(t, err)

			passed := 0
			failed := 0
			for _, r := range results {
				if r.Passed {
					passed++
				} else {
					failed++
				}
			}

			assert.Equal(t, tt.expectedPassed, passed)
			assert.Equal(t, tt.expectedFailed, failed)
			assert.True(t, reporter.StartSuiteCalled)
			assert.True(t, reporter.EndSuiteCalled)
		})
	}
}

func TestRunner_RunSuite_SetupFailure(t *testing.T) {
	suite := &TestSuite{
		Name: "setup-failure-suite",
		Setup: func(ctx context.Context, env *TestEnvironment) error {
			return errors.New("setup failed")
		},
		Tests: []TestCase{
			{
				Name: "test-1",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					return TestResult{Name: "test-1", Passed: true}
				},
			},
		},
	}

	config := &TestConfig{}
	reporter := &MockReporter{}
	env := &TestEnvironment{Logger: &MockLogger{}}

	runner := NewRunner(config, reporter, env)
	results, err := runner.RunSuite(context.Background(), suite)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "suite setup failed")
	assert.Nil(t, results)
}

func TestRunner_runSequential(t *testing.T) {
	tests := []struct {
		name     string
		config   *TestConfig
		tests    []TestCase
		expected int
	}{
		{
			name:   "run_all_tests",
			config: &TestConfig{FailFast: false},
			tests: []TestCase{
				{
					Name: "test-1",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-1", Passed: true}
					},
				},
				{
					Name: "test-2",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-2", Passed: false}
					},
				},
				{
					Name: "test-3",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-3", Passed: true}
					},
				},
			},
			expected: 3,
		},
		{
			name:   "fail_fast_enabled",
			config: &TestConfig{FailFast: true},
			tests: []TestCase{
				{
					Name: "test-1",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-1", Passed: true}
					},
				},
				{
					Name: "test-2",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-2", Passed: false}
					},
				},
				{
					Name: "test-3",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-3", Passed: true}
					},
				},
			},
			expected: 2, // Should stop after first failure
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := &MockReporter{}
			env := &TestEnvironment{Logger: &MockLogger{}}
			runner := NewRunner(tt.config, reporter, env)

			results := runner.runSequential(context.Background(), tt.tests)
			assert.Len(t, results, tt.expected)
		})
	}
}

func TestRunner_runParallel(t *testing.T) {
	var counter int32
	tests := []TestCase{
		{
			Name: "test-1",
			Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
				atomic.AddInt32(&counter, 1)
				time.Sleep(10 * time.Millisecond)
				return TestResult{Name: "test-1", Passed: true}
			},
		},
		{
			Name: "test-2",
			Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
				atomic.AddInt32(&counter, 1)
				time.Sleep(10 * time.Millisecond)
				return TestResult{Name: "test-2", Passed: true}
			},
		},
		{
			Name: "test-3",
			Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
				atomic.AddInt32(&counter, 1)
				time.Sleep(10 * time.Millisecond)
				return TestResult{Name: "test-3", Passed: true}
			},
		},
	}

	config := &TestConfig{Parallel: true}
	reporter := &MockReporter{}
	env := &TestEnvironment{Logger: &MockLogger{}}
	runner := NewRunner(config, reporter, env)

	start := time.Now()
	results := runner.runParallel(context.Background(), tests)
	duration := time.Since(start)

	assert.Len(t, results, 3)
	assert.Equal(t, int32(3), atomic.LoadInt32(&counter))
	// Should complete faster than sequential (30ms sequential vs ~10ms parallel)
	assert.Less(t, duration, 25*time.Millisecond)
}

func TestRunner_runTest(t *testing.T) {
	tests := []struct {
		name     string
		testCase TestCase
		timeout  time.Duration
		wantPass bool
	}{
		{
			name: "successful_test",
			testCase: TestCase{
				Name: "success",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					return TestResult{Name: "success", Passed: true, Message: "Test passed"}
				},
			},
			wantPass: true,
		},
		{
			name: "failed_test",
			testCase: TestCase{
				Name: "failure",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					return TestResult{Name: "failure", Passed: false, Error: errors.New("test error")}
				},
			},
			wantPass: false,
		},
		{
			name: "test_with_timeout",
			testCase: TestCase{
				Name:    "timeout-test",
				Timeout: 50 * time.Millisecond,
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					time.Sleep(10 * time.Millisecond)
					return TestResult{Name: "timeout-test", Passed: true}
				},
			},
			wantPass: true,
		},
		{
			name: "test_timeout_exceeded",
			testCase: TestCase{
				Name:    "timeout-exceeded",
				Timeout: 10 * time.Millisecond,
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					time.Sleep(50 * time.Millisecond)
					return TestResult{Name: "timeout-exceeded", Passed: true}
				},
			},
			wantPass: false,
		},
		{
			name: "test_panic_recovery",
			testCase: TestCase{
				Name: "panic-test",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					panic("test panic")
				},
			},
			wantPass: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &TestConfig{}
			reporter := &MockReporter{}
			env := &TestEnvironment{Logger: &MockLogger{}}
			runner := NewRunner(config, reporter, env)

			result := runner.runTest(context.Background(), tt.testCase)

			assert.Equal(t, tt.wantPass, result.Passed)
			assert.True(t, reporter.StartTestCalled)
			assert.True(t, reporter.EndTestCalled)

			if !tt.wantPass {
				if tt.name == "test_timeout_exceeded" {
					assert.Contains(t, result.Message, "timed out")
				} else if tt.name == "test_panic_recovery" {
					assert.Contains(t, result.Message, "panic")
				}
			}
		})
	}
}

func TestRunner_RunMultipleSuites(t *testing.T) {
	suites := []*TestSuite{
		{
			Name: "suite-1",
			Tests: []TestCase{
				{
					Name: "test-1",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-1", Passed: true}
					},
				},
			},
		},
		{
			Name: "suite-2",
			Tests: []TestCase{
				{
					Name: "test-2",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-2", Passed: true}
					},
				},
				{
					Name: "test-3",
					Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
						return TestResult{Name: "test-3", Passed: false}
					},
				},
			},
		},
	}

	t.Run("run_all_suites", func(t *testing.T) {
		config := &TestConfig{FailFast: false}
		reporter := &MockReporter{}
		env := &TestEnvironment{Logger: &MockLogger{}}
		runner := NewRunner(config, reporter, env)

		results, err := runner.RunMultipleSuites(context.Background(), suites)
		assert.NoError(t, err)
		assert.Len(t, results, 2)
		assert.Len(t, results["suite-1"], 1)
		assert.Len(t, results["suite-2"], 2)
	})

	t.Run("fail_fast_across_suites", func(t *testing.T) {
		config := &TestConfig{FailFast: true}
		reporter := &MockReporter{}
		env := &TestEnvironment{Logger: &MockLogger{}}
		runner := NewRunner(config, reporter, env)

		results, err := runner.RunMultipleSuites(context.Background(), suites)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "stopping execution")
		// Should have results from both suites since failure is in second suite
		assert.Len(t, results, 2)
	})
}

func TestRunner_ComplexScenario(t *testing.T) {
	// Test a complex scenario with setup, teardown, parallel execution, and mixed results
	setupCalled := false
	teardownCalled := false
	var executionOrder []string
	var mu sync.Mutex

	suite := &TestSuite{
		Name: "complex-suite",
		Setup: func(ctx context.Context, env *TestEnvironment) error {
			setupCalled = true
			env.Context["setupData"] = "initialized"
			return nil
		},
		Teardown: func(ctx context.Context, env *TestEnvironment) error {
			teardownCalled = true
			return nil
		},
		Tests: []TestCase{
			{
				Name:    "fast-test",
				Timeout: 100 * time.Millisecond,
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					mu.Lock()
					executionOrder = append(executionOrder, "fast-test")
					mu.Unlock()

					if env.Context["setupData"] != "initialized" {
						return TestResult{Name: "fast-test", Passed: false, Message: "Setup data missing"}
					}
					return TestResult{Name: "fast-test", Passed: true}
				},
			},
			{
				Name: "slow-test",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					time.Sleep(20 * time.Millisecond)
					mu.Lock()
					executionOrder = append(executionOrder, "slow-test")
					mu.Unlock()
					return TestResult{Name: "slow-test", Passed: true}
				},
			},
			{
				Name: "failing-test",
				Fn: func(ctx context.Context, env *TestEnvironment) TestResult {
					mu.Lock()
					executionOrder = append(executionOrder, "failing-test")
					mu.Unlock()
					return TestResult{
						Name:    "failing-test",
						Passed:  false,
						Message: "Expected failure",
						Error:   errors.New("test error"),
						Details: map[string]interface{}{
							"reason": "validation failed",
						},
					}
				},
			},
		},
	}

	config := &TestConfig{
		Namespace:      "test-ns",
		Scenario:       "complex",
		Parallel:       true,
		FailFast:       false,
		VerboseLogging: true,
	}
	reporter := &MockReporter{}
	logger := &MockLogger{}
	env := &TestEnvironment{
		Logger:  logger,
		Context: make(map[string]interface{}),
		Config:  config,
	}

	runner := NewRunner(config, reporter, env)
	results, err := runner.RunSuite(context.Background(), suite)

	require.NoError(t, err)
	assert.True(t, setupCalled)
	assert.True(t, teardownCalled)
	assert.Len(t, results, 3)
	assert.Len(t, executionOrder, 3)

	// Verify results
	passedCount := 0
	failedCount := 0
	for _, r := range results {
		if r.Passed {
			passedCount++
		} else {
			failedCount++
		}
	}
	assert.Equal(t, 2, passedCount)
	assert.Equal(t, 1, failedCount)
}
