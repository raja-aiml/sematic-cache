package reporters

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
	"github.com/stretchr/testify/assert"
)

func TestNewConsoleReporter(t *testing.T) {
	tests := []struct {
		name        string
		verbose     bool
		showDetails bool
	}{
		{
			name:        "verbose_with_details",
			verbose:     true,
			showDetails: true,
		},
		{
			name:        "verbose_no_details",
			verbose:     true,
			showDetails: false,
		},
		{
			name:        "non_verbose",
			verbose:     false,
			showDetails: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewConsoleReporter(tt.verbose, tt.showDetails)
			assert.NotNil(t, reporter)
			assert.Equal(t, tt.verbose, reporter.verbose)
			assert.Equal(t, tt.showDetails, reporter.showDetails)
			assert.NotNil(t, reporter.buffer)
		})
	}
}

// captureOutput captures stdout for testing console output
func captureOutput(f func()) string {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	f()

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

func TestConsoleReporter_StartSuite(t *testing.T) {
	tests := []struct {
		name             string
		suite            *framework.TestSuite
		expectedContains []string
	}{
		{
			name: "suite_with_description",
			suite: &framework.TestSuite{
				Name:        "test-suite",
				Description: "Test suite description",
			},
			expectedContains: []string{
				"=== Test Suite: test-suite ===",
				"Description: Test suite description",
				"Started at:",
			},
		},
		{
			name: "suite_without_description",
			suite: &framework.TestSuite{
				Name: "simple-suite",
			},
			expectedContains: []string{
				"=== Test Suite: simple-suite ===",
				"Started at:",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewConsoleReporter(true, true)

			output := captureOutput(func() {
				reporter.StartSuite(tt.suite)
			})

			for _, expected := range tt.expectedContains {
				assert.Contains(t, output, expected)
			}
			assert.Equal(t, tt.suite, reporter.currentSuite)
			assert.False(t, reporter.suiteStart.IsZero())
		})
	}
}

func TestConsoleReporter_EndSuite(t *testing.T) {
	tests := []struct {
		name     string
		suite    *framework.TestSuite
		results  []framework.TestResult
		expected []string
	}{
		{
			name: "all_passed",
			suite: &framework.TestSuite{
				Name: "success-suite",
			},
			results: []framework.TestResult{
				{Name: "test-1", Passed: true},
				{Name: "test-2", Passed: true},
			},
			expected: []string{
				"=== Suite Summary: success-suite ===",
				"Total tests: 2",
				"Passed: 2",
				"Failed: 0",
				"Duration:",
			},
		},
		{
			name: "mixed_results",
			suite: &framework.TestSuite{
				Name: "mixed-suite",
			},
			results: []framework.TestResult{
				{Name: "test-1", Passed: true},
				{Name: "test-2", Passed: false, Message: "Test failed"},
				{Name: "test-3", Passed: false, Message: "Another failure"},
			},
			expected: []string{
				"=== Suite Summary: mixed-suite ===",
				"Total tests: 3",
				"Passed: 1",
				"Failed: 2",
				"Failed tests:",
				"test-2: Test failed",
				"test-3: Another failure",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewConsoleReporter(true, true)
			reporter.StartSuite(tt.suite) // Initialize suiteStart

			output := captureOutput(func() {
				reporter.EndSuite(tt.suite, tt.results)
			})

			for _, expected := range tt.expected {
				assert.Contains(t, output, expected)
			}
		})
	}
}

func TestConsoleReporter_StartTest(t *testing.T) {
	tests := []struct {
		name             string
		verbose          bool
		test             *framework.TestCase
		expectedContains []string
		expectedEmpty    bool
	}{
		{
			name:    "verbose_with_description",
			verbose: true,
			test: &framework.TestCase{
				Name:        "test-case",
				Description: "Test description",
			},
			expectedContains: []string{
				"Running: test-case",
				"Test description",
				"...",
			},
		},
		{
			name:    "verbose_without_description",
			verbose: true,
			test: &framework.TestCase{
				Name: "simple-test",
			},
			expectedContains: []string{
				"Running: simple-test",
				"...",
			},
		},
		{
			name:    "non_verbose",
			verbose: false,
			test: &framework.TestCase{
				Name: "test-case",
			},
			expectedEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewConsoleReporter(tt.verbose, false)

			output := captureOutput(func() {
				reporter.StartTest(tt.test)
			})

			if tt.expectedEmpty {
				assert.Empty(t, output)
			} else {
				for _, expected := range tt.expectedContains {
					assert.Contains(t, output, expected)
				}
			}
		})
	}
}

func TestConsoleReporter_EndTest(t *testing.T) {
	tests := []struct {
		name        string
		verbose     bool
		showDetails bool
		result      framework.TestResult
		expected    []string
	}{
		{
			name:    "verbose_passed_test",
			verbose: true,
			result: framework.TestResult{
				Name:     "test-1",
				Passed:   true,
				Duration: 100 * time.Millisecond,
			},
			expected: []string{
				"✓ PASS",
				"100ms",
			},
		},
		{
			name:    "verbose_failed_test",
			verbose: true,
			result: framework.TestResult{
				Name:     "test-2",
				Passed:   false,
				Duration: 50 * time.Millisecond,
				Message:  "Test failed",
				Error:    fmt.Errorf("test error"),
			},
			expected: []string{
				"✗ FAIL",
				"50ms",
				"Message: Test failed",
				"Error: test error",
			},
		},
		{
			name:        "verbose_with_details",
			verbose:     true,
			showDetails: true,
			result: framework.TestResult{
				Name:     "test-3",
				Passed:   true,
				Duration: 200 * time.Millisecond,
				Details: map[string]interface{}{
					"key1": "value1",
					"key2": 42,
				},
			},
			expected: []string{
				"✓ PASS",
				"200ms",
				"Details:",
				"key1: value1",
				"key2: 42",
			},
		},
		{
			name:    "non_verbose_passed",
			verbose: false,
			result: framework.TestResult{
				Name:   "test-4",
				Passed: true,
			},
			expected: []string{"."},
		},
		{
			name:    "non_verbose_failed",
			verbose: false,
			result: framework.TestResult{
				Name:   "test-5",
				Passed: false,
			},
			expected: []string{"F"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewConsoleReporter(tt.verbose, tt.showDetails)

			output := captureOutput(func() {
				reporter.EndTest(tt.result)
			})

			for _, expected := range tt.expected {
				assert.Contains(t, output, expected)
			}
		})
	}
}

func TestConsoleReporter_GenerateReport(t *testing.T) {
	reporter := NewConsoleReporter(true, true)

	report, err := reporter.GenerateReport()
	assert.NoError(t, err)
	// Buffer should be empty initially
	assert.Empty(t, report)
}

func TestConsoleReporter_CompleteWorkflow(t *testing.T) {
	reporter := NewConsoleReporter(true, true)

	suite := &framework.TestSuite{
		Name:        "integration-suite",
		Description: "Integration test suite",
	}

	test1 := &framework.TestCase{
		Name:        "test-1",
		Description: "First test",
	}

	test2 := &framework.TestCase{
		Name: "test-2",
	}

	result1 := framework.TestResult{
		Name:     "test-1",
		Passed:   true,
		Duration: 150 * time.Millisecond,
		Details: map[string]interface{}{
			"status": "ok",
		},
	}

	result2 := framework.TestResult{
		Name:     "test-2",
		Passed:   false,
		Duration: 75 * time.Millisecond,
		Message:  "Test failed",
		Error:    fmt.Errorf("assertion failed"),
	}

	results := []framework.TestResult{result1, result2}

	// Capture the entire workflow
	output := captureOutput(func() {
		reporter.StartSuite(suite)
		reporter.StartTest(test1)
		reporter.EndTest(result1)
		reporter.StartTest(test2)
		reporter.EndTest(result2)
		reporter.EndSuite(suite, results)
	})

	// Verify key elements are present
	expectedElements := []string{
		"=== Test Suite: integration-suite ===",
		"Integration test suite",
		"Running: test-1",
		"First test",
		"✓ PASS",
		"150ms",
		"Details:",
		"status: ok",
		"Running: test-2",
		"✗ FAIL",
		"75ms",
		"Message: Test failed",
		"Error: assertion failed",
		"=== Suite Summary: integration-suite ===",
		"Total tests: 2",
		"Passed: 1",
		"Failed: 1",
		"Failed tests:",
		"test-2: Test failed",
	}

	for _, expected := range expectedElements {
		assert.Contains(t, output, expected)
	}
}

func TestConsoleReporter_NonVerboseWorkflow(t *testing.T) {
	reporter := NewConsoleReporter(false, false)

	suite := &framework.TestSuite{
		Name: "non-verbose-suite",
	}

	test1 := &framework.TestCase{Name: "test-1"}
	test2 := &framework.TestCase{Name: "test-2"}
	test3 := &framework.TestCase{Name: "test-3"}

	result1 := framework.TestResult{Name: "test-1", Passed: true}
	result2 := framework.TestResult{Name: "test-2", Passed: false}
	result3 := framework.TestResult{Name: "test-3", Passed: true}

	results := []framework.TestResult{result1, result2, result3}

	output := captureOutput(func() {
		reporter.StartSuite(suite)
		reporter.StartTest(test1)
		reporter.EndTest(result1)
		reporter.StartTest(test2)
		reporter.EndTest(result2)
		reporter.StartTest(test3)
		reporter.EndTest(result3)
		reporter.EndSuite(suite, results)
	})

	// In non-verbose mode, should see dots and F for results
	assert.Contains(t, output, ".")
	assert.Contains(t, output, "F")
	assert.Contains(t, output, "=== Suite Summary: non-verbose-suite ===")
	assert.Contains(t, output, "Total tests: 3")
	assert.Contains(t, output, "Passed: 2")
	assert.Contains(t, output, "Failed: 1")

	// Should NOT contain verbose elements
	assert.NotContains(t, output, "Running:")
	assert.NotContains(t, output, "✓ PASS")
	assert.NotContains(t, output, "✗ FAIL")
}

func TestConsoleReporter_EmptyResults(t *testing.T) {
	reporter := NewConsoleReporter(true, true)

	suite := &framework.TestSuite{
		Name: "empty-suite",
	}

	output := captureOutput(func() {
		reporter.StartSuite(suite)
		reporter.EndSuite(suite, []framework.TestResult{})
	})

	assert.Contains(t, output, "=== Test Suite: empty-suite ===")
	assert.Contains(t, output, "Total tests: 0")
	assert.Contains(t, output, "Passed: 0")
	assert.Contains(t, output, "Failed: 0")
	assert.NotContains(t, output, "Failed tests:")
}

func TestConsoleReporter_LongSuiteName(t *testing.T) {
	reporter := NewConsoleReporter(true, false)

	longName := strings.Repeat("a", 100)
	suite := &framework.TestSuite{
		Name: longName,
	}

	output := captureOutput(func() {
		reporter.StartSuite(suite)
		reporter.EndSuite(suite, []framework.TestResult{})
	})

	assert.Contains(t, output, fmt.Sprintf("=== Test Suite: %s ===", longName))
	assert.Contains(t, output, fmt.Sprintf("=== Suite Summary: %s ===", longName))
}
