package reporters

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewJSONReporter(t *testing.T) {
	reporter := NewJSONReporter()
	assert.NotNil(t, reporter)
	assert.NotNil(t, reporter.suites)
	assert.Empty(t, reporter.suites)
}

func TestJSONReporter_StartSuite(t *testing.T) {
	tests := []struct {
		name  string
		suite *framework.TestSuite
	}{
		{
			name: "suite_with_description",
			suite: &framework.TestSuite{
				Name:        "test-suite",
				Description: "Test suite description",
			},
		},
		{
			name: "suite_without_description",
			suite: &framework.TestSuite{
				Name: "simple-suite",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewJSONReporter()

			startTime := time.Now()
			reporter.StartSuite(tt.suite)

			require.Len(t, reporter.suites, 1)
			suite := reporter.suites[0]

			assert.Equal(t, tt.suite.Name, suite.Name)
			assert.Equal(t, tt.suite.Description, suite.Description)
			assert.True(t, suite.StartTime.After(startTime.Add(-time.Second)))
			assert.True(t, suite.StartTime.Before(time.Now().Add(time.Second)))
			assert.Empty(t, suite.Tests)
			assert.Zero(t, suite.EndTime)
			assert.Zero(t, suite.Duration)
		})
	}
}

func TestJSONReporter_EndSuite(t *testing.T) {
	tests := []struct {
		name           string
		suite          *framework.TestSuite
		results        []framework.TestResult
		expectedPassed int
		expectedFailed int
	}{
		{
			name: "all_passed",
			suite: &framework.TestSuite{
				Name: "success-suite",
			},
			results: []framework.TestResult{
				{
					Name:     "test-1",
					Passed:   true,
					Duration: 100 * time.Millisecond,
					Message:  "Test passed",
				},
				{
					Name:     "test-2",
					Passed:   true,
					Duration: 150 * time.Millisecond,
				},
			},
			expectedPassed: 2,
			expectedFailed: 0,
		},
		{
			name: "mixed_results",
			suite: &framework.TestSuite{
				Name: "mixed-suite",
			},
			results: []framework.TestResult{
				{
					Name:     "test-1",
					Passed:   true,
					Duration: 50 * time.Millisecond,
				},
				{
					Name:     "test-2",
					Passed:   false,
					Duration: 75 * time.Millisecond,
					Message:  "Test failed",
					Error:    errors.New("assertion failed"),
					Details: map[string]interface{}{
						"expected": "foo",
						"actual":   "bar",
					},
				},
				{
					Name:     "test-3",
					Passed:   false,
					Duration: 25 * time.Millisecond,
					Error:    errors.New("timeout"),
				},
			},
			expectedPassed: 1,
			expectedFailed: 2,
		},
		{
			name: "empty_results",
			suite: &framework.TestSuite{
				Name: "empty-suite",
			},
			results:        []framework.TestResult{},
			expectedPassed: 0,
			expectedFailed: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewJSONReporter()
			reporter.StartSuite(tt.suite)

			endTime := time.Now()
			reporter.EndSuite(tt.suite, tt.results)

			require.Len(t, reporter.suites, 1)
			suite := reporter.suites[0]

			assert.True(t, suite.EndTime.After(endTime.Add(-time.Second)))
			assert.True(t, suite.EndTime.Before(time.Now().Add(time.Second)))
			assert.True(t, suite.Duration > 0)
			assert.Equal(t, len(tt.results), len(suite.Tests))
			assert.Equal(t, len(tt.results), suite.Summary.Total)
			assert.Equal(t, tt.expectedPassed, suite.Summary.Passed)
			assert.Equal(t, tt.expectedFailed, suite.Summary.Failed)

			// Verify test reports
			for i, result := range tt.results {
				testReport := suite.Tests[i]
				assert.Equal(t, result.Name, testReport.Name)
				assert.Equal(t, result.Passed, testReport.Passed)
				assert.Equal(t, result.Duration, testReport.Duration)
				assert.Equal(t, result.Message, testReport.Message)
				assert.Equal(t, result.Details, testReport.Details)

				if result.Error != nil {
					assert.Equal(t, result.Error.Error(), testReport.Error)
				} else {
					assert.Empty(t, testReport.Error)
				}
			}
		})
	}
}

func TestJSONReporter_EndSuite_EmptyReporter(t *testing.T) {
	reporter := NewJSONReporter()
	suite := &framework.TestSuite{Name: "test-suite"}

	// Call EndSuite without StartSuite - should not panic
	reporter.EndSuite(suite, []framework.TestResult{})
	assert.Empty(t, reporter.suites)
}

func TestJSONReporter_StartTest(t *testing.T) {
	reporter := NewJSONReporter()
	test := &framework.TestCase{Name: "test-case"}

	// StartTest should be a no-op for JSON reporter
	reporter.StartTest(test)
	assert.Empty(t, reporter.suites)
}

func TestJSONReporter_EndTest(t *testing.T) {
	reporter := NewJSONReporter()
	result := framework.TestResult{Name: "test-result", Passed: true}

	// EndTest should be a no-op for JSON reporter
	reporter.EndTest(result)
	assert.Empty(t, reporter.suites)
}

func TestJSONReporter_GenerateReport(t *testing.T) {
	tests := []struct {
		name           string
		setupFunc      func(*JSONReporter)
		expectedSuites int
		expectedTests  int
		expectedPassed int
		expectedFailed int
	}{
		{
			name: "empty_reporter",
			setupFunc: func(r *JSONReporter) {
				// No setup - empty reporter
			},
			expectedSuites: 0,
			expectedTests:  0,
			expectedPassed: 0,
			expectedFailed: 0,
		},
		{
			name: "single_suite",
			setupFunc: func(r *JSONReporter) {
				suite := &framework.TestSuite{
					Name:        "test-suite",
					Description: "Test description",
				}
				results := []framework.TestResult{
					{Name: "test-1", Passed: true, Duration: 100 * time.Millisecond},
					{Name: "test-2", Passed: false, Duration: 50 * time.Millisecond, Error: errors.New("failed")},
				}
				r.StartSuite(suite)
				r.EndSuite(suite, results)
			},
			expectedSuites: 1,
			expectedTests:  2,
			expectedPassed: 1,
			expectedFailed: 1,
		},
		{
			name: "multiple_suites",
			setupFunc: func(r *JSONReporter) {
				// First suite
				suite1 := &framework.TestSuite{Name: "suite-1"}
				results1 := []framework.TestResult{
					{Name: "test-1", Passed: true},
					{Name: "test-2", Passed: true},
				}
				r.StartSuite(suite1)
				r.EndSuite(suite1, results1)

				// Second suite
				suite2 := &framework.TestSuite{Name: "suite-2"}
				results2 := []framework.TestResult{
					{Name: "test-3", Passed: false, Message: "Failed test"},
				}
				r.StartSuite(suite2)
				r.EndSuite(suite2, results2)
			},
			expectedSuites: 2,
			expectedTests:  3,
			expectedPassed: 2,
			expectedFailed: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := NewJSONReporter()
			tt.setupFunc(reporter)

			reportBytes, err := reporter.GenerateReport()
			require.NoError(t, err)
			require.NotEmpty(t, reportBytes)

			// Parse the JSON report
			var report struct {
				Timestamp time.Time      `json:"timestamp"`
				Suites    []SuiteReport  `json:"suites"`
				Summary   OverallSummary `json:"summary"`
			}

			err = json.Unmarshal(reportBytes, &report)
			require.NoError(t, err)

			// Verify timestamp is recent
			assert.True(t, report.Timestamp.After(time.Now().Add(-time.Minute)))

			// Verify suite count
			assert.Len(t, report.Suites, tt.expectedSuites)

			// Verify overall summary
			assert.Equal(t, tt.expectedSuites, report.Summary.TotalSuites)
			assert.Equal(t, tt.expectedTests, report.Summary.TotalTests)
			assert.Equal(t, tt.expectedPassed, report.Summary.Passed)
			assert.Equal(t, tt.expectedFailed, report.Summary.Failed)

			if tt.expectedSuites > 0 {
				assert.True(t, report.Summary.Duration > 0)
			} else {
				assert.Zero(t, report.Summary.Duration)
			}
		})
	}
}

func TestJSONReporter_GenerateReport_ValidJSON(t *testing.T) {
	reporter := NewJSONReporter()

	// Setup a complex scenario
	suite := &framework.TestSuite{
		Name:        "complex-suite",
		Description: "Complex test suite with various scenarios",
	}

	results := []framework.TestResult{
		{
			Name:     "simple-pass",
			Passed:   true,
			Duration: 50 * time.Millisecond,
			Message:  "Test passed successfully",
		},
		{
			Name:     "test-with-details",
			Passed:   true,
			Duration: 100 * time.Millisecond,
			Details: map[string]interface{}{
				"iterations": 10,
				"average":    1.23,
				"status":     "ok",
				"nested": map[string]interface{}{
					"key": "value",
				},
			},
		},
		{
			Name:     "failed-test",
			Passed:   false,
			Duration: 75 * time.Millisecond,
			Message:  "Assertion failed",
			Error:    errors.New("expected 'foo' but got 'bar'"),
			Details: map[string]interface{}{
				"expected": "foo",
				"actual":   "bar",
				"line":     42,
			},
		},
	}

	reporter.StartSuite(suite)
	reporter.EndSuite(suite, results)

	reportBytes, err := reporter.GenerateReport()
	require.NoError(t, err)

	// Verify it's valid JSON by unmarshaling
	var reportData map[string]interface{}
	err = json.Unmarshal(reportBytes, &reportData)
	require.NoError(t, err)

	// Verify the JSON is properly formatted (indented)
	compactBytes, err := json.Marshal(reportData)
	require.NoError(t, err)
	assert.True(t, len(reportBytes) > len(compactBytes), "Report should be indented")

	// Check that specific fields exist in the JSON
	assert.Contains(t, string(reportBytes), `"timestamp"`)
	assert.Contains(t, string(reportBytes), `"suites"`)
	assert.Contains(t, string(reportBytes), `"summary"`)
	assert.Contains(t, string(reportBytes), `"complex-suite"`)
	assert.Contains(t, string(reportBytes), `"expected 'foo' but got 'bar'"`)
	assert.Contains(t, string(reportBytes), `"iterations": 10`)
}

func TestJSONReporter_calculateOverallSummary(t *testing.T) {
	tests := []struct {
		name     string
		suites   []SuiteReport
		expected OverallSummary
	}{
		{
			name:   "empty_suites",
			suites: []SuiteReport{},
			expected: OverallSummary{
				TotalSuites: 0,
				TotalTests:  0,
				Passed:      0,
				Failed:      0,
				Duration:    0,
			},
		},
		{
			name: "single_suite",
			suites: []SuiteReport{
				{
					StartTime: time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC),
					EndTime:   time.Date(2023, 1, 1, 10, 1, 0, 0, time.UTC),
					Summary: SuiteSummary{
						Total:  3,
						Passed: 2,
						Failed: 1,
					},
				},
			},
			expected: OverallSummary{
				TotalSuites: 1,
				TotalTests:  3,
				Passed:      2,
				Failed:      1,
				Duration:    time.Minute,
			},
		},
		{
			name: "multiple_suites",
			suites: []SuiteReport{
				{
					StartTime: time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC),
					EndTime:   time.Date(2023, 1, 1, 10, 1, 0, 0, time.UTC),
					Summary: SuiteSummary{
						Total:  2,
						Passed: 2,
						Failed: 0,
					},
				},
				{
					StartTime: time.Date(2023, 1, 1, 10, 0, 30, 0, time.UTC),
					EndTime:   time.Date(2023, 1, 1, 10, 2, 0, 0, time.UTC),
					Summary: SuiteSummary{
						Total:  4,
						Passed: 3,
						Failed: 1,
					},
				},
			},
			expected: OverallSummary{
				TotalSuites: 2,
				TotalTests:  6,
				Passed:      5,
				Failed:      1,
				Duration:    2 * time.Minute, // From earliest start to latest end
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reporter := &JSONReporter{suites: tt.suites}
			summary := reporter.calculateOverallSummary()

			assert.Equal(t, tt.expected.TotalSuites, summary.TotalSuites)
			assert.Equal(t, tt.expected.TotalTests, summary.TotalTests)
			assert.Equal(t, tt.expected.Passed, summary.Passed)
			assert.Equal(t, tt.expected.Failed, summary.Failed)
			assert.Equal(t, tt.expected.Duration, summary.Duration)
		})
	}
}

func TestJSONReporter_CompleteWorkflow(t *testing.T) {
	reporter := NewJSONReporter()

	// First suite
	suite1 := &framework.TestSuite{
		Name:        "unit-tests",
		Description: "Unit test suite",
	}

	results1 := []framework.TestResult{
		{
			Name:     "test-addition",
			Passed:   true,
			Duration: 10 * time.Millisecond,
			Message:  "Math works correctly",
		},
		{
			Name:     "test-division",
			Passed:   false,
			Duration: 5 * time.Millisecond,
			Message:  "Division by zero not handled",
			Error:    errors.New("division by zero"),
			Details: map[string]interface{}{
				"dividend": 10,
				"divisor":  0,
			},
		},
	}

	// Second suite
	suite2 := &framework.TestSuite{
		Name: "integration-tests",
	}

	results2 := []framework.TestResult{
		{
			Name:     "test-api-endpoint",
			Passed:   true,
			Duration: 250 * time.Millisecond,
			Details: map[string]interface{}{
				"status_code":      200,
				"response_time_ms": 249,
			},
		},
	}

	// Run complete workflow
	reporter.StartSuite(suite1)
	reporter.EndSuite(suite1, results1)

	reporter.StartSuite(suite2)
	reporter.EndSuite(suite2, results2)

	// Generate and verify report
	reportBytes, err := reporter.GenerateReport()
	require.NoError(t, err)

	var report struct {
		Timestamp time.Time      `json:"timestamp"`
		Suites    []SuiteReport  `json:"suites"`
		Summary   OverallSummary `json:"summary"`
	}

	err = json.Unmarshal(reportBytes, &report)
	require.NoError(t, err)

	// Verify overall structure
	assert.Len(t, report.Suites, 2)
	assert.Equal(t, 2, report.Summary.TotalSuites)
	assert.Equal(t, 3, report.Summary.TotalTests)
	assert.Equal(t, 2, report.Summary.Passed)
	assert.Equal(t, 1, report.Summary.Failed)

	// Verify first suite
	assert.Equal(t, "unit-tests", report.Suites[0].Name)
	assert.Equal(t, "Unit test suite", report.Suites[0].Description)
	assert.Len(t, report.Suites[0].Tests, 2)
	assert.Equal(t, 2, report.Suites[0].Summary.Total)
	assert.Equal(t, 1, report.Suites[0].Summary.Passed)
	assert.Equal(t, 1, report.Suites[0].Summary.Failed)

	// Verify second suite
	assert.Equal(t, "integration-tests", report.Suites[1].Name)
	assert.Empty(t, report.Suites[1].Description)
	assert.Len(t, report.Suites[1].Tests, 1)
	assert.Equal(t, 1, report.Suites[1].Summary.Total)
	assert.Equal(t, 1, report.Suites[1].Summary.Passed)
	assert.Equal(t, 0, report.Suites[1].Summary.Failed)

	// Verify specific test details
	failedTest := report.Suites[0].Tests[1]
	assert.Equal(t, "test-division", failedTest.Name)
	assert.False(t, failedTest.Passed)
	assert.Equal(t, "division by zero", failedTest.Error)
	assert.Equal(t, float64(0), failedTest.Details["divisor"])

	passedTest := report.Suites[1].Tests[0]
	assert.Equal(t, "test-api-endpoint", passedTest.Name)
	assert.True(t, passedTest.Passed)
	assert.Equal(t, float64(200), passedTest.Details["status_code"])
}
