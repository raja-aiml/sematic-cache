package reporters

import (
	"encoding/json"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
)

// JSONReporter generates JSON test reports
type JSONReporter struct {
	suites []SuiteReport
}

// SuiteReport represents a test suite report
type SuiteReport struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	StartTime   time.Time     `json:"start_time"`
	EndTime     time.Time     `json:"end_time"`
	Duration    time.Duration `json:"duration"`
	Tests       []TestReport  `json:"tests"`
	Summary     SuiteSummary  `json:"summary"`
}

// TestReport represents a single test report
type TestReport struct {
	Name     string                 `json:"name"`
	Passed   bool                   `json:"passed"`
	Duration time.Duration          `json:"duration"`
	Message  string                 `json:"message,omitempty"`
	Error    string                 `json:"error,omitempty"`
	Details  map[string]interface{} `json:"details,omitempty"`
}

// SuiteSummary provides test statistics
type SuiteSummary struct {
	Total  int `json:"total"`
	Passed int `json:"passed"`
	Failed int `json:"failed"`
}

// NewJSONReporter creates a new JSON reporter
func NewJSONReporter() *JSONReporter {
	return &JSONReporter{
		suites: make([]SuiteReport, 0),
	}
}

// StartSuite marks the beginning of a test suite
func (r *JSONReporter) StartSuite(suite *framework.TestSuite) {
	report := SuiteReport{
		Name:        suite.Name,
		Description: suite.Description,
		StartTime:   time.Now(),
		Tests:       make([]TestReport, 0),
	}
	r.suites = append(r.suites, report)
}

// EndSuite marks the end of a test suite
func (r *JSONReporter) EndSuite(suite *framework.TestSuite, results []framework.TestResult) {
	if len(r.suites) == 0 {
		return
	}

	// Get the last suite (current one)
	currentIdx := len(r.suites) - 1
	currentSuite := &r.suites[currentIdx]

	currentSuite.EndTime = time.Now()
	currentSuite.Duration = currentSuite.EndTime.Sub(currentSuite.StartTime)

	// Process results
	var passed, failed int
	for _, result := range results {
		testReport := TestReport{
			Name:     result.Name,
			Passed:   result.Passed,
			Duration: result.Duration,
			Message:  result.Message,
			Details:  result.Details,
		}

		if result.Error != nil {
			testReport.Error = result.Error.Error()
		}

		currentSuite.Tests = append(currentSuite.Tests, testReport)

		if result.Passed {
			passed++
		} else {
			failed++
		}
	}

	currentSuite.Summary = SuiteSummary{
		Total:  len(results),
		Passed: passed,
		Failed: failed,
	}
}

// StartTest marks the beginning of a test
func (r *JSONReporter) StartTest(test *framework.TestCase) {
	// JSON reporter doesn't need to do anything on test start
}

// EndTest marks the end of a test
func (r *JSONReporter) EndTest(result framework.TestResult) {
	// Test results are collected in EndSuite
}

// GenerateReport generates a JSON report
func (r *JSONReporter) GenerateReport() ([]byte, error) {
	report := struct {
		Timestamp time.Time      `json:"timestamp"`
		Suites    []SuiteReport  `json:"suites"`
		Summary   OverallSummary `json:"summary"`
	}{
		Timestamp: time.Now(),
		Suites:    r.suites,
		Summary:   r.calculateOverallSummary(),
	}

	return json.MarshalIndent(report, "", "  ")
}

// OverallSummary provides overall test statistics
type OverallSummary struct {
	TotalSuites int           `json:"total_suites"`
	TotalTests  int           `json:"total_tests"`
	Passed      int           `json:"passed"`
	Failed      int           `json:"failed"`
	Duration    time.Duration `json:"duration"`
}

func (r *JSONReporter) calculateOverallSummary() OverallSummary {
	summary := OverallSummary{
		TotalSuites: len(r.suites),
	}

	var minStart, maxEnd time.Time
	for i, suite := range r.suites {
		summary.TotalTests += suite.Summary.Total
		summary.Passed += suite.Summary.Passed
		summary.Failed += suite.Summary.Failed

		if i == 0 || suite.StartTime.Before(minStart) {
			minStart = suite.StartTime
		}
		if i == 0 || suite.EndTime.After(maxEnd) {
			maxEnd = suite.EndTime
		}
	}

	if !minStart.IsZero() && !maxEnd.IsZero() {
		summary.Duration = maxEnd.Sub(minStart)
	}

	return summary
}
