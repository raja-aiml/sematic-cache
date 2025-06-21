package reporters

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
)

// ConsoleReporter outputs test results to the console
type ConsoleReporter struct {
	verbose      bool
	showDetails  bool
	buffer       bytes.Buffer
	currentSuite *framework.TestSuite
	suiteStart   time.Time
}

// NewConsoleReporter creates a new console reporter
func NewConsoleReporter(verbose, showDetails bool) *ConsoleReporter {
	return &ConsoleReporter{
		verbose:     verbose,
		showDetails: showDetails,
	}
}

// StartSuite marks the beginning of a test suite
func (r *ConsoleReporter) StartSuite(suite *framework.TestSuite) {
	r.currentSuite = suite
	r.suiteStart = time.Now()

	header := fmt.Sprintf("\n=== Test Suite: %s ===", suite.Name)
	fmt.Println(header)
	if suite.Description != "" {
		fmt.Printf("Description: %s\n", suite.Description)
	}
	fmt.Printf("Started at: %s\n\n", r.suiteStart.Format(time.RFC3339))
}

// EndSuite marks the end of a test suite
func (r *ConsoleReporter) EndSuite(suite *framework.TestSuite, results []framework.TestResult) {
	duration := time.Since(r.suiteStart)

	var passed, failed int
	for _, result := range results {
		if result.Passed {
			passed++
		} else {
			failed++
		}
	}

	fmt.Printf("\n=== Suite Summary: %s ===\n", suite.Name)
	fmt.Printf("Total tests: %d\n", len(results))
	fmt.Printf("Passed: %d\n", passed)
	fmt.Printf("Failed: %d\n", failed)
	fmt.Printf("Duration: %s\n", duration.Round(time.Millisecond))

	if failed > 0 {
		fmt.Println("\nFailed tests:")
		for _, result := range results {
			if !result.Passed {
				fmt.Printf("  - %s: %s\n", result.Name, result.Message)
			}
		}
	}
	fmt.Println(strings.Repeat("=", 50))
}

// StartTest marks the beginning of a test
func (r *ConsoleReporter) StartTest(test *framework.TestCase) {
	if r.verbose {
		fmt.Printf("Running: %s", test.Name)
		if test.Description != "" {
			fmt.Printf(" - %s", test.Description)
		}
		fmt.Print("...")
	}
}

// EndTest marks the end of a test
func (r *ConsoleReporter) EndTest(result framework.TestResult) {
	if r.verbose {
		if result.Passed {
			fmt.Printf(" ✓ PASS (%s)\n", result.Duration.Round(time.Millisecond))
		} else {
			fmt.Printf(" ✗ FAIL (%s)\n", result.Duration.Round(time.Millisecond))
			if result.Message != "" {
				fmt.Printf("    Message: %s\n", result.Message)
			}
			if result.Error != nil {
				fmt.Printf("    Error: %v\n", result.Error)
			}
		}

		if r.showDetails && len(result.Details) > 0 {
			fmt.Println("    Details:")
			for k, v := range result.Details {
				fmt.Printf("      %s: %v\n", k, v)
			}
		}
	} else {
		// Non-verbose mode: show dots for progress
		if result.Passed {
			fmt.Print(".")
		} else {
			fmt.Print("F")
		}
	}
}

// GenerateReport generates a final report
func (r *ConsoleReporter) GenerateReport() ([]byte, error) {
	return r.buffer.Bytes(), nil
}
