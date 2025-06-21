package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/reporters"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/suites/integration"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/suites/smoke"
	"github.com/spf13/cobra"
)

var (
	testSuite      string
	testScenario   string
	testNamespace  string
	testTimeout    int
	testParallel   bool
	testFailFast   bool
	testVerbose    bool
	testReportType string
	testOutputFile string
)

// TestCmd returns the test command
func TestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "test [suite]",
		Short: "Run validation tests",
		Long: `Run validation tests for the deployed blueprint.

Available test suites:
  smoke        - Quick validation that essential components are working
  integration  - Test component interactions and data flow
  performance  - Run performance and load tests
  security     - Validate security configurations
  all          - Run all test suites

Examples:
  # Run smoke tests
  iaac test smoke

  # Run smoke tests for a specific scenario
  iaac test smoke --scenario full-stack

  # Run all tests with verbose output
  iaac test all --verbose

  # Run tests in parallel with JSON output
  iaac test smoke --parallel --report json --output results.json`,
		Args: cobra.MaximumNArgs(1),
		RunE: runTest,
	}

	// Add flags
	cmd.Flags().StringVar(&testScenario, "scenario", "minimal", "Blueprint scenario to test")
	cmd.Flags().StringVar(&testNamespace, "namespace", "", "Namespace to test (default: based on scenario)")
	cmd.Flags().IntVar(&testTimeout, "timeout", 300, "Test timeout in seconds")
	cmd.Flags().BoolVar(&testParallel, "parallel", false, "Run tests in parallel")
	cmd.Flags().BoolVar(&testFailFast, "fail-fast", false, "Stop on first failure")
	cmd.Flags().BoolVarP(&testVerbose, "verbose", "v", false, "Verbose output")
	cmd.Flags().StringVar(&testReportType, "report", "console", "Report type (console, json)")
	cmd.Flags().StringVarP(&testOutputFile, "output", "o", "", "Output file for test report")

	return cmd
}

func runTest(cmd *cobra.Command, args []string) error {
	// Determine test suite
	if len(args) > 0 {
		testSuite = args[0]
	} else {
		testSuite = "smoke"
	}

	// Create test configuration
	config := &framework.TestConfig{
		Scenario:       testScenario,
		Namespace:      testNamespace,
		Timeout:        time.Duration(testTimeout) * time.Second,
		Parallel:       testParallel,
		FailFast:       testFailFast,
		VerboseLogging: testVerbose,
	}

	// Create reporter
	var reporter framework.Reporter
	switch testReportType {
	case "json":
		reporter = reporters.NewJSONReporter()
	case "console":
		reporter = reporters.NewConsoleReporter(testVerbose, true)
	default:
		return fmt.Errorf("unknown report type: %s", testReportType)
	}

	// Create test environment
	logger := framework.NewSimpleLogger("test", testVerbose)
	env := &framework.TestEnvironment{
		KubeClient: nil, // TODO: Initialize kubernetes client
		Config:     config,
		Logger:     logger,
		Context:    make(map[string]interface{}),
	}

	// Create test runner
	runner := framework.NewRunner(config, reporter, env)

	// Get test suites
	suites, err := getTestSuites(testSuite)
	if err != nil {
		return err
	}

	// Run tests
	ctx := context.Background()
	results, err := runner.RunMultipleSuites(ctx, suites)
	if err != nil {
		return fmt.Errorf("test execution failed: %w", err)
	}

	// Generate report
	report, err := reporter.GenerateReport()
	if err != nil {
		return fmt.Errorf("failed to generate report: %w", err)
	}

	// Write report to file if specified
	if testOutputFile != "" {
		if err := os.WriteFile(testOutputFile, report, 0644); err != nil {
			return fmt.Errorf("failed to write report: %w", err)
		}
		fmt.Printf("Test report written to: %s\n", testOutputFile)
	}

	// Check for failures
	for suiteName, suiteResults := range results {
		for _, result := range suiteResults {
			if !result.Passed {
				return fmt.Errorf("test suite '%s' has failures", suiteName)
			}
		}
	}

	return nil
}

// getTestSuites returns the test suites based on selection
func getTestSuites(selection string) ([]*framework.TestSuite, error) {
	var suites []*framework.TestSuite

	switch strings.ToLower(selection) {
	case "smoke":
		suites = append(suites, smoke.NewSmokeTestSuite())
	case "integration":
		suites = append(suites, integration.NewIntegrationTestSuite())
	case "performance":
		// TODO: Add performance test suite
		return nil, fmt.Errorf("performance tests not yet implemented")
	case "security":
		// TODO: Add security test suite
		return nil, fmt.Errorf("security tests not yet implemented")
	case "all":
		suites = append(suites, smoke.NewSmokeTestSuite())
		suites = append(suites, integration.NewIntegrationTestSuite())
		// TODO: Add other suites
	default:
		return nil, fmt.Errorf("unknown test suite: %s", selection)
	}

	return suites, nil
}
