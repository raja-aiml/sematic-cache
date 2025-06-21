package framework

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Runner executes test suites
type Runner struct {
	config   *TestConfig
	reporter Reporter
	env      *TestEnvironment
}

// NewRunner creates a new test runner
func NewRunner(config *TestConfig, reporter Reporter, env *TestEnvironment) *Runner {
	return &Runner{
		config:   config,
		reporter: reporter,
		env:      env,
	}
}

// RunSuite executes a test suite
func (r *Runner) RunSuite(ctx context.Context, suite *TestSuite) ([]TestResult, error) {
	r.reporter.StartSuite(suite)
	
	// Run setup if provided
	if suite.Setup != nil {
		if err := suite.Setup(ctx, r.env); err != nil {
			return nil, fmt.Errorf("suite setup failed: %w", err)
		}
	}
	
	// Run tests
	var results []TestResult
	if r.config.Parallel && !r.config.FailFast {
		results = r.runParallel(ctx, suite.Tests)
	} else {
		results = r.runSequential(ctx, suite.Tests)
	}
	
	// Run teardown if provided
	if suite.Teardown != nil {
		if err := suite.Teardown(ctx, r.env); err != nil {
			r.env.Logger.Error("suite teardown failed", "error", err)
		}
	}
	
	r.reporter.EndSuite(suite, results)
	return results, nil
}

// runSequential executes tests one by one
func (r *Runner) runSequential(ctx context.Context, tests []TestCase) []TestResult {
	results := make([]TestResult, 0, len(tests))
	
	for _, test := range tests {
		result := r.runTest(ctx, test)
		results = append(results, result)
		
		if r.config.FailFast && !result.Passed {
			r.env.Logger.Info("Stopping test execution due to failure (fail-fast enabled)")
			break
		}
	}
	
	return results
}

// runParallel executes tests concurrently
func (r *Runner) runParallel(ctx context.Context, tests []TestCase) []TestResult {
	results := make([]TestResult, len(tests))
	var wg sync.WaitGroup
	
	for i, test := range tests {
		wg.Add(1)
		go func(idx int, tc TestCase) {
			defer wg.Done()
			results[idx] = r.runTest(ctx, tc)
		}(i, test)
	}
	
	wg.Wait()
	return results
}

// runTest executes a single test
func (r *Runner) runTest(ctx context.Context, test TestCase) TestResult {
	r.reporter.StartTest(&test)
	
	// Set timeout if specified
	testCtx := ctx
	if test.Timeout > 0 {
		var cancel context.CancelFunc
		testCtx, cancel = context.WithTimeout(ctx, test.Timeout)
		defer cancel()
	}
	
	start := time.Now()
	
	// Run the test in a goroutine to handle panics
	resultChan := make(chan TestResult, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				resultChan <- TestResult{
					Name:     test.Name,
					Passed:   false,
					Duration: time.Since(start),
					Message:  fmt.Sprintf("Test panicked: %v", r),
					Error:    fmt.Errorf("panic: %v", r),
				}
			}
		}()
		
		result := test.Fn(testCtx, r.env)
		result.Duration = time.Since(start)
		resultChan <- result
	}()
	
	// Wait for result or timeout
	select {
	case result := <-resultChan:
		r.reporter.EndTest(result)
		return result
	case <-testCtx.Done():
		result := TestResult{
			Name:     test.Name,
			Passed:   false,
			Duration: time.Since(start),
			Message:  "Test timed out",
			Error:    testCtx.Err(),
		}
		r.reporter.EndTest(result)
		return result
	}
}

// RunMultipleSuites executes multiple test suites
func (r *Runner) RunMultipleSuites(ctx context.Context, suites []*TestSuite) (map[string][]TestResult, error) {
	allResults := make(map[string][]TestResult)
	
	for _, suite := range suites {
		r.env.Logger.Info("Running test suite", "suite", suite.Name)
		results, err := r.RunSuite(ctx, suite)
		if err != nil {
			return allResults, fmt.Errorf("suite %s failed: %w", suite.Name, err)
		}
		allResults[suite.Name] = results
		
		// Check if we should stop on failure
		if r.config.FailFast {
			for _, result := range results {
				if !result.Passed {
					return allResults, fmt.Errorf("stopping execution: test %s failed", result.Name)
				}
			}
		}
	}
	
	return allResults, nil
}