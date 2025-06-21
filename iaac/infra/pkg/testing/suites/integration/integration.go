package integration

import (
	"context"
	"fmt"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing/framework"
)

// NewIntegrationTestSuite creates an integration test suite
func NewIntegrationTestSuite() *framework.TestSuite {
	return &framework.TestSuite{
		Name:        "integration",
		Description: "Test component interactions and data flow",
		Setup:       setupIntegrationTests,
		Teardown:    teardownIntegrationTests,
		Tests: []framework.TestCase{
			{
				Name:        "postgres-integration",
				Description: "Test PostgreSQL database operations",
				Timeout:     2 * time.Minute,
				Fn:          testPostgresIntegration,
			},
			{
				Name:        "redis-integration",
				Description: "Test Redis cache operations",
				Timeout:     2 * time.Minute,
				Fn:          testRedisIntegration,
			},
			{
				Name:        "ingress-integration",
				Description: "Test ingress routing",
				Timeout:     2 * time.Minute,
				Fn:          testIngressIntegration,
			},
			{
				Name:        "app-database-connectivity",
				Description: "Test application to database connectivity",
				Timeout:     2 * time.Minute,
				Fn:          testAppDatabaseConnectivity,
			},
			{
				Name:        "cross-namespace-communication",
				Description: "Test cross-namespace communication policies",
				Timeout:     2 * time.Minute,
				Fn:          testCrossNamespaceCommunication,
			},
			{
				Name:        "persistent-data",
				Description: "Test data persistence across pod restarts",
				Timeout:     3 * time.Minute,
				Fn:          testPersistentData,
			},
		},
	}
}

// setupIntegrationTests prepares the environment for integration tests
func setupIntegrationTests(ctx context.Context, env *framework.TestEnvironment) error {
	env.Logger.Info("Setting up integration test environment")

	// Create test namespace
	testNamespace := "integration-test"
	env.Context["test_namespace"] = testNamespace

	env.Logger.Info("Creating test namespace", "namespace", testNamespace)

	// TODO: Actually create namespace using kubernetes client
	// client := env.KubeClient
	// namespace := &v1.Namespace{
	//     ObjectMeta: metav1.ObjectMeta{
	//         Name: testNamespace,
	//     },
	// }
	// _, err := client.CoreV1().Namespaces().Create(ctx, namespace, metav1.CreateOptions{})

	return nil
}

// teardownIntegrationTests cleans up after integration tests
func teardownIntegrationTests(ctx context.Context, env *framework.TestEnvironment) error {
	env.Logger.Info("Cleaning up integration test environment")

	testNamespace, ok := env.Context["test_namespace"].(string)
	if !ok {
		return fmt.Errorf("test namespace not found in context")
	}

	env.Logger.Info("Deleting test namespace", "namespace", testNamespace)

	// TODO: Delete test namespace
	// client := env.KubeClient
	// return client.CoreV1().Namespaces().Delete(ctx, testNamespace, metav1.DeleteOptions{})

	return nil
}

// testPostgresIntegration tests PostgreSQL operations
func testPostgresIntegration(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"basic-connectivity", testPostgresConnectivity},
		{"create-database", testPostgresCreateDatabase},
		{"create-table", testPostgresCreateTable},
		{"insert-data", testPostgresInsertData},
		{"query-data", testPostgresQueryData},
		{"vector-extension", testPostgresVectorExtension},
		{"concurrent-connections", testPostgresConcurrentConnections},
	}

	results := make(map[string]bool)
	var failedTests []string

	for _, test := range tests {
		err := test.fn()
		results[test.name] = err == nil
		if err != nil {
			failedTests = append(failedTests, fmt.Sprintf("%s: %v", test.name, err))
			env.Logger.Error("PostgreSQL test failed", "test", test.name, "error", err)
		}
	}

	passed := len(failedTests) == 0
	message := "All PostgreSQL integration tests passed"
	if !passed {
		message = fmt.Sprintf("Failed tests: %v", failedTests)
	}

	return framework.TestResult{
		Name:    "postgres-integration",
		Passed:  passed,
		Message: message,
		Details: map[string]interface{}{
			"test_results": results,
			"failed_tests": failedTests,
		},
	}
}

// testRedisIntegration tests Redis operations
func testRedisIntegration(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"basic-connectivity", testRedisConnectivity},
		{"set-get-operations", testRedisSetGet},
		{"list-operations", testRedisListOps},
		{"hash-operations", testRedisHashOps},
		{"expire-operations", testRedisExpire},
		{"concurrent-operations", testRedisConcurrent},
	}

	results := make(map[string]bool)
	var failedTests []string

	for _, test := range tests {
		err := test.fn()
		results[test.name] = err == nil
		if err != nil {
			failedTests = append(failedTests, fmt.Sprintf("%s: %v", test.name, err))
			env.Logger.Error("Redis test failed", "test", test.name, "error", err)
		}
	}

	passed := len(failedTests) == 0
	message := "All Redis integration tests passed"
	if !passed {
		message = fmt.Sprintf("Failed tests: %v", failedTests)
	}

	return framework.TestResult{
		Name:    "redis-integration",
		Passed:  passed,
		Message: message,
		Details: map[string]interface{}{
			"test_results": results,
			"failed_tests": failedTests,
		},
	}
}

// testIngressIntegration tests ingress routing
func testIngressIntegration(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"ingress-controller-ready", testIngressControllerReady},
		{"http-routing", testHTTPRouting},
		{"https-routing", testHTTPSRouting},
		{"path-based-routing", testPathBasedRouting},
		{"host-based-routing", testHostBasedRouting},
	}

	results := make(map[string]bool)
	var failedTests []string

	for _, test := range tests {
		err := test.fn()
		results[test.name] = err == nil
		if err != nil {
			failedTests = append(failedTests, fmt.Sprintf("%s: %v", test.name, err))
		}
	}

	passed := len(failedTests) == 0
	message := "All ingress integration tests passed"
	if !passed {
		message = fmt.Sprintf("Failed tests: %v", failedTests)
	}

	return framework.TestResult{
		Name:    "ingress-integration",
		Passed:  passed,
		Message: message,
		Details: map[string]interface{}{
			"test_results": results,
			"failed_tests": failedTests,
		},
	}
}

// testAppDatabaseConnectivity tests application to database connectivity
func testAppDatabaseConnectivity(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	// Deploy a test application that connects to databases
	testApp := "connectivity-test-app"

	steps := []struct {
		name string
		fn   func() error
	}{
		{"deploy-test-app", func() error { return deployTestApp(ctx, env, testApp) }},
		{"wait-for-ready", func() error { return waitForAppReady(ctx, env, testApp) }},
		{"test-postgres-connection", func() error { return testAppPostgresConnection(ctx, env, testApp) }},
		{"test-redis-connection", func() error { return testAppRedisConnection(ctx, env, testApp) }},
		{"cleanup-test-app", func() error { return cleanupTestApp(ctx, env, testApp) }},
	}

	for _, step := range steps {
		if err := step.fn(); err != nil {
			return framework.TestResult{
				Name:    "app-database-connectivity",
				Passed:  false,
				Message: fmt.Sprintf("Failed at step '%s': %v", step.name, err),
				Error:   err,
			}
		}
	}

	return framework.TestResult{
		Name:    "app-database-connectivity",
		Passed:  true,
		Message: "Application successfully connected to all databases",
		Details: map[string]interface{}{
			"test_app": testApp,
		},
	}
}

// testCrossNamespaceCommunication tests network policies
func testCrossNamespaceCommunication(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	tests := []struct {
		name     string
		from     string
		to       string
		expected bool
	}{
		{"app-to-infra-allowed", "app", "infra", true},
		{"infra-to-app-denied", "infra", "app", false},
		{"app-to-monitoring-allowed", "app", "monitoring", true},
		{"default-to-infra-denied", "default", "infra", false},
	}

	results := make(map[string]bool)
	allPassed := true

	for _, test := range tests {
		canConnect := testNamespaceConnectivity(ctx, env, test.from, test.to)
		results[test.name] = canConnect == test.expected

		if canConnect != test.expected {
			allPassed = false
			env.Logger.Error("Network policy test failed",
				"test", test.name,
				"expected", test.expected,
				"actual", canConnect)
		}
	}

	message := "All network policy tests passed"
	if !allPassed {
		message = "Some network policy tests failed"
	}

	return framework.TestResult{
		Name:    "cross-namespace-communication",
		Passed:  allPassed,
		Message: message,
		Details: map[string]interface{}{
			"test_results": results,
		},
	}
}

// testPersistentData tests data persistence
func testPersistentData(ctx context.Context, env *framework.TestEnvironment) framework.TestResult {
	testData := map[string]string{
		"postgres": "test-value-postgres",
		"redis":    "test-value-redis",
	}

	// Step 1: Write test data
	for db, value := range testData {
		if err := writeTestData(ctx, env, db, value); err != nil {
			return framework.TestResult{
				Name:    "persistent-data",
				Passed:  false,
				Message: fmt.Sprintf("Failed to write test data to %s: %v", db, err),
				Error:   err,
			}
		}
	}

	// Step 2: Restart pods
	if err := restartDatabasePods(ctx, env); err != nil {
		return framework.TestResult{
			Name:    "persistent-data",
			Passed:  false,
			Message: fmt.Sprintf("Failed to restart pods: %v", err),
			Error:   err,
		}
	}

	// Step 3: Wait for pods to be ready
	if err := waitForDatabasesReady(ctx, env); err != nil {
		return framework.TestResult{
			Name:    "persistent-data",
			Passed:  false,
			Message: fmt.Sprintf("Databases not ready after restart: %v", err),
			Error:   err,
		}
	}

	// Step 4: Verify data exists
	for db, expectedValue := range testData {
		actualValue, err := readTestData(ctx, env, db)
		if err != nil {
			return framework.TestResult{
				Name:    "persistent-data",
				Passed:  false,
				Message: fmt.Sprintf("Failed to read test data from %s: %v", db, err),
				Error:   err,
			}
		}

		if actualValue != expectedValue {
			return framework.TestResult{
				Name:    "persistent-data",
				Passed:  false,
				Message: fmt.Sprintf("Data mismatch in %s: expected '%s', got '%s'", db, expectedValue, actualValue),
			}
		}
	}

	return framework.TestResult{
		Name:    "persistent-data",
		Passed:  true,
		Message: "Data persisted successfully across pod restarts",
		Details: map[string]interface{}{
			"test_data": testData,
		},
	}
}
