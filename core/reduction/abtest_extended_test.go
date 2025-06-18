package reduction

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"
)

// TestGetStrategy tests the GetStrategy method with various scenarios
func TestGetStrategy(t *testing.T) {
	defaultStrategy := Strategy{
		ID:        "default",
		Name:      "Default",
		TargetDim: 100,
	}
	
	manager := NewABTestManager(defaultStrategy)
	ctx := context.Background()
	
	// Test 1: No active test - should return default strategy
	strategy := manager.GetStrategy(ctx, "request1")
	if strategy.ID != defaultStrategy.ID {
		t.Errorf("Expected default strategy, got %s", strategy.ID)
	}
	
	// Test 2: Create and start a test
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{
		MinImpressions:   1000,
		MinDurationHours: 1,
	}
	
	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	// Start the test
	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}
	
	// Test 3: Get strategy with active test - should return one of the test strategies
	strategyCount := make(map[string]int)
	for i := 0; i < 1000; i++ {
		requestID := fmt.Sprintf("request_%d", i)
		strategy := manager.GetStrategy(ctx, requestID)
		strategyCount[strategy.ID]++
	}
	
	// Verify both strategies were selected (with some tolerance for randomness)
	// Allow wider margin due to hash distribution
	for _, s := range strategies {
		if count, ok := strategyCount[s.ID]; !ok || count < 300 || count > 700 {
			t.Errorf("Strategy %s was selected %d times, expected ~500", s.ID, count)
		}
	}
	
	// Test 4: Stop test and verify default strategy is returned
	err = manager.StopTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to stop test: %v", err)
	}
	
	strategy = manager.GetStrategy(ctx, "request_after_stop")
	if strategy.ID != defaultStrategy.ID {
		t.Errorf("Expected default strategy after test stop, got %s", strategy.ID)
	}
}

// TestCheckTestCompletion tests the test completion checking logic
func TestCheckTestCompletion(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{
		MinImpressions:        100,
		MinDurationHours:      0.0001, // Very short for testing
		SignificanceThreshold: 0.05,
		MetricWeights: MetricWeights{
			HitRate: 0.4,
			Latency: 0.3,
			Memory:  0.2,
			Quality: 0.1,
		},
	}
	
	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}
	
	// Test 1: Check before minimum impressions - should not be complete
	complete, summary := manager.CheckTestCompletion(test.ID)
	if complete {
		t.Error("Test should not be complete before minimum impressions")
	}
	if summary != nil {
		t.Error("Summary should be nil when test is not complete due to impressions")
	}
	
	// Test 2: Record sufficient impressions
	ctx := context.Background()
	for i := 0; i < 60; i++ {
		for _, strategy := range strategies {
			metrics := ImpressionMetrics{
				CacheHit:        i%2 == 0,
				LatencyMs:       int64(10 + i%5),
				SearchLatencyMs: int64(5 + i%3),
				SimilarityScore: 0.8 + float64(i%10)/100.0,
			}
			manager.RecordImpression(ctx, strategy.ID, metrics)
		}
	}
	
	// Wait for minimum duration (0.0001 hours = 0.36 seconds)
	time.Sleep(400 * time.Millisecond)
	
	// Test 3: Check after sufficient impressions and duration
	complete, summary = manager.CheckTestCompletion(test.ID)
	// Summary should be returned even if not statistically significant
	if summary == nil {
		t.Fatal("Summary should not be nil after minimum requirements")
	}
	
	// Verify summary contents
	if summary.TotalImpressions < config.MinImpressions {
		t.Errorf("Total impressions %d < minimum %d", summary.TotalImpressions, config.MinImpressions)
	}
	
	if len(summary.Results) != len(strategies) {
		t.Errorf("Expected %d strategy results, got %d", len(strategies), len(summary.Results))
	}
	
	// Test 4: Check non-existent test
	complete, summary = manager.CheckTestCompletion("non-existent")
	if complete || summary != nil {
		t.Error("Non-existent test should return false and nil summary")
	}
}

// TestCalculateSignificance tests the statistical significance calculation
func TestCalculateSignificance(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	tests := []struct {
		name      string
		results   []StrategyResult
		wantRange [2]float64 // min and max expected significance
	}{
		{
			name:      "empty results",
			results:   []StrategyResult{},
			wantRange: [2]float64{0.0, 0.0},
		},
		{
			name: "single result",
			results: []StrategyResult{
				{HitRate: 0.8, Impressions: 1000},
			},
			wantRange: [2]float64{0.0, 0.0},
		},
		{
			name: "identical hit rates",
			results: []StrategyResult{
				{HitRate: 0.8, Impressions: 1000},
				{HitRate: 0.8, Impressions: 1000},
			},
			wantRange: [2]float64{0.0, 0.1}, // Very low significance
		},
		{
			name: "different hit rates with high impressions",
			results: []StrategyResult{
				{HitRate: 0.8, Impressions: 10000},
				{HitRate: 0.7, Impressions: 10000},
			},
			wantRange: [2]float64{0.9, 1.0}, // High significance
		},
		{
			name: "different hit rates with low impressions",
			results: []StrategyResult{
				{HitRate: 0.8, Impressions: 10},
				{HitRate: 0.7, Impressions: 10},
			},
			wantRange: [2]float64{0.0, 0.5}, // Low significance due to small sample
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			significance := manager.calculateSignificance(tt.results)
			if significance < tt.wantRange[0] || significance > tt.wantRange[1] {
				t.Errorf("Significance %f not in expected range [%f, %f]", 
					significance, tt.wantRange[0], tt.wantRange[1])
			}
		})
	}
}

// TestStopTest tests stopping tests with various scenarios
func TestStopTest(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	// Test 1: Stop non-existent test
	err := manager.StopTest("non-existent")
	if err == nil {
		t.Error("Expected error when stopping non-existent test")
	}
	
	// Test 2: Create, start, and stop a test
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{MinImpressions: 100}
	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	// Start test
	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}
	
	// Verify test is active
	if manager.activeTestID != test.ID {
		t.Error("Test should be active after starting")
	}
	
	// Stop test
	err = manager.StopTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to stop test: %v", err)
	}
	
	// Verify test is no longer active
	if manager.activeTestID != "" {
		t.Error("No test should be active after stopping")
	}
	
	// Verify test status
	stoppedTest := manager.getTest(test.ID)
	if stoppedTest.Status != TestStatusStopped {
		t.Errorf("Test status should be 'stopped', got %s", stoppedTest.Status)
	}
	
	// Verify end time is set
	if stoppedTest.EndTime.IsZero() {
		t.Error("End time should be set when test is stopped")
	}
}

// TestGetTest tests the getTest helper method
func TestGetTest(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	// Test 1: Get non-existent test
	test := manager.getTest("non-existent")
	if test != nil {
		t.Error("Expected nil for non-existent test")
	}
	
	// Test 2: Create test and retrieve it
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{MinImpressions: 100}
	createdTest, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	// Retrieve test
	retrievedTest := manager.getTest(createdTest.ID)
	if retrievedTest == nil {
		t.Fatal("Failed to retrieve created test")
	}
	
	if retrievedTest.ID != createdTest.ID {
		t.Errorf("Retrieved test ID mismatch: got %s, want %s", retrievedTest.ID, createdTest.ID)
	}
}

// TestExportResults tests exporting test results
func TestExportResults(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	// Test 1: Export non-existent test
	_, err := manager.ExportResults("non-existent")
	if err == nil {
		t.Error("Expected error when exporting non-existent test")
	}
	
	// Test 2: Create test and export results
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{
		MinImpressions: 100,
		MetricWeights: MetricWeights{
			HitRate: 0.4,
			Latency: 0.3,
			Memory:  0.2,
			Quality: 0.1,
		},
	}
	
	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	// Start the test so we can record impressions
	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}
	
	// Record some impressions
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		for _, strategy := range strategies {
			metrics := ImpressionMetrics{
				CacheHit:        i%2 == 0,
				LatencyMs:       int64(10 + i%5),
				SearchLatencyMs: int64(5 + i%3),
				SimilarityScore: 0.8 + float64(i%10)/100.0,
			}
			manager.RecordImpression(ctx, strategy.ID, metrics)
		}
	}
	
	// Export results
	export, err := manager.ExportResults(test.ID)
	if err != nil {
		t.Fatalf("Failed to export results: %v", err)
	}
	
	// Verify export
	if export.Test.ID != test.ID {
		t.Errorf("Export test ID mismatch: got %s, want %s", export.Test.ID, test.ID)
	}
	
	if export.Summary == nil {
		t.Fatal("Export summary should not be nil")
	}
	
	if export.Summary.TotalImpressions != 100 {
		t.Errorf("Expected 100 total impressions, got %d", export.Summary.TotalImpressions)
	}
	
	if len(export.Summary.Results) != len(strategies) {
		t.Errorf("Expected %d strategy results, got %d", len(strategies), len(export.Summary.Results))
	}
	
	// Verify timestamp is recent
	if time.Since(export.Timestamp) > time.Second {
		t.Error("Export timestamp should be recent")
	}
}

// TestHashString tests the hash function
func TestHashString(t *testing.T) {
	tests := []struct {
		input    string
		wantSame bool
		other    string
	}{
		{"test", true, "test"},
		{"hello", false, "world"},
		{"", true, ""},
		{"a", false, "b"},
		{"long string with spaces", true, "long string with spaces"},
	}
	
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			hash1 := hashString(tt.input)
			hash2 := hashString(tt.other)
			
			if tt.wantSame && hash1 != hash2 {
				t.Errorf("Expected same hash for %q and %q", tt.input, tt.other)
			}
			if !tt.wantSame && hash1 == hash2 {
				t.Errorf("Expected different hash for %q and %q", tt.input, tt.other)
			}
		})
	}
}

// TestNormalCDF tests the normal CDF approximation
func TestNormalCDF(t *testing.T) {
	tests := []struct {
		x        float64
		expected float64
		epsilon  float64
	}{
		{0, 0.5, 0.01},         // CDF(0) = 0.5
		{1, 0.8413, 0.01},      // CDF(1) ≈ 0.8413
		{-1, 0.1587, 0.01},     // CDF(-1) ≈ 0.1587
		{2, 0.9772, 0.01},      // CDF(2) ≈ 0.9772
		{-2, 0.0228, 0.01},     // CDF(-2) ≈ 0.0228
		{3, 0.9987, 0.01},      // CDF(3) ≈ 0.9987
		{-3, 0.0013, 0.01},     // CDF(-3) ≈ 0.0013
	}
	
	for _, tt := range tests {
		t.Run(fmt.Sprintf("x=%f", tt.x), func(t *testing.T) {
			result := normalCDF(tt.x)
			if math.Abs(result-tt.expected) > tt.epsilon {
				t.Errorf("normalCDF(%f) = %f, want %f ± %f", tt.x, result, tt.expected, tt.epsilon)
			}
		})
	}
}

// TestMonitoringDashboard tests the monitoring dashboard
func TestMonitoringDashboard(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	dashboard := NewMonitoringDashboard(manager)
	
	// Test 1: No active test
	metrics := dashboard.GetLiveMetrics()
	if metrics != nil {
		t.Error("Expected nil metrics when no test is active")
	}
	
	// Test 2: Create and start a test
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{MinImpressions: 100}
	test, err := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Failed to create test: %v", err)
	}
	
	err = manager.StartTest(test.ID)
	if err != nil {
		t.Fatalf("Failed to start test: %v", err)
	}
	
	// Record some impressions
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		for _, strategy := range strategies {
			metrics := ImpressionMetrics{
				CacheHit:        i%2 == 0,
				LatencyMs:       int64(10),
				SearchLatencyMs: int64(5),
				SimilarityScore: 0.85,
				Error:           nil,
			}
			if i%10 == 0 && strategy.ID == "s1" {
				metrics.Error = fmt.Errorf("test error")
			}
			manager.RecordImpression(ctx, strategy.ID, metrics)
		}
	}
	
	// Test 3: Get live metrics
	liveMetrics := dashboard.GetLiveMetrics()
	if liveMetrics == nil {
		t.Fatal("Expected live metrics for active test")
	}
	
	if liveMetrics.TestID != test.ID {
		t.Errorf("Live metrics test ID mismatch: got %s, want %s", liveMetrics.TestID, test.ID)
	}
	
	if len(liveMetrics.Strategies) != len(strategies) {
		t.Errorf("Expected %d strategy metrics, got %d", len(strategies), len(liveMetrics.Strategies))
	}
	
	// Verify strategy metrics
	for _, stratMetrics := range liveMetrics.Strategies {
		if stratMetrics.Impressions != 50 {
			t.Errorf("Strategy %s: expected 50 impressions, got %d", 
				stratMetrics.StrategyID, stratMetrics.Impressions)
		}
		
		expectedHitRate := 0.5 // 50% cache hit rate based on i%2 == 0
		if math.Abs(stratMetrics.HitRate-expectedHitRate) > 0.01 {
			t.Errorf("Strategy %s: expected hit rate ~%f, got %f", 
				stratMetrics.StrategyID, expectedHitRate, stratMetrics.HitRate)
		}
		
		// Check error rate
		if stratMetrics.StrategyID == "s1" {
			expectedErrorRate := 0.1 // 10% errors for s1
			if math.Abs(stratMetrics.ErrorRate-expectedErrorRate) > 0.01 {
				t.Errorf("Strategy s1: expected error rate ~%f, got %f", 
					expectedErrorRate, stratMetrics.ErrorRate)
			}
		} else {
			if stratMetrics.ErrorRate != 0 {
				t.Errorf("Strategy %s: expected 0 error rate, got %f", 
					stratMetrics.StrategyID, stratMetrics.ErrorRate)
			}
		}
	}
	
	// Verify duration is positive
	if liveMetrics.Duration <= 0 {
		t.Error("Live metrics duration should be positive")
	}
}

// TestCreateTestValidation tests validation in CreateTest
func TestCreateTestValidation(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	
	tests := []struct {
		name       string
		strategies []Strategy
		allocation []float64
		wantError  bool
		errorMsg   string
	}{
		{
			name:       "single strategy",
			strategies: []Strategy{{ID: "s1", Name: "S1", TargetDim: 50}},
			allocation: []float64{1.0},
			wantError:  true,
			errorMsg:   "need at least 2 strategies for A/B test",
		},
		{
			name: "allocation mismatch",
			strategies: []Strategy{
				{ID: "s1", Name: "S1", TargetDim: 50},
				{ID: "s2", Name: "S2", TargetDim: 100},
			},
			allocation: []float64{0.5}, // Only one allocation for two strategies
			wantError:  true,
			errorMsg:   "allocation length must match strategies",
		},
		{
			name: "allocation doesn't sum to 1",
			strategies: []Strategy{
				{ID: "s1", Name: "S1", TargetDim: 50},
				{ID: "s2", Name: "S2", TargetDim: 100},
			},
			allocation: []float64{0.3, 0.3}, // Sums to 0.6
			wantError:  true,
			errorMsg:   "allocation must sum to 1.0, got 0.600",
		},
		{
			name: "valid configuration",
			strategies: []Strategy{
				{ID: "s1", Name: "S1", TargetDim: 50},
				{ID: "s2", Name: "S2", TargetDim: 100},
			},
			allocation: []float64{0.4, 0.6},
			wantError:  false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := ABTestConfig{MinImpressions: 100}
			_, err := manager.CreateTest(config, tt.strategies, tt.allocation)
			
			if tt.wantError {
				if err == nil {
					t.Error("Expected error but got none")
				} else if tt.errorMsg != "" && err.Error() != tt.errorMsg {
					t.Errorf("Error message mismatch: got %q, want %q", err.Error(), tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

// TestRecordImpressionEdgeCases tests edge cases in RecordImpression
func TestRecordImpressionEdgeCases(t *testing.T) {
	manager := NewABTestManager(Strategy{ID: "default", Name: "Default", TargetDim: 100})
	ctx := context.Background()
	
	// Test 1: Record impression with no active test
	manager.RecordImpression(ctx, "s1", ImpressionMetrics{})
	// Should not panic or error
	
	// Test 2: Record impression for non-existent strategy
	strategies := []Strategy{
		{ID: "s1", Name: "Strategy 1", TargetDim: 50},
		{ID: "s2", Name: "Strategy 2", TargetDim: 100},
	}
	
	config := ABTestConfig{MinImpressions: 100}
	test, _ := manager.CreateTest(config, strategies, []float64{0.5, 0.5})
	manager.StartTest(test.ID)
	
	// Record for non-existent strategy
	manager.RecordImpression(ctx, "non-existent", ImpressionMetrics{})
	// Should not panic or error
}