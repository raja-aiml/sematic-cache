package reduction

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// ABTestManager manages A/B testing for dimension reduction
type ABTestManager struct {
	mu              sync.RWMutex
	tests           map[string]*ABTest
	activeTestID    string
	defaultStrategy Strategy
}

// ABTest represents an active A/B test
type ABTest struct {
	ID          string
	Name        string
	Description string
	StartTime   time.Time
	EndTime     time.Time
	Status      TestStatus
	Strategies  []Strategy
	Allocation  []float64 // Traffic allocation percentages
	Results     map[string]*TestResults
	Config      ABTestConfig
	mu          sync.RWMutex
}

// TestStatus represents the status of an A/B test
type TestStatus string

const (
	TestStatusPending  TestStatus = "pending"
	TestStatusRunning  TestStatus = "running"
	TestStatusComplete TestStatus = "complete"
	TestStatusStopped  TestStatus = "stopped"
)

// Strategy represents a dimension reduction strategy
type Strategy struct {
	ID        string
	Name      string
	TargetDim int
	Algorithm string // "pca", "autoencoder", "none"
	UseHybrid bool
	Config    map[string]interface{}
}

// TestResults tracks metrics for a strategy
type TestResults struct {
	Strategy        Strategy
	Impressions     int64
	CacheHits       int64
	CacheMisses     int64
	TotalLatencyMs  int64
	SearchLatencyMs int64
	MemoryUsedMB    uint64 // Stored as uint64 for atomic operations (convert from/to float64)
	Errors          int64

	// Quality metrics - stored as atomic uint64, converted from/to float64
	AvgSimilarityScore   uint64 // Use math.Float64bits/Float64frombits for conversion
	SimilarityScoreSum   uint64 // Use math.Float64bits/Float64frombits for conversion
	SimilarityScoreCount int64

	// User satisfaction metrics
	SuccessfulQueries int64
	FailedQueries     int64
}

// ABTestConfig configures an A/B test
type ABTestConfig struct {
	MinImpressions        int64
	MinDurationHours      float64
	ConfidenceLevel       float64
	SignificanceThreshold float64
	MetricWeights         MetricWeights
}

// MetricWeights defines importance of different metrics
type MetricWeights struct {
	HitRate float64
	Latency float64
	Memory  float64
	Quality float64
}

// NewABTestManager creates a new A/B test manager
func NewABTestManager(defaultStrategy Strategy) *ABTestManager {
	return &ABTestManager{
		tests:           make(map[string]*ABTest),
		defaultStrategy: defaultStrategy,
	}
}

// CreateTest creates a new A/B test
func (m *ABTestManager) CreateTest(config ABTestConfig, strategies []Strategy, allocation []float64) (*ABTest, error) {
	if len(strategies) < 2 {
		return nil, fmt.Errorf("need at least 2 strategies for A/B test")
	}

	if len(allocation) != len(strategies) {
		return nil, fmt.Errorf("allocation length must match strategies")
	}

	// Validate allocation sums to 1.0
	sum := 0.0
	for _, a := range allocation {
		sum += a
	}
	if math.Abs(sum-1.0) > 0.001 {
		return nil, fmt.Errorf("allocation must sum to 1.0, got %.3f", sum)
	}

	testID := fmt.Sprintf("test_%d", time.Now().Unix())

	results := make(map[string]*TestResults)
	for _, strategy := range strategies {
		results[strategy.ID] = &TestResults{
			Strategy: strategy,
		}
	}

	test := &ABTest{
		ID:         testID,
		StartTime:  time.Now(),
		Status:     TestStatusPending,
		Strategies: strategies,
		Allocation: allocation,
		Results:    results,
		Config:     config,
	}

	m.mu.Lock()
	m.tests[testID] = test
	m.mu.Unlock()

	return test, nil
}

// StartTest starts an A/B test
func (m *ABTestManager) StartTest(testID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	test, exists := m.tests[testID]
	if !exists {
		return fmt.Errorf("test %s not found", testID)
	}

	test.mu.Lock()
	test.Status = TestStatusRunning
	test.StartTime = time.Now()
	test.mu.Unlock()

	m.activeTestID = testID
	return nil
}

// GetStrategy returns the strategy to use for a given request
func (m *ABTestManager) GetStrategy(ctx context.Context, requestID string) Strategy {
	m.mu.RLock()
	activeTest := m.activeTestID
	m.mu.RUnlock()

	if activeTest == "" {
		return m.defaultStrategy
	}

	test := m.getTest(activeTest)
	if test == nil || test.Status != TestStatusRunning {
		return m.defaultStrategy
	}

	// Use hash of requestID for consistent assignment
	hash := hashString(requestID)
	bucket := float64(hash%1000) / 1000.0

	// Find which strategy bucket this falls into
	cumulative := 0.0
	for i, allocation := range test.Allocation {
		cumulative += allocation
		if bucket < cumulative {
			return test.Strategies[i]
		}
	}

	return test.Strategies[len(test.Strategies)-1]
}

// RecordImpression records an impression for a strategy
func (m *ABTestManager) RecordImpression(ctx context.Context, strategyID string, metrics ImpressionMetrics) {
	test := m.getActiveTest()
	if test == nil {
		return
	}

	test.mu.RLock()
	results, exists := test.Results[strategyID]
	test.mu.RUnlock()

	if !exists {
		return
	}

	// Update metrics atomically
	atomic.AddInt64(&results.Impressions, 1)

	if metrics.CacheHit {
		atomic.AddInt64(&results.CacheHits, 1)
	} else {
		atomic.AddInt64(&results.CacheMisses, 1)
	}

	atomic.AddInt64(&results.TotalLatencyMs, metrics.LatencyMs)
	atomic.AddInt64(&results.SearchLatencyMs, metrics.SearchLatencyMs)

	if metrics.Error != nil {
		atomic.AddInt64(&results.Errors, 1)
		atomic.AddInt64(&results.FailedQueries, 1)
	} else {
		atomic.AddInt64(&results.SuccessfulQueries, 1)
	}

	// Update similarity score atomically
	for {
		oldSum := atomic.LoadUint64(&results.SimilarityScoreSum)
		oldSumFloat := math.Float64frombits(oldSum)
		newSumFloat := oldSumFloat + metrics.SimilarityScore
		newSum := math.Float64bits(newSumFloat)
		if atomic.CompareAndSwapUint64(&results.SimilarityScoreSum, oldSum, newSum) {
			// Successfully updated sum, now update count
			newCount := atomic.AddInt64(&results.SimilarityScoreCount, 1)
			// Update average
			avg := newSumFloat / float64(newCount)
			atomic.StoreUint64(&results.AvgSimilarityScore, math.Float64bits(avg))
			break
		}
	}
}

// ImpressionMetrics contains metrics for a single impression
type ImpressionMetrics struct {
	CacheHit        bool
	LatencyMs       int64
	SearchLatencyMs int64
	SimilarityScore float64
	Error           error
}

// CheckTestCompletion checks if test should be completed
func (m *ABTestManager) CheckTestCompletion(testID string) (bool, *TestSummary) {
	test := m.getTest(testID)
	if test == nil {
		return false, nil
	}

	test.mu.RLock()
	defer test.mu.RUnlock()

	// Check minimum impressions
	totalImpressions := int64(0)
	for _, results := range test.Results {
		totalImpressions += atomic.LoadInt64(&results.Impressions)
	}

	if totalImpressions < test.Config.MinImpressions {
		return false, nil
	}

	// Check minimum duration
	duration := time.Since(test.StartTime).Hours()
	if duration < test.Config.MinDurationHours {
		return false, nil
	}

	// Calculate summary
	summary := m.calculateTestSummary(test)

	// Check if we have statistical significance
	if summary.StatisticalSignificance < test.Config.SignificanceThreshold {
		return false, summary
	}

	return true, summary
}

// TestSummary contains the summary of test results
type TestSummary struct {
	TestID                  string
	Duration                time.Duration
	TotalImpressions        int64
	WinningStrategy         Strategy
	StatisticalSignificance float64
	Results                 []StrategyResult
}

// StrategyResult contains results for a single strategy
type StrategyResult struct {
	Strategy           Strategy
	Impressions        int64
	HitRate            float64
	AvgLatencyMs       float64
	AvgSearchLatencyMs float64
	AvgSimilarityScore float64
	ErrorRate          float64
	Score              float64 // Weighted score
}

// calculateTestSummary calculates the test summary
func (m *ABTestManager) calculateTestSummary(test *ABTest) *TestSummary {
	summary := &TestSummary{
		TestID:   test.ID,
		Duration: time.Since(test.StartTime),
	}

	// Calculate metrics for each strategy
	for _, strategy := range test.Strategies {
		results := test.Results[strategy.ID]

		impressions := atomic.LoadInt64(&results.Impressions)
		if impressions == 0 {
			continue
		}

		hits := atomic.LoadInt64(&results.CacheHits)
		misses := atomic.LoadInt64(&results.CacheMisses)
		totalLatency := atomic.LoadInt64(&results.TotalLatencyMs)
		searchLatency := atomic.LoadInt64(&results.SearchLatencyMs)
		errors := atomic.LoadInt64(&results.Errors)

		stratResult := StrategyResult{
			Strategy:           strategy,
			Impressions:        impressions,
			HitRate:            float64(hits) / float64(hits+misses),
			AvgLatencyMs:       float64(totalLatency) / float64(impressions),
			AvgSearchLatencyMs: float64(searchLatency) / float64(impressions),
			AvgSimilarityScore: math.Float64frombits(atomic.LoadUint64(&results.AvgSimilarityScore)),
			ErrorRate:          float64(errors) / float64(impressions),
		}

		// Calculate weighted score
		weights := test.Config.MetricWeights
		stratResult.Score = weights.HitRate*stratResult.HitRate +
			weights.Latency*(1.0-stratResult.AvgLatencyMs/100.0) + // Normalize latency
			weights.Quality*stratResult.AvgSimilarityScore +
			weights.Memory*(1.0-float64(strategy.TargetDim)/1536.0) // Normalize by original dim

		summary.Results = append(summary.Results, stratResult)
		summary.TotalImpressions += impressions
	}

	// Find winning strategy
	if len(summary.Results) > 0 {
		maxScore := -1.0
		for _, result := range summary.Results {
			if result.Score > maxScore {
				maxScore = result.Score
				summary.WinningStrategy = result.Strategy
			}
		}

		// Calculate statistical significance (simplified)
		summary.StatisticalSignificance = m.calculateSignificance(summary.Results)
	}

	return summary
}

// calculateSignificance calculates statistical significance (simplified)
func (m *ABTestManager) calculateSignificance(results []StrategyResult) float64 {
	if len(results) < 2 {
		return 0.0
	}

	// Simple z-test for hit rate difference
	r1 := results[0]
	r2 := results[1]

	p1 := r1.HitRate
	p2 := r2.HitRate
	n1 := float64(r1.Impressions)
	n2 := float64(r2.Impressions)

	pooledP := (p1*n1 + p2*n2) / (n1 + n2)
	se := math.Sqrt(pooledP * (1 - pooledP) * (1/n1 + 1/n2))

	if se == 0 {
		return 0.0
	}

	z := math.Abs(p1-p2) / se

	// Convert z-score to confidence level
	confidence := 2 * (1 - normalCDF(z))
	return 1 - confidence
}

// StopTest stops an A/B test
func (m *ABTestManager) StopTest(testID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	test, exists := m.tests[testID]
	if !exists {
		return fmt.Errorf("test %s not found", testID)
	}

	test.mu.Lock()
	test.Status = TestStatusStopped
	test.EndTime = time.Now()
	test.mu.Unlock()

	if m.activeTestID == testID {
		m.activeTestID = ""
	}

	return nil
}

// GetActiveTest returns the currently active test
func (m *ABTestManager) getActiveTest() *ABTest {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.activeTestID == "" {
		return nil
	}

	return m.tests[m.activeTestID]
}

// getTest returns a test by ID
func (m *ABTestManager) getTest(testID string) *ABTest {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.tests[testID]
}

// ExportResults exports test results for analysis
func (m *ABTestManager) ExportResults(testID string) (*TestExport, error) {
	test := m.getTest(testID)
	if test == nil {
		return nil, fmt.Errorf("test %s not found", testID)
	}

	summary := m.calculateTestSummary(test)

	export := &TestExport{
		Test:      test, // Store pointer to avoid copying mutex
		Summary:   summary,
		Timestamp: time.Now(),
	}

	return export, nil
}

// TestExport contains exportable test data
type TestExport struct {
	Test      *ABTest
	Summary   *TestSummary
	Timestamp time.Time
}

// Helper functions
func hashString(s string) uint32 {
	h := uint32(0)
	for _, c := range s {
		h = h*31 + uint32(c)
	}
	return h
}

func normalCDF(x float64) float64 {
	// Approximation of normal CDF
	return 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
}

// SetMemoryUsedMB atomically sets the memory usage
func (r *TestResults) SetMemoryUsedMB(mb float64) {
	atomic.StoreUint64(&r.MemoryUsedMB, math.Float64bits(mb))
}

// GetMemoryUsedMB atomically gets the memory usage
func (r *TestResults) GetMemoryUsedMB() float64 {
	return math.Float64frombits(atomic.LoadUint64(&r.MemoryUsedMB))
}

// MonitoringDashboard provides real-time test monitoring
type MonitoringDashboard struct {
	manager *ABTestManager
}

// NewMonitoringDashboard creates a monitoring dashboard
func NewMonitoringDashboard(manager *ABTestManager) *MonitoringDashboard {
	return &MonitoringDashboard{manager: manager}
}

// GetLiveMetrics returns live metrics for active test
func (d *MonitoringDashboard) GetLiveMetrics() *LiveMetrics {
	test := d.manager.getActiveTest()
	if test == nil {
		return nil
	}

	metrics := &LiveMetrics{
		TestID:     test.ID,
		StartTime:  test.StartTime,
		Duration:   time.Since(test.StartTime),
		Strategies: make([]LiveStrategyMetrics, 0),
	}

	for _, strategy := range test.Strategies {
		results := test.Results[strategy.ID]

		impressions := atomic.LoadInt64(&results.Impressions)
		hits := atomic.LoadInt64(&results.CacheHits)
		misses := atomic.LoadInt64(&results.CacheMisses)

		hitRate := 0.0
		if impressions > 0 {
			hitRate = float64(hits) / float64(hits+misses)
		}

		stratMetrics := LiveStrategyMetrics{
			StrategyID:   strategy.ID,
			StrategyName: strategy.Name,
			Impressions:  impressions,
			HitRate:      hitRate,
			ErrorRate:    float64(atomic.LoadInt64(&results.Errors)) / float64(impressions),
		}

		metrics.Strategies = append(metrics.Strategies, stratMetrics)
	}

	return metrics
}

// LiveMetrics contains real-time test metrics
type LiveMetrics struct {
	TestID     string
	StartTime  time.Time
	Duration   time.Duration
	Strategies []LiveStrategyMetrics
}

// LiveStrategyMetrics contains real-time metrics for a strategy
type LiveStrategyMetrics struct {
	StrategyID   string
	StrategyName string
	Impressions  int64
	HitRate      float64
	ErrorRate    float64
}
