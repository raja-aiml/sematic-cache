package validation

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidationResult_IsValid(t *testing.T) {
	tests := []struct {
		name   string
		result *ValidationResult
		want   bool
	}{
		{
			name: "valid_no_errors",
			result: &ValidationResult{
				Errors:   []string{},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
			},
			want: true,
		},
		{
			name: "invalid_with_errors",
			result: &ValidationResult{
				Errors:   []string{"error1", "error2"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
			},
			want: false,
		},
		{
			name:   "valid_empty_result",
			result: &ValidationResult{},
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.result.IsValid()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestNewValidationResult(t *testing.T) {
	result := NewValidationResult()

	assert.NotNil(t, result)
	assert.NotNil(t, result.Errors)
	assert.NotNil(t, result.Warnings)
	assert.NotNil(t, result.Info)
	assert.NotNil(t, result.Details)
	assert.Empty(t, result.Errors)
	assert.Empty(t, result.Warnings)
	assert.Empty(t, result.Info)
	assert.Empty(t, result.Details)
}

func TestValidationResult_AddError(t *testing.T) {
	result := NewValidationResult()

	result.AddError("error %s", "test")
	assert.Len(t, result.Errors, 1)
	assert.Equal(t, "error test", result.Errors[0])

	result.AddError("error %d: %s", 2, "another test")
	assert.Len(t, result.Errors, 2)
	assert.Equal(t, "error 2: another test", result.Errors[1])
}

func TestValidationResult_AddWarning(t *testing.T) {
	result := NewValidationResult()

	result.AddWarning("warning %s", "test")
	assert.Len(t, result.Warnings, 1)
	assert.Equal(t, "warning test", result.Warnings[0])

	result.AddWarning("warning %d: %s", 2, "another test")
	assert.Len(t, result.Warnings, 2)
	assert.Equal(t, "warning 2: another test", result.Warnings[1])
}

func TestValidationResult_AddInfo(t *testing.T) {
	result := NewValidationResult()

	result.AddInfo("info %s", "test")
	assert.Len(t, result.Info, 1)
	assert.Equal(t, "info test", result.Info[0])

	result.AddInfo("info %d: %s", 2, "another test")
	assert.Len(t, result.Info, 2)
	assert.Equal(t, "info 2: another test", result.Info[1])
}

func TestValidationResult_Merge(t *testing.T) {
	tests := []struct {
		name     string
		result   *ValidationResult
		other    *ValidationResult
		expected *ValidationResult
	}{
		{
			name: "merge_all_fields",
			result: &ValidationResult{
				Errors:   []string{"error1"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
				Details:  map[string]interface{}{"key1": "value1"},
			},
			other: &ValidationResult{
				Errors:   []string{"error2"},
				Warnings: []string{"warning2"},
				Info:     []string{"info2"},
				Details:  map[string]interface{}{"key2": "value2"},
			},
			expected: &ValidationResult{
				Errors:   []string{"error1", "error2"},
				Warnings: []string{"warning1", "warning2"},
				Info:     []string{"info1", "info2"},
				Details: map[string]interface{}{
					"key1": "value1",
					"key2": "value2",
				},
			},
		},
		{
			name:   "merge_empty_to_populated",
			result: NewValidationResult(),
			other: &ValidationResult{
				Errors:   []string{"error1"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
				Details:  map[string]interface{}{"key1": "value1"},
			},
			expected: &ValidationResult{
				Errors:   []string{"error1"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
				Details:  map[string]interface{}{"key1": "value1"},
			},
		},
		{
			name: "merge_empty_from_populated",
			result: &ValidationResult{
				Errors:   []string{"error1"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
				Details:  map[string]interface{}{"key1": "value1"},
			},
			other: NewValidationResult(),
			expected: &ValidationResult{
				Errors:   []string{"error1"},
				Warnings: []string{"warning1"},
				Info:     []string{"info1"},
				Details:  map[string]interface{}{"key1": "value1"},
			},
		},
		{
			name:   "merge_with_detail_override",
			result: NewValidationResult(),
			other: &ValidationResult{
				Details: map[string]interface{}{"key2": "value2", "shared": "new"},
			},
			expected: &ValidationResult{
				Errors:   []string{},
				Warnings: []string{},
				Info:     []string{},
				Details: map[string]interface{}{
					"key2":   "value2",
					"shared": "new",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.result.Merge(tt.other)
			assert.Equal(t, tt.expected, tt.result)
		})
	}
}

func TestDeploymentValidationOptions(t *testing.T) {
	opts := DeploymentValidationOptions{
		Namespace: "test-namespace",
		Scenario:  "minimal",
		Timeout:   30,
	}

	assert.Equal(t, "test-namespace", opts.Namespace)
	assert.Equal(t, "minimal", opts.Scenario)
	assert.Equal(t, 30, opts.Timeout)
}

func TestValidationResult_ComplexScenario(t *testing.T) {
	// Test a complex scenario with multiple operations
	result := NewValidationResult()

	// Add various messages
	result.AddError("Critical error: %s", "database connection failed")
	result.AddWarning("Performance warning: %s", "slow query detected")
	result.AddInfo("System info: %s", "version 1.0.0")

	// Add details
	result.Details["timestamp"] = "2024-01-01T00:00:00Z"
	result.Details["environment"] = "production"
	result.Details["metrics"] = map[string]interface{}{
		"cpu":    "80%",
		"memory": "4GB",
	}

	// Create another result to merge
	other := NewValidationResult()
	other.AddError("Network error: %s", "timeout")
	other.AddWarning("Resource warning: %s", "high memory usage")
	other.Details["node"] = "node-1"
	other.Details["metrics"] = map[string]interface{}{
		"disk": "90%",
	}

	// Merge results
	result.Merge(other)

	// Verify merged state
	assert.False(t, result.IsValid())
	assert.Len(t, result.Errors, 2)
	assert.Len(t, result.Warnings, 2)
	assert.Len(t, result.Info, 1)
	assert.Equal(t, "node-1", result.Details["node"])

	// Check that metrics was overridden (not merged)
	metrics, ok := result.Details["metrics"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "90%", metrics["disk"])
	assert.Nil(t, metrics["cpu"]) // Original metrics were replaced
}

func TestValidationResult_NilSafety(t *testing.T) {
	// Test that operations are safe with nil slices/maps
	var result ValidationResult

	// These should not panic
	assert.True(t, result.IsValid())
	result.AddError("error")
	result.AddWarning("warning")
	result.AddInfo("info")

	// Merge with another result that has nil fields
	var other ValidationResult
	result.Merge(&other)

	// Should still have our messages
	assert.Len(t, result.Errors, 1)
	assert.Len(t, result.Warnings, 1)
	assert.Len(t, result.Info, 1)
}
