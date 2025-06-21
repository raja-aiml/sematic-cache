package validation

import "fmt"

// ValidationResult holds the results of validation
type ValidationResult struct {
	Errors   []string               `json:"errors,omitempty"`
	Warnings []string               `json:"warnings,omitempty"`
	Info     []string               `json:"info,omitempty"`
	Details  map[string]interface{} `json:"details,omitempty"`
}

// IsValid returns true if there are no errors
func (v *ValidationResult) IsValid() bool {
	return len(v.Errors) == 0
}

// NewValidationResult creates a new validation result
func NewValidationResult() *ValidationResult {
	return &ValidationResult{
		Errors:   []string{},
		Warnings: []string{},
		Info:     []string{},
		Details:  make(map[string]interface{}),
	}
}

// AddError adds an error to the validation result
func (v *ValidationResult) AddError(format string, args ...interface{}) {
	v.Errors = append(v.Errors, fmt.Sprintf(format, args...))
}

// AddWarning adds a warning to the validation result
func (v *ValidationResult) AddWarning(format string, args ...interface{}) {
	v.Warnings = append(v.Warnings, fmt.Sprintf(format, args...))
}

// AddInfo adds an info message to the validation result
func (v *ValidationResult) AddInfo(format string, args ...interface{}) {
	v.Info = append(v.Info, fmt.Sprintf(format, args...))
}

// Merge combines another validation result into this one
func (v *ValidationResult) Merge(other *ValidationResult) {
	v.Errors = append(v.Errors, other.Errors...)
	v.Warnings = append(v.Warnings, other.Warnings...)
	v.Info = append(v.Info, other.Info...)
	
	for k, val := range other.Details {
		v.Details[k] = val
	}
}

// DeploymentValidationOptions contains options for deployment validation
type DeploymentValidationOptions struct {
	Namespace string
	Scenario  string
	Timeout   int
}

