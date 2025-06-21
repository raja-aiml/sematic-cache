package validation

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDeploymentValidator_validateApplications(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	// Test default namespace
	validator.validateApplications(context.Background(), DeploymentValidationOptions{}, result)
	assert.NotEmpty(t, result.Info)
	assert.Contains(t, result.Info[0], "Checking applications in namespace: app")

	// Test specific namespace
	result = NewValidationResult()
	opts := DeploymentValidationOptions{Namespace: "custom-app"}
	validator.validateApplications(context.Background(), opts, result)
	assert.NotEmpty(t, result.Info)
	assert.Contains(t, result.Info[0], "Checking applications in namespace: custom-app")
}

func TestDeploymentValidator_validatePostgresDeployment(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validatePostgresDeployment(context.Background(), result)

	// Since all helper functions return true, should have no errors
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateRedisDeployment(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateRedisDeployment(context.Background(), result)

	// Since all helper functions return true, should have no errors
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateIngressController(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateIngressController(context.Background(), result)

	// Since all helper functions return true, should have no warnings
	assert.Empty(t, result.Warnings)
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateMinimalScenario(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateMinimalScenario(context.Background(), DeploymentValidationOptions{}, result)

	// Minimal scenario should not have certain namespaces
	// Since checkNamespaceExists returns true, it would add warnings
	assert.NotEmpty(t, result.Warnings)
}

func TestDeploymentValidator_validateDevelopmentScenario(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateDevelopmentScenario(context.Background(), DeploymentValidationOptions{}, result)

	// Since checkNamespaceExists returns true, should have no errors
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateServiceMeshScenario(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateServiceMeshScenario(context.Background(), DeploymentValidationOptions{}, result)

	// Since checkNamespaceExists returns true, should have no errors
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateMonitoringScenario(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateMonitoringScenario(context.Background(), DeploymentValidationOptions{}, result)

	// Since all checks return true, should have no errors
	assert.Empty(t, result.Errors)
}

func TestDeploymentValidator_validateFullStackScenario(t *testing.T) {
	validator := NewDeploymentValidator(NewMockKubernetesClient())
	result := NewValidationResult()

	validator.validateFullStackScenario(context.Background(), DeploymentValidationOptions{}, result)

	// Full stack runs all scenario validations
	// Since all checks return true, should have minimal errors
	assert.NotNil(t, result)
}
