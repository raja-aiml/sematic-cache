package secrets

import (
	"context"
	"fmt"
	"os"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// Manager handles Kubernetes secret operations
type Manager struct {
	k8sClient *kubernetes.Client
	logger    *utils.Logger
}

// NewManager creates a new secret manager
func NewManager(k8sClient *kubernetes.Client) *Manager {
	return &Manager{
		k8sClient: k8sClient,
		logger:    utils.NewLogger("secrets"),
	}
}

// SecretData represents the data for the application secret
type SecretData struct {
	OpenAIAPIKey string
	DatabaseURL  string
}

// EnsureAppSecrets creates or updates the application secrets
func (m *Manager) EnsureAppSecrets(ctx context.Context) error {
	// Load environment variables
	if err := utils.LoadEnvFile(); err != nil {
		m.logger.Warn("Failed to load .env file: %v", err)
	}

	// Get secret data
	data := m.getSecretData()

	// Convert to byte map
	secretData := map[string][]byte{
		"openai-api-key": []byte(data.OpenAIAPIKey),
		"database-url":   []byte(data.DatabaseURL),
	}

	// Try to update first, then create if it doesn't exist
	err := m.k8sClient.UpdateSecret(ctx, constants.AppNamespace, constants.AppSecretName, secretData)
	if err != nil {
		// Secret doesn't exist, create it
		err = m.k8sClient.CreateSecret(ctx, constants.AppNamespace, constants.AppSecretName, secretData)
		if err != nil {
			return fmt.Errorf("failed to create secret: %w", err)
		}
		m.logger.Info("Created secrets")
	} else {
		m.logger.Info("Updated secrets")
	}

	return nil
}

// CreateSecret creates a new secret
func (m *Manager) CreateSecret(ctx context.Context, namespace, name string, data map[string][]byte) error {
	if err := m.k8sClient.CreateSecret(ctx, namespace, name, data); err != nil {
		return fmt.Errorf("failed to create secret %s/%s: %w", namespace, name, err)
	}
	m.logger.Info("Created secret %s/%s", namespace, name)
	return nil
}

// UpdateSecret updates an existing secret
func (m *Manager) UpdateSecret(ctx context.Context, namespace, name string, data map[string][]byte) error {
	if err := m.k8sClient.UpdateSecret(ctx, namespace, name, data); err != nil {
		return fmt.Errorf("failed to update secret %s/%s: %w", namespace, name, err)
	}
	m.logger.Info("Updated secret %s/%s", namespace, name)
	return nil
}

// GetSecretData retrieves the secret data from environment
func (m *Manager) getSecretData() SecretData {
	data := SecretData{
		OpenAIAPIKey: os.Getenv("OPENAI_API_KEY"),
		DatabaseURL:  os.Getenv("DATABASE_URL"),
	}

	// Set defaults if not provided
	if data.OpenAIAPIKey == "" {
		m.logger.Warn("OPENAI_API_KEY not set, application may not function properly")
		data.OpenAIAPIKey = "dummy-key-for-testing"
	}

	if data.DatabaseURL == "" {
		data.DatabaseURL = constants.DefaultDatabaseURL
		m.logger.Info("Using default DATABASE_URL for local deployment")
	}

	return data
}

// ValidateSecrets checks if required secrets exist
func (m *Manager) ValidateSecrets(ctx context.Context) error {
	// Check if secret exists
	secretData := map[string][]byte{}
	err := m.k8sClient.UpdateSecret(ctx, constants.AppNamespace, constants.AppSecretName, secretData)
	if err != nil {
		return fmt.Errorf("secret %s/%s does not exist", constants.AppNamespace, constants.AppSecretName)
	}

	m.logger.Info("Secret validation passed")
	return nil
}
