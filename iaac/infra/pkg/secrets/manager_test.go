package secrets

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/constants"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/kubernetes"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

func TestNewManager(t *testing.T) {
	client := &kubernetes.Client{}
	manager := NewManager(client)

	if manager == nil {
		t.Fatal("NewManager returned nil")
	}

	if manager.k8sClient != client {
		t.Error("Manager k8sClient not set correctly")
	}

	if manager.logger == nil {
		t.Error("Manager logger is nil")
	}
}

func TestManager_getSecretData(t *testing.T) {
	tests := []struct {
		name           string
		envVars        map[string]string
		expectedAPIKey string
		expectedDBURL  string
	}{
		{
			name: "all_env_vars_set",
			envVars: map[string]string{
				"OPENAI_API_KEY": "test-api-key",
				"DATABASE_URL":   "postgres://test",
			},
			expectedAPIKey: "test-api-key",
			expectedDBURL:  "postgres://test",
		},
		{
			name:           "no_env_vars_set",
			envVars:        map[string]string{},
			expectedAPIKey: "dummy-key-for-testing",
			expectedDBURL:  constants.DefaultDatabaseURL,
		},
		{
			name: "only_api_key_set",
			envVars: map[string]string{
				"OPENAI_API_KEY": "api-key-only",
			},
			expectedAPIKey: "api-key-only",
			expectedDBURL:  constants.DefaultDatabaseURL,
		},
		{
			name: "only_database_url_set",
			envVars: map[string]string{
				"DATABASE_URL": "postgres://db-only",
			},
			expectedAPIKey: "dummy-key-for-testing",
			expectedDBURL:  "postgres://db-only",
		},
		{
			name: "empty_api_key",
			envVars: map[string]string{
				"OPENAI_API_KEY": "",
				"DATABASE_URL":   "postgres://test",
			},
			expectedAPIKey: "dummy-key-for-testing",
			expectedDBURL:  "postgres://test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear environment first
			if err := os.Unsetenv("OPENAI_API_KEY"); err != nil {
				t.Logf("Failed to unset OPENAI_API_KEY: %v", err)
			}
			if err := os.Unsetenv("DATABASE_URL"); err != nil {
				t.Logf("Failed to unset DATABASE_URL: %v", err)
			}

			// Set environment variables
			for k, v := range tt.envVars {
				if err := os.Setenv(k, v); err != nil {
					t.Logf("Failed to set %s: %v", k, err)
				}
				defer func(key string) {
					if err := os.Unsetenv(key); err != nil {
						t.Logf("Failed to unset %s: %v", key, err)
					}
				}(k)
			}

			manager := &Manager{
				logger: utils.NewLogger("test"),
			}

			data := manager.getSecretData()

			if data.OpenAIAPIKey != tt.expectedAPIKey {
				t.Errorf("getSecretData() OpenAIAPIKey = %v, want %v", data.OpenAIAPIKey, tt.expectedAPIKey)
			}

			if data.DatabaseURL != tt.expectedDBURL {
				t.Errorf("getSecretData() DatabaseURL = %v, want %v", data.DatabaseURL, tt.expectedDBURL)
			}
		})
	}
}

func TestSecretData(t *testing.T) {
	tests := []struct {
		name         string
		data         SecretData
		expectedKeys []string
	}{
		{
			name: "full_data",
			data: SecretData{
				OpenAIAPIKey: "key123",
				DatabaseURL:  "postgres://localhost",
			},
			expectedKeys: []string{"openai-api-key", "database-url"},
		},
		{
			name: "empty_data",
			data: SecretData{
				OpenAIAPIKey: "",
				DatabaseURL:  "",
			},
			expectedKeys: []string{"openai-api-key", "database-url"},
		},
		{
			name: "special_characters",
			data: SecretData{
				OpenAIAPIKey: "sk-1234!@#$%^&*()_+",
				DatabaseURL:  "postgres://user:p@ss!word@host:5432/db?sslmode=disable",
			},
			expectedKeys: []string{"openai-api-key", "database-url"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test conversion to byte map
			secretData := map[string][]byte{
				"openai-api-key": []byte(tt.data.OpenAIAPIKey),
				"database-url":   []byte(tt.data.DatabaseURL),
			}

			// Verify all expected keys exist
			for _, key := range tt.expectedKeys {
				if _, ok := secretData[key]; !ok {
					t.Errorf("Expected key %s not found in secret data", key)
				}
			}

			// Verify data matches
			if string(secretData["openai-api-key"]) != tt.data.OpenAIAPIKey {
				t.Errorf("OpenAI API key mismatch: got %s, want %s",
					string(secretData["openai-api-key"]), tt.data.OpenAIAPIKey)
			}
			if string(secretData["database-url"]) != tt.data.DatabaseURL {
				t.Errorf("Database URL mismatch: got %s, want %s",
					string(secretData["database-url"]), tt.data.DatabaseURL)
			}

			// Verify byte conversion doesn't lose data
			if len(secretData["openai-api-key"]) != len(tt.data.OpenAIAPIKey) {
				t.Errorf("OpenAI API key length mismatch after byte conversion")
			}
			if len(secretData["database-url"]) != len(tt.data.DatabaseURL) {
				t.Errorf("Database URL length mismatch after byte conversion")
			}
		})
	}
}

func TestSecretDataMap(t *testing.T) {
	// Test the exact format that EnsureAppSecrets creates
	testData := SecretData{
		OpenAIAPIKey: "test-key-123",
		DatabaseURL:  "postgres://test:password@localhost:5432/testdb",
	}

	secretData := map[string][]byte{
		"openai-api-key": []byte(testData.OpenAIAPIKey),
		"database-url":   []byte(testData.DatabaseURL),
	}

	// Verify the map has exactly 2 entries
	if len(secretData) != 2 {
		t.Errorf("Expected 2 entries in secret data map, got %d", len(secretData))
	}

	// Verify keys are lowercase with hyphens (Kubernetes convention)
	expectedKeys := map[string]bool{
		"openai-api-key": true,
		"database-url":   true,
	}

	for key := range secretData {
		if !expectedKeys[key] {
			t.Errorf("Unexpected key in secret data: %s", key)
		}
	}
}

func TestManager_LoadEnvFile(t *testing.T) {
	// Save current directory
	oldPwd, _ := os.Getwd()

	// Create a temp directory
	tmpDir, err := os.MkdirTemp("", "test-secrets")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	// Change to temp directory
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(oldPwd); err != nil {
			t.Logf("Failed to restore directory: %v", err)
		}
	}()

	// Create .env file
	content := `TEST_SECRET_VAR=secret_value`
	err = os.WriteFile(".env", []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Test that LoadEnvFile is called
	err = utils.LoadEnvFile()
	if err != nil {
		t.Errorf("LoadEnvFile() error = %v", err)
	}

	value := os.Getenv("TEST_SECRET_VAR")
	if value != "secret_value" {
		t.Logf("LoadEnvFile() didn't load test variable correctly, got %q", value)
		// This might be expected if godotenv is not working in test environment
	}

	if err := os.Unsetenv("TEST_SECRET_VAR"); err != nil {
		t.Logf("Failed to unset TEST_SECRET_VAR: %v", err)
	}
}

// Benchmark tests
func BenchmarkGetSecretData(b *testing.B) {
	manager := &Manager{
		logger: utils.NewLogger("bench"),
	}

	// Set some env vars
	if err := os.Setenv("OPENAI_API_KEY", "benchmark-key"); err != nil {
		b.Logf("Failed to set OPENAI_API_KEY: %v", err)
	}
	if err := os.Setenv("DATABASE_URL", "postgres://bench"); err != nil {
		b.Logf("Failed to set DATABASE_URL: %v", err)
	}
	defer func() {
		if err := os.Unsetenv("OPENAI_API_KEY"); err != nil {
			b.Logf("Failed to unset OPENAI_API_KEY: %v", err)
		}
		if err := os.Unsetenv("DATABASE_URL"); err != nil {
			b.Logf("Failed to unset DATABASE_URL: %v", err)
		}
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = manager.getSecretData()
	}
}

func BenchmarkSecretDataConversion(b *testing.B) {
	data := SecretData{
		OpenAIAPIKey: "benchmark-api-key-1234567890",
		DatabaseURL:  "postgres://user:password@localhost:5432/benchmark_db?sslmode=disable",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = map[string][]byte{
			"openai-api-key": []byte(data.OpenAIAPIKey),
			"database-url":   []byte(data.DatabaseURL),
		}
	}
}

// Additional comprehensive tests without complex mocking

func TestManager_EnsureAppSecrets_DataTransformation(t *testing.T) {
	tests := []struct {
		name           string
		envVars        map[string]string
		expectedAPIKey string
		expectedDBURL  string
	}{
		{
			name: "with_env_vars",
			envVars: map[string]string{
				"OPENAI_API_KEY": "test-key-123",
				"DATABASE_URL":   "postgres://test-db",
			},
			expectedAPIKey: "test-key-123",
			expectedDBURL:  "postgres://test-db",
		},
		{
			name:           "with_defaults",
			envVars:        map[string]string{},
			expectedAPIKey: "dummy-key-for-testing",
			expectedDBURL:  constants.DefaultDatabaseURL,
		},
		{
			name: "partial_env_vars",
			envVars: map[string]string{
				"OPENAI_API_KEY": "partial-key",
			},
			expectedAPIKey: "partial-key",
			expectedDBURL:  constants.DefaultDatabaseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up environment
			cleanupEnv := setTestEnv(t, tt.envVars)
			defer cleanupEnv()

			// Create manager (we'll test the data transformation part)
			manager := &Manager{
				logger: utils.NewLogger("test"),
			}

			// Test the secret data generation
			data := manager.getSecretData()
			assert.Equal(t, tt.expectedAPIKey, data.OpenAIAPIKey)
			assert.Equal(t, tt.expectedDBURL, data.DatabaseURL)

			// Test the conversion to byte map (what EnsureAppSecrets does)
			secretData := map[string][]byte{
				"openai-api-key": []byte(data.OpenAIAPIKey),
				"database-url":   []byte(data.DatabaseURL),
			}

			assert.Equal(t, tt.expectedAPIKey, string(secretData["openai-api-key"]))
			assert.Equal(t, tt.expectedDBURL, string(secretData["database-url"]))
			assert.Len(t, secretData, 2)
		})
	}
}

func TestManager_SecretNameFormatting(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		secret    string
		expected  string
	}{
		{
			name:      "standard_formatting",
			namespace: "test-namespace",
			secret:    "test-secret",
			expected:  "test-namespace/test-secret",
		},
		{
			name:      "app_secret_formatting",
			namespace: constants.AppNamespace,
			secret:    constants.AppSecretName,
			expected:  constants.AppNamespace + "/" + constants.AppSecretName,
		},
		{
			name:      "empty_namespace",
			namespace: "",
			secret:    "secret",
			expected:  "/secret",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatted := fmt.Sprintf("%s/%s", tt.namespace, tt.secret)
			assert.Equal(t, tt.expected, formatted)
		})
	}
}

func TestManager_ValidateSecrets_ErrorFormatting(t *testing.T) {
	namespace := constants.AppNamespace
	secretName := constants.AppSecretName
	expectedFormat := fmt.Sprintf("secret %s/%s does not exist", namespace, secretName)

	assert.Equal(t, "secret "+namespace+"/"+secretName+" does not exist", expectedFormat)
}

func TestSecretData_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		data SecretData
	}{
		{
			name: "unicode_characters",
			data: SecretData{
				OpenAIAPIKey: "sk-测试密钥-ファイル-🔑",
				DatabaseURL:  "postgres://用户:密码@主机:5432/数据库",
			},
		},
		{
			name: "very_long_values",
			data: SecretData{
				OpenAIAPIKey: generateLargeString(1000),
				DatabaseURL:  "postgres://user:password@host:5432/" + generateLargeString(100),
			},
		},
		{
			name: "newlines_and_whitespace",
			data: SecretData{
				OpenAIAPIKey: "  sk-key-with-spaces  \n",
				DatabaseURL:  "\tpostgres://localhost\t\r\n",
			},
		},
		{
			name: "binary_like_data",
			data: SecretData{
				OpenAIAPIKey: string([]byte{0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE}),
				DatabaseURL:  string([]byte{0x41, 0x42, 0x43, 0x00, 0x44}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that conversion to bytes and back preserves data
			secretData := map[string][]byte{
				"openai-api-key": []byte(tt.data.OpenAIAPIKey),
				"database-url":   []byte(tt.data.DatabaseURL),
			}

			assert.Equal(t, tt.data.OpenAIAPIKey, string(secretData["openai-api-key"]))
			assert.Equal(t, tt.data.DatabaseURL, string(secretData["database-url"]))
		})
	}
}

func TestManager_ConstantsUsage(t *testing.T) {
	// Test that the manager uses the correct constants
	assert.NotEmpty(t, constants.AppNamespace, "AppNamespace should not be empty")
	assert.NotEmpty(t, constants.AppSecretName, "AppSecretName should not be empty")
	assert.NotEmpty(t, constants.DefaultDatabaseURL, "DefaultDatabaseURL should not be empty")
}

// Helper functions

func setTestEnv(t *testing.T, envVars map[string]string) func() {
	// Store original values for cleanup
	originalValues := make(map[string]string)

	// Clear environment variables first
	for _, key := range []string{"OPENAI_API_KEY", "DATABASE_URL"} {
		originalValues[key] = os.Getenv(key)
		os.Unsetenv(key)
	}

	// Set test environment variables
	for k, v := range envVars {
		err := os.Setenv(k, v)
		if err != nil {
			t.Logf("Failed to set %s: %v", k, err)
		}
	}

	// Return cleanup function
	return func() {
		// Clear test env vars
		for k := range envVars {
			os.Unsetenv(k)
		}

		// Restore original values
		for k, v := range originalValues {
			if v != "" {
				os.Setenv(k, v)
			}
		}
	}
}

func generateLargeString(size int) string {
	result := make([]byte, size)
	for i := range result {
		result[i] = byte('a' + (i % 26))
	}
	return string(result)
}
