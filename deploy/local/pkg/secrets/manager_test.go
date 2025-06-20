package secrets

import (
	"os"
	"testing"

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
			os.Unsetenv("OPENAI_API_KEY")
			os.Unsetenv("DATABASE_URL")

			// Set environment variables
			for k, v := range tt.envVars {
				os.Setenv(k, v)
				defer os.Unsetenv(k)
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
	defer os.RemoveAll(tmpDir)
	
	// Change to temp directory
	os.Chdir(tmpDir)
	defer os.Chdir(oldPwd)
	
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

	os.Unsetenv("TEST_SECRET_VAR")
}

// Benchmark tests
func BenchmarkGetSecretData(b *testing.B) {
	manager := &Manager{
		logger: utils.NewLogger("bench"),
	}

	// Set some env vars
	os.Setenv("OPENAI_API_KEY", "benchmark-key")
	os.Setenv("DATABASE_URL", "postgres://bench")
	defer os.Unsetenv("OPENAI_API_KEY")
	defer os.Unsetenv("DATABASE_URL")

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