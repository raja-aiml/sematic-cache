package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadConfig(t *testing.T) {
	tests := []struct {
		name        string
		envVars     map[string]string
		wantErr     bool
		errContains string
		validate    func(t *testing.T, cfg *Config)
	}{
		{
			name: "valid configuration with required fields",
			envVars: map[string]string{
				"DATABASE_URL":   "postgresql://user:pass@localhost/db",
				"OPENAI_API_KEY": "sk-test-key",
			},
			wantErr: false,
			validate: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "postgresql://user:pass@localhost/db", cfg.DatabaseURL)
				assert.Equal(t, "sk-test-key", cfg.OpenAIAPIKey)
				assert.Equal(t, 0.8, cfg.SimilarityThreshold) // default value
			},
		},
		{
			name: "missing DATABASE_URL",
			envVars: map[string]string{
				"OPENAI_API_KEY": "sk-test-key",
			},
			wantErr:     true,
			errContains: "DATABASE_URL is required",
		},
		{
			name: "missing OPENAI_API_KEY",
			envVars: map[string]string{
				"DATABASE_URL": "postgresql://user:pass@localhost/db",
			},
			wantErr:     true,
			errContains: "OPENAI_API_KEY is required",
		},
		{
			name: "invalid similarity threshold",
			envVars: map[string]string{
				"DATABASE_URL":         "postgresql://user:pass@localhost/db",
				"OPENAI_API_KEY":       "sk-test-key",
				"SIMILARITY_THRESHOLD": "1.5",
			},
			wantErr:     true,
			errContains: "SIMILARITY_THRESHOLD must be between 0 and 1",
		},
		{
			name: "custom values override defaults",
			envVars: map[string]string{
				"DATABASE_URL":             "postgresql://custom:pass@custom/db",
				"OPENAI_API_KEY":           "sk-custom-key",
				"SERVER_PORT":              "9090",
				"LOG_LEVEL":                "debug",
				"SIMILARITY_THRESHOLD":     "0.9",
				"DATABASE_MAX_CONNECTIONS": "50",
			},
			wantErr: false,
			validate: func(t *testing.T, cfg *Config) {
				assert.Equal(t, "postgresql://custom:pass@custom/db", cfg.DatabaseURL)
				assert.Equal(t, "sk-custom-key", cfg.OpenAIAPIKey)
				assert.Equal(t, "9090", cfg.ServerPort)
				assert.Equal(t, "debug", cfg.LogLevel)
				assert.Equal(t, 0.9, cfg.SimilarityThreshold)
				assert.Equal(t, 50, cfg.DatabaseMaxConnections)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variables
			for k, v := range tt.envVars {
				os.Setenv(k, v)
				defer os.Unsetenv(k)
			}

			// Load configuration
			cfg, err := LoadConfig()

			if tt.wantErr {
				require.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
			} else {
				require.NoError(t, err)
				require.NotNil(t, cfg)
				if tt.validate != nil {
					tt.validate(t, cfg)
				}
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		config      Config
		wantErr     bool
		errContains string
	}{
		{
			name: "valid configuration",
			config: Config{
				DatabaseURL:         "postgresql://user:pass@localhost/db",
				OpenAIAPIKey:        "sk-test-key",
				SimilarityThreshold: 0.8,
				VectorDimensions:    1536,
			},
			wantErr: false,
		},
		{
			name: "missing database URL",
			config: Config{
				OpenAIAPIKey:        "sk-test-key",
				SimilarityThreshold: 0.8,
				VectorDimensions:    1536,
			},
			wantErr:     true,
			errContains: "DATABASE_URL is required",
		},
		{
			name: "missing OpenAI API key",
			config: Config{
				DatabaseURL:         "postgresql://user:pass@localhost/db",
				SimilarityThreshold: 0.8,
				VectorDimensions:    1536,
			},
			wantErr:     true,
			errContains: "OPENAI_API_KEY is required",
		},
		{
			name: "invalid similarity threshold (negative)",
			config: Config{
				DatabaseURL:         "postgresql://user:pass@localhost/db",
				OpenAIAPIKey:        "sk-test-key",
				SimilarityThreshold: -0.1,
				VectorDimensions:    1536,
			},
			wantErr:     true,
			errContains: "SIMILARITY_THRESHOLD must be between 0 and 1",
		},
		{
			name: "invalid similarity threshold (greater than 1)",
			config: Config{
				DatabaseURL:         "postgresql://user:pass@localhost/db",
				OpenAIAPIKey:        "sk-test-key",
				SimilarityThreshold: 1.1,
				VectorDimensions:    1536,
			},
			wantErr:     true,
			errContains: "SIMILARITY_THRESHOLD must be between 0 and 1",
		},
		{
			name: "invalid vector dimensions",
			config: Config{
				DatabaseURL:         "postgresql://user:pass@localhost/db",
				OpenAIAPIKey:        "sk-test-key",
				SimilarityThreshold: 0.8,
				VectorDimensions:    0,
			},
			wantErr:     true,
			errContains: "VECTOR_DIMENSIONS must be positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()

			if tt.wantErr {
				require.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestFindProjectRoot(t *testing.T) {
	// Create a temporary directory structure
	tmpDir := t.TempDir()
	projectDir := filepath.Join(tmpDir, "project")
	subDir := filepath.Join(projectDir, "sub", "dir")

	// Create directories
	err := os.MkdirAll(subDir, 0755)
	require.NoError(t, err)

	// Create go.mod in project directory
	goModPath := filepath.Join(projectDir, "go.mod")
	err = os.WriteFile(goModPath, []byte("module test\n"), 0644)
	require.NoError(t, err)

	// Test from project directory
	oldWd, _ := os.Getwd()
	defer os.Chdir(oldWd)

	err = os.Chdir(projectDir)
	require.NoError(t, err)

	root, err := findProjectRoot()
	assert.NoError(t, err)
	assert.Equal(t, projectDir, root)

	// Test from subdirectory
	err = os.Chdir(subDir)
	require.NoError(t, err)

	root, err = findProjectRoot()
	assert.NoError(t, err)
	assert.Equal(t, projectDir, root)

	// Test from directory without go.mod
	err = os.Chdir(tmpDir)
	require.NoError(t, err)

	_, err = findProjectRoot()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "could not find project root")
}
