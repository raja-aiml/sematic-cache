package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg == nil {
		t.Fatal("DefaultConfig() returned nil")
	}

	// Test cluster defaults
	if cfg.Cluster.Name != "local-k8s" {
		t.Errorf("DefaultConfig() Cluster.Name = %v, want %v", cfg.Cluster.Name, "local-k8s")
	}
	if cfg.Cluster.APIPort != "6550" {
		t.Errorf("DefaultConfig() Cluster.APIPort = %v, want %v", cfg.Cluster.APIPort, "6550")
	}
	if cfg.Cluster.Servers != 1 {
		t.Errorf("DefaultConfig() Cluster.Servers = %v, want %v", cfg.Cluster.Servers, 1)
	}
	if cfg.Cluster.Timeout != 5*time.Minute {
		t.Errorf("DefaultConfig() Cluster.Timeout = %v, want %v", cfg.Cluster.Timeout, 5*time.Minute)
	}

	// Test build defaults
	if cfg.Build.ImageName != "semantic-cache:local" {
		t.Errorf("DefaultConfig() Build.ImageName = %v, want %v", cfg.Build.ImageName, "semantic-cache:local")
	}
	if cfg.Build.Dockerfile != "Dockerfile" {
		t.Errorf("DefaultConfig() Build.Dockerfile = %v, want %v", cfg.Build.Dockerfile, "Dockerfile")
	}
	if cfg.Build.Timeout != 10*time.Minute {
		t.Errorf("DefaultConfig() Build.Timeout = %v, want %v", cfg.Build.Timeout, 10*time.Minute)
	}

	// Test deploy defaults
	if cfg.Deploy.Namespace != "app" {
		t.Errorf("DefaultConfig() Deploy.Namespace = %v, want %v", cfg.Deploy.Namespace, "app")
	}
	if cfg.Deploy.HealthCheckPath != "/health" {
		t.Errorf("DefaultConfig() Deploy.HealthCheckPath = %v, want %v", cfg.Deploy.HealthCheckPath, "/health")
	}
	if !cfg.Deploy.WaitForReady {
		t.Error("DefaultConfig() Deploy.WaitForReady should be true")
	}

	// Test security defaults
	if !cfg.Security.EnableRBAC {
		t.Error("DefaultConfig() Security.EnableRBAC should be true")
	}
	if cfg.Security.SecretBackend != "env" {
		t.Errorf("DefaultConfig() Security.SecretBackend = %v, want %v", cfg.Security.SecretBackend, "env")
	}

	// Test test config defaults
	if !cfg.Test.Enabled {
		t.Error("DefaultConfig() Test.Enabled should be true")
	}
	if cfg.Test.ConcurrentTests != 5 {
		t.Errorf("DefaultConfig() Test.ConcurrentTests = %v, want %v", cfg.Test.ConcurrentTests, 5)
	}
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name:    "valid_default_config",
			config:  DefaultConfig(),
			wantErr: false,
		},
		{
			name: "invalid_cluster_name_too_short",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Cluster.Name = "ab"
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_cluster_name_too_long",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Cluster.Name = "this-is-a-very-long-cluster-name-that-exceeds-the-maximum-allowed-length-limit"
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_servers_count",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Cluster.Servers = 0
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_servers_count_too_many",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Cluster.Servers = 11
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_agents_count",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Cluster.Agents = -1
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_health_check_path",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Deploy.HealthCheckPath = "health" // missing leading slash
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_secret_backend",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Security.SecretBackend = "invalid"
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "invalid_encryption_algorithm",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Security.Encryption.Algorithm = "invalid"
				return cfg
			}(),
			wantErr: true,
		},
		{
			name: "valid_alternative_values",
			config: func() *Config {
				cfg := DefaultConfig()
				cfg.Security.SecretBackend = "vault"
				cfg.Security.Encryption.Algorithm = "rsa"
				cfg.Test.ConcurrentTests = 10
				cfg.Deploy.MaxRetries = 5
				return cfg
			}(),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestApplyEnvOverrides(t *testing.T) {
	tests := []struct {
		name     string
		envVars  map[string]string
		validate func(*Config) bool
	}{
		{
			name: "debug_mode_enabled",
			envVars: map[string]string{
				"SC_DEPLOY_DEBUG": "true",
			},
			validate: func(cfg *Config) bool {
				return cfg.Debug == true
			},
		},
		{
			name: "cluster_name_override",
			envVars: map[string]string{
				"SC_DEPLOY_CLUSTER_NAME": "custom-cluster",
			},
			validate: func(cfg *Config) bool {
				return cfg.Cluster.Name == "custom-cluster"
			},
		},
		{
			name: "cluster_timeout_override",
			envVars: map[string]string{
				"SC_DEPLOY_CLUSTER_TIMEOUT": "10m",
			},
			validate: func(cfg *Config) bool {
				return cfg.Cluster.Timeout == 10*time.Minute
			},
		},
		{
			name: "build_image_override",
			envVars: map[string]string{
				"SC_DEPLOY_BUILD_IMAGE": "custom-image:v1",
			},
			validate: func(cfg *Config) bool {
				return cfg.Build.ImageName == "custom-image:v1"
			},
		},
		{
			name: "build_timeout_override",
			envVars: map[string]string{
				"SC_DEPLOY_BUILD_TIMEOUT": "30m",
			},
			validate: func(cfg *Config) bool {
				return cfg.Build.Timeout == 30*time.Minute
			},
		},
		{
			name: "namespace_override",
			envVars: map[string]string{
				"SC_DEPLOY_NAMESPACE": "custom-ns",
			},
			validate: func(cfg *Config) bool {
				return cfg.Deploy.Namespace == "custom-ns"
			},
		},
		{
			name: "test_disabled",
			envVars: map[string]string{
				"SC_DEPLOY_TEST_ENABLED": "false",
			},
			validate: func(cfg *Config) bool {
				return cfg.Test.Enabled == false
			},
		},
		{
			name: "rbac_disabled",
			envVars: map[string]string{
				"SC_DEPLOY_SECURITY_RBAC": "false",
			},
			validate: func(cfg *Config) bool {
				return cfg.Security.EnableRBAC == false
			},
		},
		{
			name: "secret_backend_override",
			envVars: map[string]string{
				"SC_DEPLOY_SECURITY_BACKEND": "vault",
			},
			validate: func(cfg *Config) bool {
				return cfg.Security.SecretBackend == "vault"
			},
		},
		{
			name: "multiple_overrides",
			envVars: map[string]string{
				"SC_DEPLOY_DEBUG":        "true",
				"SC_DEPLOY_CLUSTER_NAME": "multi-test",
				"SC_DEPLOY_TEST_ENABLED": "false",
			},
			validate: func(cfg *Config) bool {
				return cfg.Debug == true &&
					cfg.Cluster.Name == "multi-test" &&
					cfg.Test.Enabled == false
			},
		},
		{
			name: "invalid_duration_ignored",
			envVars: map[string]string{
				"SC_DEPLOY_CLUSTER_TIMEOUT": "invalid",
			},
			validate: func(cfg *Config) bool {
				return cfg.Cluster.Timeout == 5*time.Minute // should remain default
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear env vars first
			for k := range tt.envVars {
				if err := os.Unsetenv(k); err != nil {
					t.Logf("Failed to unset env var %s: %v", k, err)
				}
			}

			// Set test env vars
			for k, v := range tt.envVars {
				if err := os.Setenv(k, v); err != nil {
					t.Fatalf("Failed to set env var %s: %v", k, err)
				}
				defer func(key string) {
					if err := os.Unsetenv(key); err != nil {
						t.Logf("Failed to unset env var %s: %v", key, err)
					}
				}(k)
			}

			cfg := DefaultConfig()
			applyEnvOverrides(cfg)

			if !tt.validate(cfg) {
				t.Errorf("applyEnvOverrides() failed validation for %s", tt.name)
			}
		})
	}
}

func TestSaveConfig(t *testing.T) {
	cfg := DefaultConfig()

	// Create temp file
	tmpFile, err := os.CreateTemp("", "config-test-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	tmpPath := tmpFile.Name()
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}
	defer func() {
		if err := os.Remove(tmpPath); err != nil {
			t.Logf("Failed to remove temp file: %v", err)
		}
	}()

	// Save config
	err = SaveConfig(cfg, tmpPath)
	if err != nil {
		t.Errorf("SaveConfig() error = %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(tmpPath); os.IsNotExist(err) {
		t.Error("SaveConfig() didn't create file")
	}

	// Read and verify content
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		t.Fatal(err)
	}

	var loaded Config
	err = yaml.Unmarshal(data, &loaded)
	if err != nil {
		t.Errorf("SaveConfig() produced invalid YAML: %v", err)
	}

	// Verify some fields
	if loaded.Cluster.Name != cfg.Cluster.Name {
		t.Errorf("Saved config has different cluster name: got %v, want %v",
			loaded.Cluster.Name, cfg.Cluster.Name)
	}
}

func TestSaveConfig_WriteError(t *testing.T) {
	cfg := DefaultConfig()

	// Try to save to invalid path
	err := SaveConfig(cfg, "/invalid/path/config.yaml")
	if err == nil {
		t.Error("SaveConfig() expected error for invalid path")
	}
}

func TestLoadConfig_NonExistentFile(t *testing.T) {
	cfg, err := LoadConfig("/non/existent/file.yaml")
	if err == nil {
		t.Fatal("LoadConfig() expected error for non-existent file")
	}

	if cfg != nil {
		t.Error("LoadConfig() should return nil config on error")
	}
}

func TestLoadConfig_ValidFile(t *testing.T) {
	// Clear any environment variables that might interfere
	if err := os.Unsetenv("SC_DEPLOY_BUILD_IMAGE"); err != nil {
		t.Logf("Failed to unset SC_DEPLOY_BUILD_IMAGE: %v", err)
	}

	// Create a minimal valid config
	// Note: sigs.k8s.io/yaml doesn't support time.Duration parsing from nanoseconds
	// so we skip duration fields in this test
	configData := `
debug: true
cluster:
  name: test-cluster
  api_port: "6550"
  http_port: "8080:80"
  https_port: "8443:443"
  servers: 1
build:
  image_name: test-image:latest
  dockerfile: Dockerfile
  context: .
deploy:
  namespace: test
  health_check_path: /health
test:
  enabled: true
  retry_attempts: 3
  concurrent_tests: 5
security:
  enable_rbac: true
  secret_backend: env
  encryption:
    algorithm: aes256
`

	// Create temp file
	tmpFile, err := os.CreateTemp("", "config-test-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if err := os.Remove(tmpPath); err != nil {
			t.Logf("Failed to remove temp file: %v", err)
		}
	}()

	// Write config
	if _, err := tmpFile.WriteString(configData); err != nil {
		t.Fatal(err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}

	// Load config
	cfg, err := LoadConfig(tmpPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg == nil {
		t.Fatal("LoadConfig() returned nil config")
	}

	// Verify loaded values
	if !cfg.Debug {
		t.Error("LoadConfig() didn't load debug value")
	}

	if cfg.Cluster.Name != "test-cluster" {
		t.Errorf("LoadConfig() Cluster.Name = %v, want %v", cfg.Cluster.Name, "test-cluster")
	}

	// Debug what actually loaded
	t.Logf("Loaded config: Debug=%v, Cluster.Name=%v, Build.ImageName=%v",
		cfg.Debug, cfg.Cluster.Name, cfg.Build.ImageName)

	if cfg.Build.ImageName != "test-image:latest" {
		t.Errorf("LoadConfig() Build.ImageName = %v, want %v", cfg.Build.ImageName, "test-image:latest")
	}
}

func TestLoadConfig_InvalidYAML(t *testing.T) {
	// Create temp file with invalid YAML
	tmpFile, err := os.CreateTemp("", "config-test-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if err := os.Remove(tmpPath); err != nil {
			t.Logf("Failed to remove temp file: %v", err)
		}
	}()

	// Write invalid YAML
	if _, err := tmpFile.WriteString("invalid: yaml: content:"); err != nil {
		t.Fatal(err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Logf("Failed to close temp file: %v", err)
	}

	// Try to load
	cfg, err := LoadConfig(tmpPath)
	if err == nil {
		t.Error("LoadConfig() expected error for invalid YAML")
	}

	if cfg != nil {
		t.Error("LoadConfig() should return nil config on error")
	}
}

func TestLoadConfig_EmptyPath(t *testing.T) {
	// When path is empty, it should try default locations
	// Since they don't exist in test environment, it should return defaults
	cfg, err := LoadConfig("")

	// Should not error - just uses defaults
	if err != nil {
		t.Errorf("LoadConfig(\"\") unexpected error = %v", err)
	}

	if cfg == nil {
		t.Fatal("LoadConfig(\"\") returned nil config")
	}

	// Should have default values
	if cfg.Cluster.Name != "local-k8s" {
		t.Errorf("LoadConfig(\"\") didn't use defaults, got cluster name = %v", cfg.Cluster.Name)
	}
}

func TestValidateConfigFile(t *testing.T) {
	// Create a valid config file using duration strings
	configData := `
cluster:
  name: test
  api_port: "6550"
  http_port: "8080:80"
  https_port: "8443:443"
  servers: 1
  timeout: 5m
  wait_time: 10s
build:
  image_name: test:latest
  dockerfile: Dockerfile
  context: .
  timeout: 10m
deploy:
  namespace: test
  timeout: 5m
  health_check_path: /health
test:
  enabled: true
  timeout: 2m
  retry_attempts: 3
  retry_delay: 5s
  concurrent_tests: 5
security:
  enable_rbac: true
  secret_backend: env
  encryption:
    algorithm: aes256
`

	tmpFile, err := os.CreateTemp("", "config-test-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		if err := os.Remove(tmpPath); err != nil {
			t.Logf("Failed to remove temp file: %v", err)
		}
	}()

	if _, err := tmpFile.WriteString(configData); err != nil {
		t.Fatal(err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Logf("Failed to close temp file: %v", err)
	}

	// Validate file
	err = ValidateConfigFile(tmpPath)
	if err != nil {
		t.Errorf("ValidateConfigFile() error = %v", err)
	}
}

func TestGetConfigPaths(t *testing.T) {
	paths := GetConfigPaths()

	if len(paths) != 3 {
		t.Errorf("GetConfigPaths() returned %d paths, want 3", len(paths))
	}

	// Check expected paths
	expectedPaths := []string{
		"deploy-config.yaml",
		"./config/deploy-config.yaml",
		filepath.Join(os.Getenv("HOME"), ".iaac", "deploy-config.yaml"),
	}

	for i, path := range paths {
		if path != expectedPaths[i] {
			t.Errorf("GetConfigPaths()[%d] = %v, want %v", i, path, expectedPaths[i])
		}
	}
}

func TestPrintConfig(t *testing.T) {
	cfg := DefaultConfig()

	// PrintConfig should not error
	err := PrintConfig(cfg)
	if err != nil {
		t.Errorf("PrintConfig() error = %v", err)
	}
}

// Benchmark tests
func BenchmarkDefaultConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultConfig()
	}
}

func BenchmarkValidateConfig(b *testing.B) {
	cfg := DefaultConfig()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ValidateConfig(cfg)
	}
}

func BenchmarkApplyEnvOverrides(b *testing.B) {
	if err := os.Setenv("SC_DEPLOY_DEBUG", "true"); err != nil {
		b.Fatalf("Failed to set SC_DEPLOY_DEBUG: %v", err)
	}
	if err := os.Setenv("SC_DEPLOY_CLUSTER_NAME", "bench-cluster"); err != nil {
		b.Fatalf("Failed to set SC_DEPLOY_CLUSTER_NAME: %v", err)
	}
	defer func() {
		if err := os.Unsetenv("SC_DEPLOY_DEBUG"); err != nil {
			b.Logf("Failed to unset SC_DEPLOY_DEBUG: %v", err)
		}
		if err := os.Unsetenv("SC_DEPLOY_CLUSTER_NAME"); err != nil {
			b.Logf("Failed to unset SC_DEPLOY_CLUSTER_NAME: %v", err)
		}
	}()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cfg := DefaultConfig()
		applyEnvOverrides(cfg)
	}
}
