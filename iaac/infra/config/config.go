package config

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/go-playground/validator/v10"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"gopkg.in/yaml.v3"
)

// Config represents the application configuration
type Config struct {
	// General settings
	Debug   bool   `yaml:"debug" validate:""`
	LogFile string `yaml:"log_file" validate:"omitempty"`

	// Cluster configuration
	Cluster ClusterConfig `yaml:"cluster" validate:"required"`

	// Build configuration
	Build BuildConfig `yaml:"build" validate:"required"`

	// Deployment configuration
	Deploy DeployConfig `yaml:"deploy" validate:"required"`

	// Testing configuration
	Test TestConfig `yaml:"test" validate:"required"`

	// Security configuration
	Security SecurityConfig `yaml:"security" validate:"required"`
}

// ClusterConfig contains k3d cluster settings
type ClusterConfig struct {
	Name           string         `yaml:"name" validate:"required,min=3,max=63"`
	APIPort        string         `yaml:"api_port" validate:"required"`
	HTTPPort       string         `yaml:"http_port" validate:"required"`
	HTTPSPort      string         `yaml:"https_port" validate:"required"`
	Servers        int            `yaml:"servers" validate:"required,min=1,max=10"`
	Agents         int            `yaml:"agents" validate:"min=0,max=100"`
	Timeout        time.Duration  `yaml:"timeout" validate:"required,min=30s"`
	WaitTime       time.Duration  `yaml:"wait_time" validate:"required,min=5s"`
	AutoRestart    bool           `yaml:"auto_restart" validate:""`
	ResourceLimits ResourceLimits `yaml:"resource_limits" validate:""`
}

// ResourceLimits defines resource constraints
type ResourceLimits struct {
	CPULimit    string `yaml:"cpu_limit" validate:"omitempty"`
	MemoryLimit string `yaml:"memory_limit" validate:"omitempty"`
}

// BuildConfig contains build settings
type BuildConfig struct {
	ImageName  string        `yaml:"image_name" validate:"required"`
	Dockerfile string        `yaml:"dockerfile" validate:"required"`
	Context    string        `yaml:"context" validate:"required"`
	Timeout    time.Duration `yaml:"timeout" validate:"required,min=1m"`
	NoCache    bool          `yaml:"no_cache" validate:""`
	BuildArgs  []string      `yaml:"build_args" validate:""`
	CacheFrom  []string      `yaml:"cache_from" validate:""`
	Platform   string        `yaml:"platform" validate:"omitempty"`
}

// DeployConfig contains deployment settings
type DeployConfig struct {
	Namespace       string            `yaml:"namespace" validate:"required,min=1,max=63"`
	Timeout         time.Duration     `yaml:"timeout" validate:"required,min=30s"`
	WaitForReady    bool              `yaml:"wait_for_ready" validate:""`
	MaxRetries      int               `yaml:"max_retries" validate:"min=0,max=10"`
	HealthCheckPath string            `yaml:"health_check_path" validate:"required,startswith=/"`
	Labels          map[string]string `yaml:"labels" validate:""`
	Annotations     map[string]string `yaml:"annotations" validate:""`
}

// TestConfig contains testing settings
type TestConfig struct {
	Enabled         bool          `yaml:"enabled" validate:""`
	Timeout         time.Duration `yaml:"timeout" validate:"required,min=10s"`
	RetryAttempts   int           `yaml:"retry_attempts" validate:"min=1,max=10"`
	RetryDelay      time.Duration `yaml:"retry_delay" validate:"required,min=1s"`
	ConcurrentTests int           `yaml:"concurrent_tests" validate:"min=1,max=20"`
	SkipCleanup     bool          `yaml:"skip_cleanup" validate:""`
}

// SecurityConfig contains security settings
type SecurityConfig struct {
	EnableRBAC      bool             `yaml:"enable_rbac" validate:""`
	EnablePSP       bool             `yaml:"enable_psp" validate:""`
	SecretRotation  time.Duration    `yaml:"secret_rotation" validate:"omitempty,min=24h"`
	AuditLogging    bool             `yaml:"audit_logging" validate:""`
	NetworkPolicies bool             `yaml:"network_policies" validate:""`
	SecretBackend   string           `yaml:"secret_backend" validate:"omitempty,oneof=env file vault"`
	Encryption      EncryptionConfig `yaml:"encryption" validate:""`
}

// EncryptionConfig contains encryption settings
type EncryptionConfig struct {
	Enabled   bool   `yaml:"enabled" validate:""`
	Algorithm string `yaml:"algorithm" validate:"omitempty,oneof=aes256 rsa"`
	KeyPath   string `yaml:"key_path" validate:"omitempty"`
}

// DefaultConfig returns a configuration with default values
func DefaultConfig() *Config {
	return &Config{
		Debug:   false,
		LogFile: "",
		Cluster: ClusterConfig{
			Name:        "semantic-cache",
			APIPort:     "6550",
			HTTPPort:    "8080:80",
			HTTPSPort:   "8443:443",
			Servers:     1,
			Agents:      0,
			Timeout:     5 * time.Minute,
			WaitTime:    10 * time.Second,
			AutoRestart: false,
			ResourceLimits: ResourceLimits{
				CPULimit:    "",
				MemoryLimit: "",
			},
		},
		Build: BuildConfig{
			ImageName:  "semantic-cache:local",
			Dockerfile: "Dockerfile",
			Context:    ".",
			Timeout:    10 * time.Minute,
			NoCache:    false,
			BuildArgs:  []string{},
			CacheFrom:  []string{},
			Platform:   "",
		},
		Deploy: DeployConfig{
			Namespace:       "app",
			Timeout:         5 * time.Minute,
			WaitForReady:    true,
			MaxRetries:      3,
			HealthCheckPath: "/health",
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
		},
		Test: TestConfig{
			Enabled:         true,
			Timeout:         2 * time.Minute,
			RetryAttempts:   3,
			RetryDelay:      5 * time.Second,
			ConcurrentTests: 5,
			SkipCleanup:     false,
		},
		Security: SecurityConfig{
			EnableRBAC:      true,
			EnablePSP:       false,
			SecretRotation:  0,
			AuditLogging:    true,
			NetworkPolicies: true,
			SecretBackend:   "env",
			Encryption: EncryptionConfig{
				Enabled:   false,
				Algorithm: "aes256",
				KeyPath:   "",
			},
		},
	}
}

// LoadConfig loads configuration from file and environment
func LoadConfig(configPath string) (*Config, error) {
	// Start with defaults
	config := DefaultConfig()

	// Try to load config file if path is provided
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		if err := yaml.Unmarshal(data, config); err != nil {
			return nil, fmt.Errorf("failed to parse config file: %w", err)
		}
	} else {
		// Try default locations
		configPaths := []string{
			"deploy-config.yaml",
			"./config/deploy-config.yaml",
			filepath.Join(os.Getenv("HOME"), ".semantic-cache", "deploy-config.yaml"),
		}

		for _, path := range configPaths {
			if data, err := os.ReadFile(path); err == nil {
				if err := yaml.Unmarshal(data, config); err == nil {
					break
				}
			}
		}
	}

	// Override with environment variables
	applyEnvOverrides(config)

	// Validate
	if err := ValidateConfig(config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// ValidateConfig validates the configuration
func ValidateConfig(config *Config) error {
	validate := validator.New()
	return validate.Struct(config)
}

// applyEnvOverrides applies environment variable overrides
func applyEnvOverrides(config *Config) {
	// Debug mode
	if val := os.Getenv("SC_DEPLOY_DEBUG"); val == "true" {
		config.Debug = true
	}

	// Cluster settings
	if val := os.Getenv("SC_DEPLOY_CLUSTER_NAME"); val != "" {
		config.Cluster.Name = val
	}
	if val := os.Getenv("SC_DEPLOY_CLUSTER_TIMEOUT"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			config.Cluster.Timeout = duration
		}
	}

	// Build settings
	if val := os.Getenv("SC_DEPLOY_BUILD_IMAGE"); val != "" {
		config.Build.ImageName = val
	}
	if val := os.Getenv("SC_DEPLOY_BUILD_TIMEOUT"); val != "" {
		if duration, err := time.ParseDuration(val); err == nil {
			config.Build.Timeout = duration
		}
	}

	// Deploy settings
	if val := os.Getenv("SC_DEPLOY_NAMESPACE"); val != "" {
		config.Deploy.Namespace = val
	}

	// Test settings
	if val := os.Getenv("SC_DEPLOY_TEST_ENABLED"); val == "false" {
		config.Test.Enabled = false
	}

	// Security settings
	if val := os.Getenv("SC_DEPLOY_SECURITY_RBAC"); val == "false" {
		config.Security.EnableRBAC = false
	}
	if val := os.Getenv("SC_DEPLOY_SECURITY_BACKEND"); val != "" {
		config.Security.SecretBackend = val
	}
}

// SaveConfig saves the configuration to a file
func SaveConfig(config *Config, path string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// ValidateConfigFile validates a configuration file without loading it
func ValidateConfigFile(path string) error {
	_, err := LoadConfig(path)
	return err
}

// GetConfigPaths returns the paths where config files are searched
func GetConfigPaths() []string {
	return []string{
		"deploy-config.yaml",
		"./config/deploy-config.yaml",
		filepath.Join(os.Getenv("HOME"), ".semantic-cache", "deploy-config.yaml"),
	}
}

// PrintConfig prints the configuration in YAML format
func PrintConfig(config *Config) error {
	logger := utils.NewLogger("config")

	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	logger.Info("Current configuration:")
	fmt.Println(string(data))
	return nil
}
