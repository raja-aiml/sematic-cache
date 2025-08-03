package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// Config holds all configuration for the CLI tool
type Config struct {
	// Server configuration
	ServerPort    string `mapstructure:"SERVER_PORT"`
	ServerAddress string `mapstructure:"SERVER_ADDRESS"`

	// Database configuration
	DatabaseURL                   string `mapstructure:"DATABASE_URL"`
	DatabaseMaxConnections        int    `mapstructure:"DATABASE_MAX_CONNECTIONS"`
	DatabaseMaxIdleConnections    int    `mapstructure:"DATABASE_MAX_IDLE_CONNECTIONS"`
	DatabaseConnectionMaxLifetime int    `mapstructure:"DATABASE_CONNECTION_MAX_LIFETIME"`

	// Vector search configuration
	SimilarityThreshold float64 `mapstructure:"SIMILARITY_THRESHOLD"`
	VectorIndexLists    int     `mapstructure:"VECTOR_INDEX_LISTS"`
	SearchLimit         int     `mapstructure:"SEARCH_LIMIT"`
	VectorDimensions    int     `mapstructure:"VECTOR_DIMENSIONS"`

	// OpenAI configuration
	OpenAIAPIKey  string `mapstructure:"OPENAI_API_KEY"`
	OpenAIBaseURL string `mapstructure:"OPENAI_BASE_URL"`
	OpenAIModel   string `mapstructure:"OPENAI_MODEL"`

	// Observability
	LogLevel           string `mapstructure:"LOG_LEVEL"`
	LogFormat          string `mapstructure:"LOG_FORMAT"`
	OTELEndpoint       string `mapstructure:"OTEL_EXPORTER_OTLP_ENDPOINT"`
	OTELServiceName    string `mapstructure:"OTEL_SERVICE_NAME"`
	OTELServiceVersion string `mapstructure:"OTEL_SERVICE_VERSION"`

	// Health check
	HealthCheckPath    string `mapstructure:"HEALTH_CHECK_PATH"`
	ReadinessCheckPath string `mapstructure:"READINESS_CHECK_PATH"`

	// Graceful shutdown
	ShutdownTimeout int `mapstructure:"SHUTDOWN_TIMEOUT"`

	// CLI specific
	CLICommandTimeout int `mapstructure:"CLI_COMMAND_TIMEOUT"`
	CLIMaxRetries     int `mapstructure:"CLI_MAX_RETRIES"`
	CLIRetryDelay     int `mapstructure:"CLI_RETRY_DELAY"`
}

// LoadConfig loads configuration from .env.app and .env files using Viper
func LoadConfig() (*Config, error) {
	return LoadConfigFrom("")
}

// LoadConfigFrom loads configuration from a specific directory or file path
// If a directory is provided, it loads .env.app and .env files from that directory
// If a file is provided, it loads that specific file
// Priority order: .env > .env.app > defaults (when loading from directory)
func LoadConfigFrom(configPath string) (*Config, error) {
	v := viper.New()

	// Set config type
	v.SetConfigType("env")

	// Set defaults FIRST
	setDefaults(v)

	var explicitConfigPath bool
	var configDir string

	// If no path specified, look in parent directory
	if configPath == "" {
		dir, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("failed to get working directory: %w", err)
		}

		// Determine the root path based on where we are
		if strings.Contains(dir, "/tools") {
			// We're in the tools directory, go to parent
			configDir = filepath.Dir(dir)
		} else {
			// We're already in the root
			configDir = dir
		}
		explicitConfigPath = false
	} else {
		// Check if provided path is a directory or file
		fileInfo, err := os.Stat(configPath)
		if err == nil && fileInfo.IsDir() {
			// It's a directory, we'll load .env.app and .env from it
			configDir = configPath
		} else if err == nil {
			// It's a file, load just that file
			v.SetConfigFile(configPath)
			if err := v.ReadInConfig(); err != nil {
				return nil, fmt.Errorf("failed to read config from %s: %w", configPath, err)
			}
			// Skip loading .env.app and .env when a specific file is provided
			configDir = ""
		} else if strings.HasSuffix(configPath, ".env") || strings.HasSuffix(configPath, ".env.app") {
			// Specific file requested but doesn't exist
			return nil, fmt.Errorf("config file not found: %s", configPath)
		} else {
			// Assume it's a directory path that might not exist yet
			configDir = configPath
		}
		explicitConfigPath = true
	}

	// Load .env.app and .env files from the directory if configDir is set
	if configDir != "" {
		// First load .env.app (base configuration)
		envAppPath := filepath.Join(configDir, ".env.app")
		if _, err := os.Stat(envAppPath); err == nil {
			v.SetConfigFile(envAppPath)
			if err := v.ReadInConfig(); err != nil {
				return nil, fmt.Errorf("failed to read .env.app from %s: %w", envAppPath, err)
			}
		}

		// Then load .env (overrides .env.app)
		envPath := filepath.Join(configDir, ".env")
		if _, err := os.Stat(envPath); err == nil {
			v.SetConfigFile(envPath)
			// MergeInConfig merges the config file into the existing config
			if err := v.MergeInConfig(); err != nil {
				return nil, fmt.Errorf("failed to read .env from %s: %w", envPath, err)
			}
		}
	}

	// Only bind environment variables if config path was not explicitly provided
	// This allows --config-path to override environment variables
	if !explicitConfigPath {
		// Bind environment variables
		v.SetEnvPrefix("") // No prefix for env vars
		v.AutomaticEnv()
		v.AllowEmptyEnv(true)

		// Explicitly bind DATABASE_URL
		v.BindEnv("DATABASE_URL", "DATABASE_URL")
	}

	// Unmarshal configuration
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Skip validation for now - it will be done when actually needed
	// This allows help commands to work without configuration

	return &cfg, nil
}

// setDefaults sets default values for configuration
func setDefaults(v *viper.Viper) {
	// Do NOT set DATABASE_URL default - let it come from env/file
	// IMPORTANT: Don't set DATABASE_URL default here

	// Server defaults
	v.SetDefault("SERVER_PORT", "8080")
	v.SetDefault("SERVER_ADDRESS", "0.0.0.0")

	// Database defaults
	v.SetDefault("DATABASE_MAX_CONNECTIONS", 25)
	v.SetDefault("DATABASE_MAX_IDLE_CONNECTIONS", 5)
	v.SetDefault("DATABASE_CONNECTION_MAX_LIFETIME", 300)

	// Vector search defaults
	v.SetDefault("SIMILARITY_THRESHOLD", 0.8)
	v.SetDefault("VECTOR_INDEX_LISTS", 100)
	v.SetDefault("SEARCH_LIMIT", 10)
	v.SetDefault("VECTOR_DIMENSIONS", 1536)

	// OpenAI defaults
	v.SetDefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
	v.SetDefault("OPENAI_MODEL", "text-embedding-3-small")

	// Observability defaults
	v.SetDefault("LOG_LEVEL", "info")
	v.SetDefault("LOG_FORMAT", "json")
	v.SetDefault("OTEL_SERVICE_NAME", "semantic-cache-cli")
	v.SetDefault("OTEL_SERVICE_VERSION", "1.0.0")

	// Health check defaults
	v.SetDefault("HEALTH_CHECK_PATH", "/health")
	v.SetDefault("READINESS_CHECK_PATH", "/ready")

	// Graceful shutdown defaults
	v.SetDefault("SHUTDOWN_TIMEOUT", 30)

	// CLI specific defaults
	v.SetDefault("CLI_COMMAND_TIMEOUT", 30)
	v.SetDefault("CLI_MAX_RETRIES", 3)
	v.SetDefault("CLI_RETRY_DELAY", 5)
}

// Validate validates the configuration
func (c *Config) Validate() error {
	// Basic validation - always passes to allow help commands
	return nil
}

// ValidateDatabase validates database configuration
func (c *Config) ValidateDatabase() error {
	if c.DatabaseURL == "" {
		return fmt.Errorf("DATABASE_URL is required. Set it via environment variable or .env file:\n  export DATABASE_URL=postgresql://user:password@localhost:5432/semantic_cache")
	}
	return nil
}

// maskDSN masks the password in a DSN for logging
func maskDSN(dsn string) string {
	if dsn == "" {
		return ""
	}
	// Simple masking - replace password
	if strings.Contains(dsn, "@") {
		parts := strings.Split(dsn, "@")
		if len(parts) >= 2 && strings.Contains(parts[0], ":") {
			userParts := strings.Split(parts[0], "://")
			if len(userParts) >= 2 {
				credParts := strings.Split(userParts[1], ":")
				if len(credParts) >= 2 {
					return fmt.Sprintf("%s://%s:****@%s", userParts[0], credParts[0], parts[1])
				}
			}
		}
	}
	return "****"
}

// ValidateOpenAI validates OpenAI configuration
func (c *Config) ValidateOpenAI() error {
	if c.OpenAIAPIKey == "" {
		return fmt.Errorf("OPENAI_API_KEY is required. Set it via environment variable or .env file:\n  export OPENAI_API_KEY=sk-...")
	}
	return nil
}

// ValidateForSearch validates configuration for search commands
func (c *Config) ValidateForSearch() error {
	var errors []string

	if err := c.ValidateDatabase(); err != nil {
		errors = append(errors, err.Error())
	}

	if err := c.ValidateOpenAI(); err != nil {
		errors = append(errors, err.Error())
	}

	if c.SimilarityThreshold < 0 || c.SimilarityThreshold > 1 {
		errors = append(errors, "SIMILARITY_THRESHOLD must be between 0 and 1")
	}

	if c.VectorDimensions <= 0 {
		errors = append(errors, "VECTOR_DIMENSIONS must be positive")
	}

	if len(errors) > 0 {
		return fmt.Errorf("%s", strings.Join(errors, "\n"))
	}

	return nil
}
