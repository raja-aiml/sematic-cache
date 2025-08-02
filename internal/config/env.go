package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// EnvConfig loads configuration entirely from environment variables
// Following Twelve-Factor App methodology (Factor III: Config)
type EnvConfig struct {
	// Server configuration
	Port string // PORT or SERVER_PORT

	// Database configuration (Factor IV: Backing services)
	DatabaseURL                   string // DATABASE_URL
	DatabaseMaxConnections        int    // DATABASE_MAX_CONNECTIONS
	DatabaseMaxIdleConnections    int    // DATABASE_MAX_IDLE_CONNECTIONS
	DatabaseConnectionMaxLifetime int    // DATABASE_CONNECTION_MAX_LIFETIME (seconds)

	// Vector search configuration
	SimilarityThreshold float64 // SIMILARITY_THRESHOLD
	VectorIndexLists    int     // VECTOR_INDEX_LISTS
	SearchLimit         int     // SEARCH_LIMIT

	// OpenAI configuration (Factor IV: Backing services)
	OpenAIAPIKey  string // OPENAI_API_KEY
	OpenAIBaseURL string // OPENAI_BASE_URL
	OpenAIModel   string // OPENAI_MODEL

	// Observability (Factor XI: Logs)
	LogLevel  string // LOG_LEVEL
	LogFormat string // LOG_FORMAT (json or text)

	// Telemetry
	OTELEndpoint    string // OTEL_EXPORTER_OTLP_ENDPOINT
	OTELServiceName string // OTEL_SERVICE_NAME

	// Health check
	HealthCheckPath    string // HEALTH_CHECK_PATH
	ReadinessCheckPath string // READINESS_CHECK_PATH

	// Graceful shutdown
	ShutdownTimeout int // SHUTDOWN_TIMEOUT (seconds)
}

// LoadFromEnv loads all configuration from environment variables
func LoadFromEnv() *EnvConfig {
	return &EnvConfig{
		// Server
		Port: getEnvOrDefault("PORT", getEnvOrDefault("SERVER_PORT", "8080")),

		// Database
		DatabaseURL:                   getEnvOrDefault("DATABASE_URL", ""),
		DatabaseMaxConnections:        getEnvAsInt("DATABASE_MAX_CONNECTIONS", 25),
		DatabaseMaxIdleConnections:    getEnvAsInt("DATABASE_MAX_IDLE_CONNECTIONS", 5),
		DatabaseConnectionMaxLifetime: getEnvAsInt("DATABASE_CONNECTION_MAX_LIFETIME", 300),

		// Vector search
		SimilarityThreshold: getEnvAsFloat("SIMILARITY_THRESHOLD", 0.8),
		VectorIndexLists:    getEnvAsInt("VECTOR_INDEX_LISTS", 100),
		SearchLimit:         getEnvAsInt("SEARCH_LIMIT", 10),

		// OpenAI
		OpenAIAPIKey:  getEnvOrDefault("OPENAI_API_KEY", ""),
		OpenAIBaseURL: getEnvOrDefault("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:   getEnvOrDefault("OPENAI_MODEL", "text-embedding-3-small"),

		// Observability
		LogLevel:  getEnvOrDefault("LOG_LEVEL", "info"),
		LogFormat: getEnvOrDefault("LOG_FORMAT", "json"),

		// Telemetry
		OTELEndpoint:    getEnvOrDefault("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
		OTELServiceName: getEnvOrDefault("OTEL_SERVICE_NAME", "semantic-cache"),

		// Health checks
		HealthCheckPath:    getEnvOrDefault("HEALTH_CHECK_PATH", "/health"),
		ReadinessCheckPath: getEnvOrDefault("READINESS_CHECK_PATH", "/ready"),

		// Graceful shutdown
		ShutdownTimeout: getEnvAsInt("SHUTDOWN_TIMEOUT", 30),
	}
}

// Validate checks if required configuration is present
func (c *EnvConfig) Validate() error {
	var errors []string

	if c.DatabaseURL == "" {
		errors = append(errors, "DATABASE_URL is required")
	}

	if c.Port == "" {
		errors = append(errors, "PORT is required")
	}

	if c.SimilarityThreshold < 0 || c.SimilarityThreshold > 1 {
		errors = append(errors, "SIMILARITY_THRESHOLD must be between 0 and 1")
	}

	if len(errors) > 0 {
		return fmt.Errorf("configuration errors: %s", strings.Join(errors, ", "))
	}

	return nil
}

// Helper functions
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	valueStr := os.Getenv(key)
	if value, err := strconv.Atoi(valueStr); err == nil {
		return value
	}
	return defaultValue
}

func getEnvAsFloat(key string, defaultValue float64) float64 {
	valueStr := os.Getenv(key)
	if value, err := strconv.ParseFloat(valueStr, 64); err == nil {
		return value
	}
	return defaultValue
}

func getEnvAsBool(key string, defaultValue bool) bool {
	valueStr := os.Getenv(key)
	if value, err := strconv.ParseBool(valueStr); err == nil {
		return value
	}
	return defaultValue
}
