package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

// StorageConfig holds storage-specific configuration
type StorageConfig struct {
	DSN                 string  `yaml:"dsn"`                  // PostgreSQL connection string
	SimilarityThreshold float64 `yaml:"similarity_threshold"` // Minimum similarity for search results
	PoolSize            int     `yaml:"pool_size"`            // Database connection pool size
	IndexLists          int     `yaml:"index_lists"`          // IVFFlat index parameter
}

// Config holds configuration for server and storage
type Config struct {
	Server struct {
		Address string `yaml:"address"`
	} `yaml:"server"`

	Storage StorageConfig `yaml:"storage"`

	OpenAI struct {
		APIKey     string `yaml:"api_key"`
		BaseURL    string `yaml:"base_url"`
		APIVersion string `yaml:"api_version"`
	} `yaml:"openai"`
}

// LoadConfig reads a YAML config file from the given path and unmarshals it
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Set defaults
	if cfg.Storage.SimilarityThreshold == 0 {
		cfg.Storage.SimilarityThreshold = 0.8
	}
	if cfg.Storage.PoolSize == 0 {
		cfg.Storage.PoolSize = 20
	}
	if cfg.Storage.IndexLists == 0 {
		cfg.Storage.IndexLists = 100
	}

	return &cfg, nil
}

// GetDSN returns the database connection string from environment or config
func (c *Config) GetDSN() string {
	// Environment variable takes precedence
	if dsn := os.Getenv("DATABASE_URL"); dsn != "" {
		return dsn
	}
	return c.Storage.DSN
}
