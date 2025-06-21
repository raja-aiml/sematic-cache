package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

// ConfigPaths holds the resolved configuration paths
type ConfigPaths struct {
	ConfigDir string
	EnvFile   string
}

// ResolveConfigPaths resolves the configuration directory and env file paths
func ResolveConfigPaths(cmd *cobra.Command) (*ConfigPaths, error) {
	configDir, _ := cmd.Flags().GetString("config-dir")
	envFile, _ := cmd.Flags().GetString("env-file")

	// If config-dir not specified, try default locations
	if configDir == "" {
		// Try ./config first
		if info, err := os.Stat("./config"); err == nil && info.IsDir() {
			configDir = "./config"
		} else if info, err := os.Stat("../config"); err == nil && info.IsDir() {
			// Try ../config (when running from iaac/infra)
			configDir = "../config"
		} else if info, err := os.Stat("../../iaac/config"); err == nil && info.IsDir() {
			// Try ../../iaac/config (when running from deeper directories)
			configDir = "../../iaac/config"
		} else {
			// Default to ./config even if it doesn't exist
			configDir = "./config"
		}
	}

	// Convert to absolute path
	absConfigDir, err := filepath.Abs(configDir)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve config directory: %w", err)
	}

	// If env-file not specified, use default
	if envFile == "" {
		envFile = filepath.Join(absConfigDir, "blueprint.env")
	} else {
		// Make env-file path absolute if relative
		if !filepath.IsAbs(envFile) {
			envFile = filepath.Join(absConfigDir, envFile)
		}
	}

	return &ConfigPaths{
		ConfigDir: absConfigDir,
		EnvFile:   envFile,
	}, nil
}

// LoadEnvFile loads environment variables from the specified file
func LoadEnvFile(envFile string) error {
	// Check if file exists
	if _, err := os.Stat(envFile); os.IsNotExist(err) {
		// Try with .example extension
		exampleFile := envFile + ".example"
		if _, err := os.Stat(exampleFile); err == nil {
			fmt.Printf("Warning: %s not found, using %s\n", envFile, exampleFile)
			envFile = exampleFile
		} else {
			return fmt.Errorf("environment file not found: %s", envFile)
		}
	}

	// Load the env file
	if err := godotenv.Load(envFile); err != nil {
		return fmt.Errorf("failed to load environment file %s: %w", envFile, err)
	}

	fmt.Printf("Loaded configuration from: %s\n", envFile)
	return nil
}

// LoadConfig loads configuration from the resolved paths
func LoadConfig(cmd *cobra.Command) error {
	paths, err := ResolveConfigPaths(cmd)
	if err != nil {
		return err
	}

	// Load environment file
	if err := LoadEnvFile(paths.EnvFile); err != nil {
		// Don't fail if env file is missing, just warn
		fmt.Printf("Warning: %v\n", err)
	}

	return nil
}

// GetConfigValue gets a configuration value with fallback to environment
func GetConfigValue(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// GetRequiredConfigValue gets a required configuration value
func GetRequiredConfigValue(key string) (string, error) {
	value := os.Getenv(key)
	if value == "" {
		return "", fmt.Errorf("required configuration '%s' not set", key)
	}
	return value, nil
}
