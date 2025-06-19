package utils

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
)

func LoadEnvFile() error {
	// Look for .env file in multiple locations
	envPaths := []string{
		".env",
		"../../.env",
		filepath.Join(os.Getenv("HOME"), ".semantic-cache.env"),
	}

	for _, path := range envPaths {
		if _, err := os.Stat(path); err == nil {
			return godotenv.Load(path)
		}
	}

	return nil // No .env file found is not an error
}

func GetEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func RequireEnv(key string) (string, error) {
	value := os.Getenv(key)
	if value == "" {
		return "", fmt.Errorf("required environment variable %s not set", key)
	}
	return value, nil
}

func ParseEnvFile(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	env := make(map[string]string)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.Trim(strings.TrimSpace(parts[1]), "\"'")
			env[key] = value
		}
	}

	return env, scanner.Err()
}
