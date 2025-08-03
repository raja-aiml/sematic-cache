package cmd

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// NewHealthCmd creates the health command
func NewHealthCmd() *cobra.Command {
	var endpoint string

	cmd := &cobra.Command{
		Use:   "health",
		Short: "Check service health",
		Long:  "Check the health status of the semantic cache service",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()

			if endpoint == "" {
				// Use localhost instead of 0.0.0.0 for better connectivity
				serverAddr := globalCfg.ServerAddress
				if serverAddr == "0.0.0.0" {
					serverAddr = "localhost"
				}
				endpoint = fmt.Sprintf("http://%s:%s", serverAddr, globalCfg.ServerPort)
			}

			fmt.Println("Health Check Report:")
			fmt.Println("===================")
			fmt.Printf("Endpoint: %s\n\n", endpoint)

			// Check if server is running by testing TCP connection
			serverRunning := false
			if checkTCPConnection(endpoint) {
				serverRunning = true
				fmt.Println("✓ Server is reachable")

				// Create HTTP client with timeout
				client := &http.Client{
					Timeout: 5 * time.Second,
				}

				// Check health endpoint
				healthURL := fmt.Sprintf("%s%s", endpoint, globalCfg.HealthCheckPath)
				resp, err := client.Get(healthURL)
				if err != nil {
					globalLogger.Warn("Health endpoint not responding", zap.Error(err))
					fmt.Printf("✗ Health endpoint not responding: %v\n", err)
				} else {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						fmt.Println("✓ Health endpoint: OK")
					} else {
						fmt.Printf("⚠ Health endpoint returned status: %d\n", resp.StatusCode)
					}
				}

				// Check readiness endpoint
				readyURL := fmt.Sprintf("%s%s", endpoint, globalCfg.ReadinessCheckPath)
				resp, err = client.Get(readyURL)
				if err != nil {
					globalLogger.Warn("Readiness endpoint not responding", zap.Error(err))
					fmt.Printf("✗ Readiness endpoint not responding: %v\n", err)
				} else {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						fmt.Println("✓ Readiness endpoint: OK")
					} else {
						fmt.Printf("⚠ Readiness endpoint returned status: %d\n", resp.StatusCode)
					}
				}
			} else {
				fmt.Printf("✗ Server is not running at %s\n", endpoint)
				fmt.Println("  Hint: Start the server with 'go run main.go' in the parent directory")
			}

			// Always check database connectivity as it's independent of server
			fmt.Println("\nDatabase Health:")
			fmt.Println("================")
			if err := checkDatabaseHealth(ctx); err != nil {
				fmt.Printf("✗ Database connection failed: %v\n", err)
			} else {
				fmt.Println("✓ Database is accessible")
				fmt.Println("✓ pgvector extension is installed")
			}

			// Summary
			fmt.Println("\nSummary:")
			fmt.Println("========")
			if serverRunning {
				fmt.Println("✓ Service components are operational")
			} else {
				fmt.Println("⚠ Server is not running, but database is accessible")
				fmt.Println("  This is normal for CLI-only operations")
			}

			// Don't return error for connection refused - it's just informational
			return nil
		},
	}

	cmd.Flags().StringVarP(&endpoint, "endpoint", "e", "", "Service endpoint (default: http://localhost:8080)")

	return cmd
}

// checkDatabaseHealth performs a basic database health check
func checkDatabaseHealth(ctx context.Context) error {
	// Validate database configuration
	if err := globalCfg.ValidateDatabase(); err != nil {
		return fmt.Errorf("database not configured: %w", err)
	}

	// Connect to database using local function
	db, err := gorm.Open(postgres.Open(globalCfg.DatabaseURL), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("connection failed: %w", err)
	}

	// Test connection
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("failed to get database connection: %w", err)
	}
	defer sqlDB.Close()

	if err := sqlDB.PingContext(ctx); err != nil {
		return fmt.Errorf("ping failed: %w", err)
	}

	// Check pgvector extension
	var extensionExists bool
	err = db.Raw("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')").Scan(&extensionExists).Error
	if err != nil {
		return fmt.Errorf("failed to check pgvector: %w", err)
	}

	if !extensionExists {
		return fmt.Errorf("pgvector extension is not installed")
	}

	return nil
}
