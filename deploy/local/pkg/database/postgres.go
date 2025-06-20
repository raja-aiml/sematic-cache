package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// PostgresManager handles PostgreSQL database operations
type PostgresManager struct {
	logger *utils.Logger
	config *Config
}

// Config holds PostgreSQL connection configuration
type Config struct {
	Host     string
	Port     int
	User     string
	Password string
	Database string
}

// NewPostgresManager creates a new PostgreSQL manager
func NewPostgresManager(config *Config) *PostgresManager {
	return &PostgresManager{
		logger: utils.NewLogger("postgres"),
		config: config,
	}
}

// InitializeDatabase creates the database and extensions
func (pm *PostgresManager) InitializeDatabase(ctx context.Context) error {
	pm.logger.Info("Initializing PostgreSQL database...")

	// Connect to the default postgres database first
	connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/postgres?sslmode=disable",
		pm.config.User, pm.config.Password, pm.config.Host, pm.config.Port)

	conn, err := pgx.Connect(ctx, connStr)
	if err != nil {
		return fmt.Errorf("failed to connect to PostgreSQL: %w", err)
	}
	defer conn.Close(ctx)

	// Create database if it doesn't exist
	pm.logger.Info("Creating database if not exists: %s", pm.config.Database)

	// Check if database exists
	var exists bool
	err = conn.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)", pm.config.Database).Scan(&exists)
	if err != nil {
		return fmt.Errorf("failed to check database existence: %w", err)
	}

	if !exists {
		_, err = conn.Exec(ctx, fmt.Sprintf("CREATE DATABASE %s", pm.config.Database))
		if err != nil {
			return fmt.Errorf("failed to create database: %w", err)
		}
		pm.logger.Info("Database created: %s", pm.config.Database)
	} else {
		pm.logger.Info("Database already exists: %s", pm.config.Database)
	}

	// Close connection to postgres database
	conn.Close(ctx)

	// Connect to the newly created database
	connStr = fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
		pm.config.User, pm.config.Password, pm.config.Host, pm.config.Port, pm.config.Database)

	conn, err = pgx.Connect(ctx, connStr)
	if err != nil {
		return fmt.Errorf("failed to connect to database %s: %w", pm.config.Database, err)
	}
	defer conn.Close(ctx)

	// Create pgvector extension
	pm.logger.Info("Creating pgvector extension...")
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		return fmt.Errorf("failed to create vector extension: %w", err)
	}

	pm.logger.Info("PostgreSQL initialization completed successfully")
	return nil
}

// WaitForReady waits for PostgreSQL to be ready
func (pm *PostgresManager) WaitForReady(ctx context.Context, timeout time.Duration) error {
	pm.logger.Info("Waiting for PostgreSQL to be ready...")

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/postgres?sslmode=disable",
		pm.config.User, pm.config.Password, pm.config.Host, pm.config.Port)

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for PostgreSQL")
		case <-ticker.C:
			conn, err := pgx.Connect(ctx, connStr)
			if err == nil {
				// Try a simple query
				var result int
				err = conn.QueryRow(ctx, "SELECT 1").Scan(&result)
				conn.Close(ctx)

				if err == nil && result == 1 {
					pm.logger.Info("PostgreSQL is ready")
					return nil
				}
			}
			pm.logger.Debug("PostgreSQL not ready yet, retrying...")
		}
	}
}

// TestConnection tests the database connection
func (pm *PostgresManager) TestConnection(ctx context.Context) error {
	connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
		pm.config.User, pm.config.Password, pm.config.Host, pm.config.Port, pm.config.Database)

	conn, err := pgx.Connect(ctx, connStr)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close(ctx)

	// Test query
	var version string
	err = conn.QueryRow(ctx, "SELECT version()").Scan(&version)
	if err != nil {
		return fmt.Errorf("failed to query version: %w", err)
	}

	pm.logger.Info("PostgreSQL version: %s", version)

	// Check if vector extension is available
	var vectorVersion string
	err = conn.QueryRow(ctx, "SELECT extversion FROM pg_extension WHERE extname = 'vector'").Scan(&vectorVersion)
	if err != nil {
		pm.logger.Warn("pgvector extension not found")
	} else {
		pm.logger.Info("pgvector version: %s", vectorVersion)
	}

	return nil
}

// ExecuteSQL executes arbitrary SQL commands
func (pm *PostgresManager) ExecuteSQL(ctx context.Context, sql string) error {
	connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
		pm.config.User, pm.config.Password, pm.config.Host, pm.config.Port, pm.config.Database)

	conn, err := pgx.Connect(ctx, connStr)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close(ctx)

	_, err = conn.Exec(ctx, sql)
	if err != nil {
		return fmt.Errorf("failed to execute SQL: %w", err)
	}

	return nil
}
