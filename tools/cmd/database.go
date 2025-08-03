package cmd

import (
	"context"
	"fmt"

	"github.com/raja-aiml/sematic-cache/internal/storage/pgvector"
	"github.com/raja-aiml/sematic-cache/tools/config"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// NewDatabaseCmd creates the database command group
func NewDatabaseCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "database",
		Short: "Database operations",
		Long:  "Manage PostgreSQL database with pgvector extension",
	}

	cmd.AddCommand(
		NewDatabasePingCmd(),
		NewDatabaseMigrateCmd(),
		NewDatabaseStatusCmd(),
	)

	return cmd
}

// NewDatabasePingCmd creates the database ping command
func NewDatabasePingCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "ping",
		Short: "Test database connection",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()

			// Validate database configuration
			if err := globalCfg.ValidateDatabase(); err != nil {
				return err
			}

			globalLogger.Info("Testing database connection", zap.String("dsn", maskDSN(globalCfg.DatabaseURL)))

			// Connect to database
			db, err := connectDatabase(ctx, globalCfg, globalLogger)
			if err != nil {
				return fmt.Errorf("failed to connect to database: %w", err)
			}

			// Test connection
			sqlDB, err := db.DB()
			if err != nil {
				return fmt.Errorf("failed to get database connection: %w", err)
			}

			if err := sqlDB.PingContext(ctx); err != nil {
				return fmt.Errorf("database ping failed: %w", err)
			}

			globalLogger.Info("Database connection successful")
			fmt.Println("✓ Database connection successful")

			// Check pgvector extension
			var extensionExists bool
			err = db.Raw("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')").Scan(&extensionExists).Error
			if err != nil {
				return fmt.Errorf("failed to check pgvector extension: %w", err)
			}

			if extensionExists {
				fmt.Println("✓ pgvector extension is installed")
			} else {
				fmt.Println("✗ pgvector extension is not installed")
			}

			return nil
		},
	}
}

// NewDatabaseMigrateCmd creates the database migrate command
func NewDatabaseMigrateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "migrate",
		Short: "Run database migrations",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()

			// Validate database configuration
			if err := globalCfg.ValidateDatabase(); err != nil {
				return err
			}

			globalLogger.Info("Running database migrations")

			// Connect to database
			db, err := connectDatabase(ctx, globalCfg, globalLogger)
			if err != nil {
				return fmt.Errorf("failed to connect to database: %w", err)
			}

			// Create pgvector extension if not exists
			if err := db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
				return fmt.Errorf("failed to create pgvector extension: %w", err)
			}

			// Use internal storage to ensure proper table creation
			store, err := pgvector.NewStore(globalCfg.DatabaseURL)
			if err != nil {
				return fmt.Errorf("failed to initialize storage and create tables: %w", err)
			}
			// Store creation handles table creation via AutoMigrate
			_ = store

			globalLogger.Info("Database migrations completed successfully")
			fmt.Println("✓ Database migrations completed successfully")

			return nil
		},
	}
}

// NewDatabaseStatusCmd creates the database status command
func NewDatabaseStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show database status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()

			// Validate database configuration
			if err := globalCfg.ValidateDatabase(); err != nil {
				return err
			}

			globalLogger.Info("Checking database status")

			// Connect to database
			db, err := connectDatabase(ctx, globalCfg, globalLogger)
			if err != nil {
				return fmt.Errorf("failed to connect to database: %w", err)
			}

			// Get database stats
			sqlDB, err := db.DB()
			if err != nil {
				return fmt.Errorf("failed to get database connection: %w", err)
			}

			stats := sqlDB.Stats()

			fmt.Println("Database Status:")
			fmt.Printf("  Open Connections: %d\n", stats.OpenConnections)
			fmt.Printf("  In Use: %d\n", stats.InUse)
			fmt.Printf("  Idle: %d\n", stats.Idle)
			fmt.Printf("  Max Open Connections: %d\n", stats.MaxOpenConnections)

			// Check table count
			var count int64
			err = db.Raw("SELECT COUNT(*) FROM embeddings").Scan(&count).Error
			if err != nil {
				fmt.Println("\n✗ Embeddings table not found (run 'database migrate' first)")
			} else {
				fmt.Printf("\nEmbeddings: %d\n", count)
			}

			// Check pgvector version
			var version string
			err = db.Raw("SELECT extversion FROM pg_extension WHERE extname = 'vector'").Scan(&version).Error
			if err == nil {
				fmt.Printf("pgvector Version: %s\n", version)
			}

			return nil
		},
	}
}

// connectDatabase creates a database connection
func connectDatabase(ctx context.Context, cfg *config.Config, logger *zap.Logger) (*gorm.DB, error) {
	db, err := gorm.Open(postgres.Open(cfg.DatabaseURL), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get database connection: %w", err)
	}

	sqlDB.SetMaxOpenConns(cfg.DatabaseMaxConnections)
	sqlDB.SetMaxIdleConns(cfg.DatabaseMaxIdleConnections)

	return db, nil
}

// maskDSN masks sensitive parts of the DSN for logging
func maskDSN(dsn string) string {
	// Simple masking - in production, use a proper URL parser
	if len(dsn) > 20 {
		return dsn[:10] + "****" + dsn[len(dsn)-10:]
	}
	return "****"
}
