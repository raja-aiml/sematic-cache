package cmd

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/tools/config"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	configPath string
	verbose    bool
	
	// Global config and logger that will be initialized
	globalCfg    *config.Config
	globalLogger *zap.Logger
)

// ExecuteWithArgs is the main entry point for the CLI
func ExecuteWithArgs() error {
	return rootCmd.Execute()
}

// rootCmd represents the base command
var rootCmd = &cobra.Command{
	Use:   "semantic-cache-cli",
	Short: "Semantic Cache CLI tool",
	Long: `A CLI tool for managing semantic cache with PostgreSQL/pgvector backend.
	
This tool provides commands for:
- Cache operations (get, set, clear)
- Similarity search
- Database management
- Health checks
- Telemetry integration`,
	Version: "1.0.0",
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		// Skip config loading for help commands
		if cmd.Name() == "help" || cmd.Name() == "completion" {
			return nil
		}
		
		// Initialize configuration with the provided path
		var err error
		globalCfg, err = config.LoadConfigFrom(configPath)
		if err != nil {
			return fmt.Errorf("failed to load configuration: %w", err)
		}
		
		// Initialize logger
		globalLogger, err = initLogger(globalCfg.LogLevel, globalCfg.LogFormat)
		if err != nil {
			return fmt.Errorf("failed to initialize logger: %w", err)
		}
		
		return nil
	},
}

func init() {
	// Add persistent flags
	rootCmd.PersistentFlags().StringVar(&configPath, "config-path", "", "Path to config directory or file (loads .env.app and .env from directory)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
	
	// Add subcommands
	rootCmd.AddCommand(
		NewCacheCmd(),
		NewDatabaseCmd(),
		NewHealthCmd(),
		NewSearchCmd(),
		NewStatsCmd(),
	)
}

// NewCacheCmd creates the cache command group
func NewCacheCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cache",
		Short: "Cache operations",
		Long:  "Perform cache operations like get, set, and clear",
	}

	cmd.AddCommand(
		NewCacheGetCmd(),
		NewCacheSetCmd(),
		NewCacheClearCmd(),
	)

	return cmd
}

// NewCacheGetCmd creates the cache get command
func NewCacheGetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get [key]",
		Short: "Get a value from cache",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			globalLogger.Info("Getting cache entry", zap.String("key", key))
			
			// Implementation will use the cache client
			fmt.Printf("Getting key: %s\n", key)
			return nil
		},
	}
}

// NewCacheSetCmd creates the cache set command
func NewCacheSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set [key] [value]",
		Short: "Set a value in cache",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			value := args[1]
			globalLogger.Info("Setting cache entry", zap.String("key", key))
			
			// Implementation will use the cache client
			fmt.Printf("Setting key: %s, value: %s\n", key, value)
			return nil
		},
	}
}

// NewCacheClearCmd creates the cache clear command
func NewCacheClearCmd() *cobra.Command {
	var all bool
	
	cmd := &cobra.Command{
		Use:   "clear",
		Short: "Clear cache entries",
		RunE: func(cmd *cobra.Command, args []string) error {
			if all {
				globalLogger.Info("Clearing all cache entries")
				fmt.Println("Clearing all cache entries")
			} else {
				globalLogger.Info("Clearing expired cache entries")
				fmt.Println("Clearing expired cache entries")
			}
			return nil
		},
	}
	
	cmd.Flags().BoolVar(&all, "all", false, "Clear all entries (not just expired)")
	
	return cmd
}

func initLogger(level, format string) (*zap.Logger, error) {
	var config zap.Config
	
	if format == "json" {
		config = zap.NewProductionConfig()
	} else {
		config = zap.NewDevelopmentConfig()
	}

	// Set log level
	switch level {
	case "debug":
		config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	case "info":
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	case "warn":
		config.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
	case "error":
		config.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	default:
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}

	return config.Build()
}