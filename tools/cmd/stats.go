package cmd

import (
	"context"
	"fmt"

	"github.com/raja-aiml/sematic-cache/tools/client"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// NewStatsCmd creates the stats command
func NewStatsCmd() *cobra.Command {
	var useAPI bool
	
	cmd := &cobra.Command{
		Use:   "stats",
		Short: "Display cache statistics",
		Long:  "Display statistics about the cache including entry count, hit rate, and storage usage",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			
			globalLogger.Info("Fetching cache statistics")
			
			// Try to get stats from API first if server is running or explicitly requested
			if useAPI || shouldUseAPI() {
				if err := getStatsFromAPI(); err == nil {
					return nil
				} else {
					globalLogger.Debug("Failed to get stats from API, falling back to database", zap.Error(err))
					if useAPI {
						// If explicitly requested API mode, don't fall back
						return fmt.Errorf("failed to get stats from API: %w", err)
					}
				}
			}
			
			// Fall back to direct database access
			return getStatsFromDatabase(ctx)
		},
	}
	
	cmd.Flags().BoolVar(&useAPI, "api", false, "Force using API instead of direct database access")
	
	return cmd
}

// shouldUseAPI checks if the cache server is running
func shouldUseAPI() bool {
	// Check if server is likely running
	serverAddr := globalCfg.ServerAddress
	if serverAddr == "0.0.0.0" {
		serverAddr = "localhost"
	}
	endpoint := fmt.Sprintf("http://%s:%s", serverAddr, globalCfg.ServerPort)
	
	// Quick check if server is reachable
	return checkTCPConnection(endpoint)
}

// getStatsFromAPI retrieves statistics from the cache server API
func getStatsFromAPI() error {
	serverAddr := globalCfg.ServerAddress
	if serverAddr == "0.0.0.0" {
		serverAddr = "localhost"
	}
	baseURL := fmt.Sprintf("http://%s:%s", serverAddr, globalCfg.ServerPort)
	
	// Create cache client
	cacheClient := client.NewCacheClient(baseURL)
	
	// Get stats from API
	stats, err := cacheClient.GetStats()
	if err != nil {
		return fmt.Errorf("failed to get stats from API: %w", err)
	}
	
	// Display API statistics
	fmt.Println("Cache Statistics (from API):")
	fmt.Println("============================")
	fmt.Printf("Cache Hits:        %d\n", stats.Hits)
	fmt.Printf("Cache Misses:      %d\n", stats.Misses)
	fmt.Printf("Hit Rate:          %.2f%%\n", stats.HitRate*100)
	
	// Try to get additional stats from database if available
	if db, err := connectDatabase(context.Background(), globalCfg, globalLogger); err == nil {
		displayDatabaseStats(db)
	}
	
	return nil
}

// getStatsFromDatabase retrieves statistics directly from database
func getStatsFromDatabase(ctx context.Context) error {
	// Connect to database
	db, err := connectDatabase(ctx, globalCfg, globalLogger)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}
	
	// Get basic statistics from database
	var stats struct {
		TotalEntries int64
		TotalSize    int64
	}
	
	// Count total entries
	err = db.Raw("SELECT COUNT(*) as total_entries FROM embeddings").Scan(&stats.TotalEntries).Error
	if err != nil {
		// Table might not exist yet
		fmt.Println("Cache Statistics:")
		fmt.Println("=================")
		fmt.Println("No cache table found (run 'database migrate' first)")
		return nil
	}
	
	// Display basic statistics
	fmt.Println("Cache Statistics (from Database):")
	fmt.Println("=================================")
	fmt.Printf("Total Entries:     %d\n", stats.TotalEntries)
	
	displayDatabaseStats(db)
	
	return nil
}

// displayDatabaseStats shows detailed database statistics
func displayDatabaseStats(db *gorm.DB) {
	// Get additional database statistics
	var dbStats struct {
		TableSize   string
		IndexSize   string
		TotalSize   string
		RowEstimate int64
	}
	
	// Get table size
	err := db.Raw(`
		SELECT 
			pg_size_pretty(pg_total_relation_size('embeddings')) as total_size,
			pg_size_pretty(pg_table_size('embeddings')) as table_size,
			pg_size_pretty(pg_indexes_size('embeddings')) as index_size,
			reltuples::BIGINT as row_estimate
		FROM pg_class 
		WHERE relname = 'embeddings'
	`).Scan(&dbStats).Error
	
	if err == nil {
		fmt.Println("\nDatabase Statistics:")
		fmt.Println("====================")
		fmt.Printf("Table Size:        %s\n", dbStats.TableSize)
		fmt.Printf("Index Size:        %s\n", dbStats.IndexSize)
		fmt.Printf("Total Size:        %s\n", dbStats.TotalSize)
		fmt.Printf("Row Estimate:      %d\n", dbStats.RowEstimate)
	}
	
	// Get top accessed entries
	var topEntries []struct {
		Prompt      string
		AccessCount int
	}
	
	err = db.Raw(`
		SELECT prompt, access_count 
		FROM embeddings 
		ORDER BY access_count DESC 
		LIMIT 5
	`).Scan(&topEntries).Error
	
	if err == nil && len(topEntries) > 0 {
		fmt.Println("\nTop Accessed Entries:")
		fmt.Println("=====================")
		for i, entry := range topEntries {
			fmt.Printf("%d. %s (accessed %d times)\n", i+1, entry.Prompt, entry.AccessCount)
		}
	}
}

// formatBytes formats bytes into human-readable format
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}