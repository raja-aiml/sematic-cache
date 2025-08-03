package cmd

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/internal/embedding"
	"github.com/raja-aiml/sematic-cache/internal/storage/pgvector"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// NewSearchCmd creates the search command
func NewSearchCmd() *cobra.Command {
	var (
		threshold float64
		limit     int
	)

	cmd := &cobra.Command{
		Use:   "search [query]",
		Short: "Search for similar entries",
		Long:  "Search for similar entries in the cache using semantic similarity",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			query := args[0]

			// Validate configuration for search
			if err := globalCfg.ValidateForSearch(); err != nil {
				return err
			}

			globalLogger.Info("Searching for similar entries",
				zap.String("query", query),
				zap.Float64("threshold", threshold),
				zap.Int("limit", limit),
			)

			// Use internal embedding client
			client := embedding.NewClient(globalCfg.OpenAIAPIKey)

			// Generate embedding for the query
			embeddingVector, err := client.Embedding(ctx, query)
			if err != nil {
				return fmt.Errorf("failed to generate embedding: %w", err)
			}

			// Use internal storage/pgvector
			store, err := pgvector.NewStore(globalCfg.DatabaseURL)
			if err != nil {
				return fmt.Errorf("failed to initialize storage: %w", err)
			}

			// Search for similar entries
			results, err := store.Search(ctx, embeddingVector, limit, threshold)
			if err != nil {
				return fmt.Errorf("failed to search: %w", err)
			}

			// Display results
			if len(results) == 0 {
				fmt.Println("No similar entries found")
			} else {
				fmt.Printf("Found %d similar entries:\n\n", len(results))
				for i, result := range results {
					fmt.Printf("%d. Prompt: %s\n", i+1, result.Prompt)
					fmt.Printf("   Similarity: %.4f\n", result.Similarity)
					fmt.Printf("   Answer: %s\n", truncateString(result.Answer, 100))
					if i < len(results)-1 {
						fmt.Println()
					}
				}
			}

			return nil
		},
	}

	cmd.Flags().Float64VarP(&threshold, "threshold", "t", 0.8, "Similarity threshold (0-1)")
	cmd.Flags().IntVarP(&limit, "limit", "l", 10, "Maximum number of results")

	return cmd
}

// truncateString truncates a string to the specified length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
