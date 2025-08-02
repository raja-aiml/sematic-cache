package pgvector

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/pgvector/pgvector-go"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Store implements vector storage using PostgreSQL with pgvector
type Store struct {
	db *gorm.DB
}

// Embedding represents a stored vector embedding
type Embedding struct {
	ID          uuid.UUID       `gorm:"type:uuid;default:gen_random_uuid()"`
	Prompt      string          `gorm:"uniqueIndex;not null"`
	Embedding   pgvector.Vector `gorm:"type:vector(1536)"`
	Answer      string          `gorm:"not null"`
	ModelName   string
	ModelID     string
	CreatedAt   time.Time
	UpdatedAt   time.Time
	AccessedAt  time.Time
	AccessCount int `gorm:"default:0"`
}

// TableName specifies the table name for GORM
func (Embedding) TableName() string {
	return "embeddings"
}

// NewStore creates a new pgvector store
func NewStore(dsn string) (*Store, error) {
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Create pgvector extension if not exists
	if err := db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
		return nil, fmt.Errorf("failed to create vector extension: %w", err)
	}

	// Auto-migrate the schema
	if err := db.AutoMigrate(&Embedding{}); err != nil {
		return nil, fmt.Errorf("failed to migrate schema: %w", err)
	}

	// Create indexes for better performance
	indexes := []string{
		"CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
		"CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at)",
		"CREATE INDEX IF NOT EXISTS idx_embeddings_accessed ON embeddings(accessed_at)",
	}

	for _, idx := range indexes {
		if err := db.Exec(idx).Error; err != nil {
			return nil, fmt.Errorf("failed to create index: %w", err)
		}
	}

	return &Store{
		db: db,
	}, nil
}

// Store saves a prompt with its embedding and answer
func (s *Store) Store(ctx context.Context, prompt string, embedding []float32, answer string, modelName string, modelID string) error {
	entry := Embedding{
		Prompt:     prompt,
		Embedding:  pgvector.NewVector(embedding),
		Answer:     answer,
		ModelName:  modelName,
		ModelID:    modelID,
		AccessedAt: time.Now().UTC(),
	}

	err := s.db.WithContext(ctx).
		Clauses(clause.OnConflict{
			Columns: []clause.Column{{Name: "prompt"}},
			DoUpdates: clause.AssignmentColumns([]string{
				"embedding", "answer", "model_name", "model_id", "updated_at", "accessed_at",
			}),
		}).
		Create(&entry).Error

	if err != nil {
		logger.Error("Failed to store embedding", logger.Fields{
			"backend": "pgvector",
			"prompt":  prompt,
			"error":   err.Error(),
		})
		return fmt.Errorf("failed to store embedding: %w", err)
	}

	return nil
}

// Get retrieves an answer by exact prompt match
func (s *Store) Get(ctx context.Context, prompt string) (string, bool) {
	var entry Embedding

	err := s.db.WithContext(ctx).
		Where("prompt = ?", prompt).
		First(&entry).Error

	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return "", false
		}
		logger.Error("Failed to get from cache", logger.Fields{
			"backend": "pgvector",
			"prompt":  prompt,
			"error":   err.Error(),
		})
		return "", false
	}

	// Update access time and count
	s.db.WithContext(ctx).
		Model(&Embedding{}).
		Where("prompt = ?", prompt).
		Updates(map[string]interface{}{
			"accessed_at":  time.Now().UTC(),
			"access_count": gorm.Expr("access_count + ?", 1),
		})

	return entry.Answer, true
}

// Search finds the k most similar entries to the given embedding
func (s *Store) Search(ctx context.Context, embedding []float32, k int, threshold float64) ([]cache.QueryResult, error) {
	if len(embedding) == 0 || k <= 0 {
		return nil, nil
	}

	vec := pgvector.NewVector(embedding)

	var results []struct {
		Prompt     string
		Answer     string
		ModelName  string
		ModelID    string
		Similarity float64
	}

	// Use cosine similarity (1 - cosine distance)
	err := s.db.WithContext(ctx).
		Model(&Embedding{}).
		Select("prompt, answer, model_name, model_id, 1 - (embedding <=> ?) as similarity", vec).
		Where("1 - (embedding <=> ?) > ?", vec, threshold).
		Order("similarity DESC").
		Limit(k).
		Scan(&results).Error

	if err != nil {
		logger.Error("Failed to search embeddings", logger.Fields{
			"backend":   "pgvector",
			"k":         k,
			"threshold": threshold,
			"error":     err.Error(),
		})
		return nil, fmt.Errorf("failed to search embeddings: %w", err)
	}

	queryResults := make([]cache.QueryResult, len(results))
	for i, r := range results {
		queryResults[i] = cache.QueryResult{
			Prompt:     r.Prompt,
			Answer:     r.Answer,
			ModelName:  r.ModelName,
			ModelID:    r.ModelID,
			Similarity: r.Similarity,
		}
	}

	// Update access stats for returned results
	if len(results) > 0 {
		prompts := make([]string, len(results))
		for i, r := range results {
			prompts[i] = r.Prompt
		}

		s.db.WithContext(ctx).
			Model(&Embedding{}).
			Where("prompt IN ?", prompts).
			Updates(map[string]interface{}{
				"accessed_at":  time.Now().UTC(),
				"access_count": gorm.Expr("access_count + ?", 1),
			})
	}

	return queryResults, nil
}

// Delete removes an entry by prompt
func (s *Store) Delete(ctx context.Context, prompt string) error {
	result := s.db.WithContext(ctx).
		Where("prompt = ?", prompt).
		Delete(&Embedding{})

	if result.Error != nil {
		logger.Error("Failed to delete entry", logger.Fields{
			"backend": "pgvector",
			"prompt":  prompt,
			"error":   result.Error.Error(),
		})
		return fmt.Errorf("failed to delete entry: %w", result.Error)
	}

	if result.RowsAffected == 0 {
		return fmt.Errorf("entry not found: %s", prompt)
	}

	return nil
}

// Flush removes all entries
func (s *Store) Flush(ctx context.Context) error {
	err := s.db.WithContext(ctx).
		Session(&gorm.Session{AllowGlobalUpdate: true}).
		Delete(&Embedding{}).Error

	if err != nil {
		logger.Error("Failed to flush all entries", logger.Fields{
			"backend": "pgvector",
			"error":   err.Error(),
		})
		return fmt.Errorf("failed to flush entries: %w", err)
	}

	return nil
}

// Stats returns statistics about the store
func (s *Store) Stats(ctx context.Context) (map[string]interface{}, error) {
	var stats struct {
		TotalEntries   int64
		TotalAccesses  int64
		AvgAccessCount float64
	}

	err := s.db.WithContext(ctx).
		Model(&Embedding{}).
		Select("COUNT(*) as total_entries, SUM(access_count) as total_accesses, AVG(access_count) as avg_access_count").
		Scan(&stats).Error

	if err != nil {
		return nil, fmt.Errorf("failed to get stats: %w", err)
	}

	return map[string]interface{}{
		"total_entries":    stats.TotalEntries,
		"total_accesses":   stats.TotalAccesses,
		"avg_access_count": stats.AvgAccessCount,
	}, nil
}

// Close closes the database connection
func (s *Store) Close() error {
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
