// Package storage provides backend implementations for caching, including GORM/Postgres with pgvector.
package storage

import (
   "context"
   "errors"
   "time"

   "github.com/pgvector/pgvector-go"
   "gorm.io/driver/postgres"
   "gorm.io/gorm"
   "gorm.io/gorm/clause"

   "github.com/raja-aiml/sematic-cache/go/core"
)

// GormStore implements a CacheBackend using GORM and pgvector.
type GormStore struct {
   db  *gorm.DB
   ttl time.Duration // currently unused, for future TTL support
}

// cacheEntry represents the GORM model for cache entries.
type cacheEntry struct {
   Prompt    string          `gorm:"primaryKey;column:prompt"`
   Embedding pgvector.Vector `gorm:"type:vector(1536)"`
   Answer    string          `gorm:"column:answer"`
   ModelName string          `gorm:"column:model_name"`
   ModelID   string          `gorm:"column:model_id"`
   CreatedAt time.Time       `gorm:"autoCreateTime;column:created_at"`
}

// NewGormStore connects to Postgres using the given DSN and returns a GormStore.
// DSN example: "host=... user=... password=... dbname=... sslmode=disable"
func NewGormStore(dsn string, ttl time.Duration) (*GormStore, error) {
   db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
   if err != nil {
       return nil, err
   }
   // Auto-migrate the schema
   if err := db.AutoMigrate(&cacheEntry{}); err != nil {
       return nil, err
   }
   return &GormStore{db: db, ttl: ttl}, nil
}

// SetWithModel upserts an entry with embedding and metadata.
func (g *GormStore) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
   ent := cacheEntry{
       Prompt:    prompt,
       Embedding: pgvector.NewVector(embedding),
       Answer:    answer,
       ModelName: modelName,
       ModelID:   modelID,
   }
   // Upsert using primary key
   g.db.Clauses(clause.OnConflict{UpdateAll: true}).Create(&ent)
}

// SetPromptWithModel is equivalent to SetWithModel (no embedding provided).
func (g *GormStore) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
   g.SetWithModel(prompt, nil, answer, modelName, modelID)
   return nil
}

// Get retrieves the answer for a prompt.
func (g *GormStore) Get(prompt string) (string, bool) {
   var ent cacheEntry
   err := g.db.First(&ent, "prompt = ?", prompt).Error
   if errors.Is(err, gorm.ErrRecordNotFound) {
       return "", false
   }
   if err != nil {
       return "", false
   }
   return ent.Answer, true
}

// GetModelInfo returns the stored model metadata for a prompt.
func (g *GormStore) GetModelInfo(prompt string) (string, string, bool) {
   var ent cacheEntry
   err := g.db.Select("model_name, model_id").First(&ent, "prompt = ?", prompt).Error
   if errors.Is(err, gorm.ErrRecordNotFound) {
       return "", "", false
   }
   if err != nil {
       return "", "", false
   }
   return ent.ModelName, ent.ModelID, true
}

// GetTopKByEmbedding returns up to k most similar entries by embedding.
func (g *GormStore) GetTopKByEmbedding(embed []float32, k int) []core.QueryResult {
   if len(embed) == 0 || k <= 0 {
       return nil
   }
   vec := pgvector.NewVector(embed)
   // We scan into a temporary struct to capture similarity
   type result struct {
       Prompt     string
       Answer     string
       ModelName  string
       ModelID    string
       Similarity float64
   }
   var rows []result
   g.db.Model(&cacheEntry{}).
       Select("prompt, answer, model_name, model_id, embedding <-> ? AS similarity", vec).
       Order("similarity").
       Limit(k).
       Scan(&rows)
   out := make([]core.QueryResult, len(rows))
   for i, r := range rows {
       out[i] = core.QueryResult{
           Prompt:     r.Prompt,
           Answer:     r.Answer,
           ModelName:  r.ModelName,
           ModelID:    r.ModelID,
           Similarity: r.Similarity,
       }
   }
   return out
}

// Flush removes all entries from the table.
func (g *GormStore) Flush() {
   g.db.Session(&gorm.Session{AllowGlobalUpdate: true}).Delete(&cacheEntry{})
}

// Stats returns zero metrics (not tracked in this store).
func (g *GormStore) Stats() (uint64, uint64, float64) {
   return 0, 0, 0
}