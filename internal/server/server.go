package server

import (
	"context"

	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/raja-aiml/sematic-cache/internal/embedding"
	"github.com/raja-aiml/sematic-cache/internal/logger"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// CreateStorage creates and configures the storage backend
func CreateStorage(cfg *config.EnvConfig) (*storage.VectorStore, error) {
	var embedFunc func(string) ([]float32, error)

	if cfg.OpenAIAPIKey != "" {
		client := embedding.NewClient(cfg.OpenAIAPIKey)
		if cfg.OpenAIBaseURL != "" {
			client.SetBaseURL(cfg.OpenAIBaseURL)
		}
		embedFunc = func(text string) ([]float32, error) {
			return client.Embedding(context.Background(), text)
		}
		logger.Info("OpenAI embeddings enabled", logger.Fields{"model": cfg.OpenAIModel})
	} else {
		logger.Warn("OpenAI API key not configured, embeddings disabled")
	}

	storeCfg := &config.Config{
		Storage: config.StorageConfig{
			DSN:                 cfg.DatabaseURL,
			SimilarityThreshold: cfg.SimilarityThreshold,
			PoolSize:            cfg.DatabaseMaxConnections,
			IndexLists:          cfg.VectorIndexLists,
		},
	}

	return storage.NewVectorStore(storeCfg, embedFunc)
}

// LogServerConfig logs the server configuration
func LogServerConfig(cfg *config.EnvConfig) {
	logger.Info("Starting semantic-cache server", logger.Fields{
		"port":                cfg.Port,
		"database_configured": cfg.DatabaseURL != "",
		"openai_configured":   cfg.OpenAIAPIKey != "",
		"otel_configured":     cfg.OTELEndpoint != "",
	})
}
