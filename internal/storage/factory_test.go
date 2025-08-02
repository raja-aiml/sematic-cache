package storage

import (
	"testing"

	core "github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFactory_NewBackend(t *testing.T) {
	// Mock embedding function
	embedFunc := func(s string) ([]float32, error) {
		return []float32{0.1, 0.2, 0.3}, nil
	}

	t.Run("DefaultToMemory", func(t *testing.T) {
		// Nil config should default to memory
		backend, err := NewBackend(nil, embedFunc)
		require.NoError(t, err)
		assert.NotNil(t, backend)

		// Should be able to use it
		backend.SetWithModel("test", nil, "answer", "", "")
		answer, found := backend.Get("test")
		assert.True(t, found)
		assert.Equal(t, "answer", answer)
	})

	t.Run("ExplicitMemory", func(t *testing.T) {
		cfg := &config.Config{}
		cfg.Cache.Type = "memory"
		cfg.Cache.Capacity = 100

		backend, err := NewBackend(cfg, embedFunc)
		require.NoError(t, err)
		assert.NotNil(t, backend)
	})

	t.Run("UnknownBackend", func(t *testing.T) {
		cfg := &config.Config{}
		cfg.Cache.Type = "unknown"

		backend, err := NewBackend(cfg, embedFunc)
		assert.Error(t, err)
		assert.Nil(t, backend)
		assert.Contains(t, err.Error(), "unknown cache backend type")
	})

	t.Run("RedisWithoutAddrs", func(t *testing.T) {
		cfg := &config.Config{}
		cfg.Cache.Type = "redis"

		backend, err := NewBackend(cfg, embedFunc)
		assert.Error(t, err)
		assert.Nil(t, backend)
		assert.Contains(t, err.Error(), "redis addresses must be configured")
	})
}

func TestFactory_Register(t *testing.T) {
	// Create a mock factory
	mockFactory := func(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
		cache, err := core.NewCache(10)
		if err != nil {
			return nil, err
		}
		return cache, nil
	}

	// Register it
	Register("test-backend", mockFactory)

	// Use it
	cfg := &config.Config{}
	cfg.Cache.Type = "test-backend"

	backend, err := NewBackend(cfg, nil)
	require.NoError(t, err)
	assert.NotNil(t, backend)
}
