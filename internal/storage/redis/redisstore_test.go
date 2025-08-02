package redis

import (
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-redis/redis/v8"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRedisStore_BasicOperations(t *testing.T) {
	// Create mini redis server for testing
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	// Create redis client
	client := redis.NewClusterClient(&redis.ClusterOptions{
		Addrs: []string{mr.Addr()},
	})

	// Create store
	store := NewRedisStore(client, 0)

	t.Run("SetAndGet", func(t *testing.T) {
		// Set a value
		store.SetWithModel("test-prompt", nil, "test-answer", "gpt-3.5", "v1")

		// Get the value
		answer, found := store.Get("test-prompt")
		assert.True(t, found)
		assert.Equal(t, "test-answer", answer)
	})

	t.Run("GetNonExistent", func(t *testing.T) {
		answer, found := store.Get("non-existent")
		assert.False(t, found)
		assert.Empty(t, answer)
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		// Set with model info
		store.SetWithModel("model-prompt", nil, "answer", "gpt-4", "v2")

		// Get model info
		modelName, modelID, found := store.GetModelInfo("model-prompt")
		assert.True(t, found)
		assert.Equal(t, "gpt-4", modelName)
		assert.Equal(t, "v2", modelID)
	})

	t.Run("Flush", func(t *testing.T) {
		// Add some data
		store.SetWithModel("flush-test", nil, "answer", "", "")

		// Flush
		store.Flush()

		// Verify it's gone
		_, found := store.Get("flush-test")
		assert.False(t, found)
	})

	t.Run("GetTopKByEmbedding", func(t *testing.T) {
		// Should return nil as Redis doesn't support vector search
		results := store.GetTopKByEmbedding([]float32{0.1, 0.2}, 5)
		assert.Nil(t, results)
	})

	t.Run("Stats", func(t *testing.T) {
		hits, misses, rate := store.Stats()
		assert.Equal(t, uint64(0), hits)
		assert.Equal(t, uint64(0), misses)
		assert.Equal(t, float64(0), rate)
	})
}
