// Package storage provides backend implementations for caching, including Redis.
package storage

import (
   "context"
   "encoding/json"
   "log"
   "time"

   "github.com/go-redis/redis/v8"
   "github.com/raja-aiml/sematic-cache/go/core"
)

// RedisStore implements a CacheBackend using Redis (e.g., AWS ElastiCache).
type RedisStore struct {
   client *redis.ClusterClient
   ttl    time.Duration
}

// NewRedisStore creates a new RedisStore with the given cluster client and TTL.
func NewRedisStore(client *redis.ClusterClient, ttl time.Duration) *RedisStore {
   return &RedisStore{client: client, ttl: ttl}
}

// value stored in Redis for a prompt
type redisValue struct {
   Answer    string `json:"answer"`
   ModelName string `json:"modelName,omitempty"`
   ModelID   string `json:"modelID,omitempty"`
}

// SetWithModel stores the answer and model metadata under the given prompt.
func (r *RedisStore) SetWithModel(prompt string, embedding []float32, answer, modelName, modelID string) {
   val := redisValue{Answer: answer, ModelName: modelName, ModelID: modelID}
   data, err := json.Marshal(val)
   if err != nil {
       log.Printf("redis marshal failed for key %q: %v", prompt, err)
       return
   }
   if err := r.client.Set(context.Background(), prompt, data, r.ttl).Err(); err != nil {
       log.Printf("redis set failed for key %q: %v", prompt, err)
   }
}

// SetPromptWithModel stores a prompt with answer and model metadata (no embedding).
func (r *RedisStore) SetPromptWithModel(prompt, answer, modelName, modelID string) error {
   r.SetWithModel(prompt, nil, answer, modelName, modelID)
   return nil
}

// Get retrieves the answer for a prompt.
func (r *RedisStore) Get(prompt string) (string, bool) {
   data, err := r.client.Get(context.Background(), prompt).Result()
   if err == redis.Nil {
       return "", false
   }
   if err != nil {
       log.Printf("redis get failed for key %q: %v", prompt, err)
       return "", false
   }
   var val redisValue
   if err := json.Unmarshal([]byte(data), &val); err != nil {
       log.Printf("redis unmarshal failed for key %q: %v", prompt, err)
       return "", false
   }
   return val.Answer, true
}

// GetModelInfo returns the stored model metadata for a prompt.
func (r *RedisStore) GetModelInfo(prompt string) (string, string, bool) {
   data, err := r.client.Get(context.Background(), prompt).Result()
   if err == redis.Nil {
       return "", "", false
   }
   if err != nil {
       log.Printf("redis get failed for key %q: %v", prompt, err)
       return "", "", false
   }
   var val redisValue
   if err := json.Unmarshal([]byte(data), &val); err != nil {
       log.Printf("redis unmarshal failed for key %q: %v", prompt, err)
       return "", "", false
   }
   return val.ModelName, val.ModelID, true
}

// GetTopKByEmbedding is not supported for Redis backend.
func (r *RedisStore) GetTopKByEmbedding(embed []float32, k int) []core.QueryResult {
   return nil
}

// Flush removes all entries from Redis.
func (r *RedisStore) Flush() {
   if err := r.client.FlushAll(context.Background()).Err(); err != nil {
       log.Printf("redis flush all failed: %v", err)
   }
}

// Stats returns zero values as Redis metrics are not tracked here.
func (r *RedisStore) Stats() (uint64, uint64, float64) {
   return 0, 0, 0
}