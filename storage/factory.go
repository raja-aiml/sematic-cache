// Package storage provides backend factory for cache implementations.
package storage

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/raja-aiml/sematic-cache/config"
	"github.com/raja-aiml/sematic-cache/core"
)

// BackendFactory creates a cache backend from configuration
type BackendFactory func(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error)

// registry holds all registered backend factories
var (
	registry = make(map[string]BackendFactory)
	mu       sync.RWMutex
)

// Register adds a new backend factory to the registry
func Register(name string, factory BackendFactory) {
	mu.Lock()
	defer mu.Unlock()
	registry[name] = factory
}

// init registers the built-in backends
func init() {
	// Register in-memory backend
	Register("memory", newMemoryBackend)
	Register("", newMemoryBackend) // default

	// Register Redis backend
	Register("redis", newRedisBackend)

	// Register GORM/PostgreSQL backend
	Register("gorm", newGormBackend)

	// Register composite backend
	Register("composite", newCompositeBackend)
}

// NewBackend creates a cache backend based on the configuration.
// It returns an appropriate implementation of core.CacheBackend.
func NewBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	// Default to memory backend if no config
	backendType := ""
	if cfg != nil {
		backendType = cfg.Cache.Type
	}

	// Look up the factory in the registry
	mu.RLock()
	factory, exists := registry[backendType]
	mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown cache backend type: %q", backendType)
	}

	return factory(cfg, embedFunc)
}

// newMemoryBackend creates an in-memory cache backend
func newMemoryBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	capacity := 100
	if cfg != nil && cfg.Cache.Capacity > 0 {
		capacity = cfg.Cache.Capacity
	}

	opts := []core.Option{
		core.WithEmbeddingFunc(embedFunc),
	}

	if cfg != nil {
		if cfg.Cache.EvictionPolicy != "" {
			opts = append(opts, core.WithEvictionPolicy(cfg.Cache.EvictionPolicy))
		}
		if ttl := cfg.TTLDuration(); ttl > 0 {
			opts = append(opts, core.WithTTL(ttl))
		}
		if cfg.Cache.MinSimilarity != 0 {
			opts = append(opts, core.WithMinSimilarity(cfg.Cache.MinSimilarity))
		}
	}

	return core.NewCache(capacity, opts...), nil
}

// newRedisBackend creates a Redis cluster cache backend
func newRedisBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	if cfg == nil || len(cfg.Cache.Redis.Addrs) == 0 {
		return nil, fmt.Errorf("redis addresses must be configured")
	}

	redisOpts := &redis.ClusterOptions{
		Addrs:    cfg.Cache.Redis.Addrs,
		Password: cfg.Cache.Redis.Password,
	}
	client := redis.NewClusterClient(redisOpts)

	// Test connection
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}

	return NewRedisStore(client, cfg.TTLDuration()), nil
}

// newGormBackend creates a GORM/PostgreSQL cache backend
func newGormBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		return nil, fmt.Errorf("DATABASE_URL must be set for GORM cache backend")
	}

	ttl := time.Duration(0)
	if cfg != nil {
		ttl = cfg.TTLDuration()
	}

	return NewGormStore(dsn, ttl)
}

// newCompositeBackend creates a composite multi-tier cache backend
func newCompositeBackend(cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	if cfg == nil || len(cfg.Cache.Composite.Tiers) == 0 {
		return nil, fmt.Errorf("composite backend requires tier configuration")
	}

	// Create tiers based on configuration
	tiers := make([]*Tier, 0, len(cfg.Cache.Composite.Tiers))

	for _, tierCfg := range cfg.Cache.Composite.Tiers {
		// Create a temporary config for each tier
		tierConfig := &config.Config{}
		tierConfig.Cache.Type = tierCfg.Type
		tierConfig.Cache.Capacity = tierCfg.Capacity
		tierConfig.Cache.EvictionPolicy = tierCfg.EvictionPolicy
		tierConfig.Cache.TTL = cfg.Cache.TTL
		tierConfig.Cache.Redis = cfg.Cache.Redis

		// Override with tier-specific Redis config if provided
		if len(tierCfg.Redis.Addrs) > 0 {
			tierConfig.Cache.Redis.Addrs = tierCfg.Redis.Addrs
			tierConfig.Cache.Redis.Password = tierCfg.Redis.Password
		}

		// Create the backend for this tier
		backend, err := createTierBackend(tierCfg.Type, tierConfig, embedFunc)
		if err != nil {
			return nil, fmt.Errorf("failed to create tier %s: %w", tierCfg.Name, err)
		}

		// Determine capabilities based on type
		capabilities := TierCapabilities{}
		switch tierCfg.Type {
		case "memory":
			capabilities = TierCapabilities{
				SupportsVectorSearch: true,
				AverageLatencyNs:     1000, // 1 microsecond
				IsPersistent:         false,
				IsDistributed:        false,
			}
		case "redis":
			capabilities = TierCapabilities{
				SupportsVectorSearch: false,
				AverageLatencyNs:     100000, // 100 microseconds
				IsPersistent:         true,
				IsDistributed:        true,
			}
		case "gorm":
			capabilities = TierCapabilities{
				SupportsVectorSearch: true,
				AverageLatencyNs:     5000000, // 5 milliseconds
				IsPersistent:         true,
				IsDistributed:        true,
			}
		}

		tier := &Tier{
			Type:         TierType(tierCfg.Type),
			Name:         tierCfg.Name,
			Backend:      backend,
			Capabilities: capabilities,
			Priority:     tierCfg.Priority,
		}

		tiers = append(tiers, tier)
	}

	// Create composite backend
	threshold := cfg.Cache.MinSimilarity
	if threshold == 0 {
		threshold = 0.8 // default
	}

	composite := NewCompositeBackend(tiers, embedFunc, threshold)
	composite.SetPromotionEnabled(cfg.Cache.Composite.PromoteOnHit)

	return composite, nil
}

// createTierBackend creates a backend for a specific tier type
func createTierBackend(tierType string, cfg *config.Config, embedFunc core.EmbeddingFunc) (core.CacheBackend, error) {
	switch tierType {
	case "memory":
		return newMemoryBackend(cfg, embedFunc)
	case "redis":
		return newRedisBackend(cfg, embedFunc)
	case "gorm":
		return newGormBackend(cfg, embedFunc)
	default:
		return nil, fmt.Errorf("unknown tier type: %s", tierType)
	}
}
