package config

import (
   "fmt"
   "os"
   "time"

   "gopkg.in/yaml.v2"
)

// Config holds configuration for server, cache, and OpenAI client.
type Config struct {
   Server struct {
       Address string `yaml:"address"`
   } `yaml:"server"`
   Cache struct {
       Type           string   `yaml:"type"`
       Capacity       int      `yaml:"capacity"`
       EvictionPolicy string   `yaml:"eviction_policy"`
       TTL            string   `yaml:"ttl"`
       MinSimilarity  float64  `yaml:"min_similarity"`
       Redis struct {
           Addrs    []string `yaml:"addrs"`
           Password string   `yaml:"password"`
           DB       int      `yaml:"db"`
       } `yaml:"redis"`
   } `yaml:"cache"`
   OpenAI struct {
       APIKey     string `yaml:"api_key"`
       BaseURL    string `yaml:"base_url"`
       APIVersion string `yaml:"api_version"`
   } `yaml:"openai"`
}

// LoadConfig reads a YAML config file from the given path and unmarshals it.
func LoadConfig(path string) (*Config, error) {
   data, err := os.ReadFile(path)
   if err != nil {
       return nil, err
   }
   var cfg Config
   if err := yaml.Unmarshal(data, &cfg); err != nil {
       return nil, err
   }
   // Validate TTL format if set
   if cfg.Cache.TTL != "" {
       if _, err := time.ParseDuration(cfg.Cache.TTL); err != nil {
           return nil, fmt.Errorf("invalid ttl %q: %w", cfg.Cache.TTL, err)
       }
   }
   return &cfg, nil
}

// TTLDuration parses the TTL string into a time.Duration. Returns 0 if parsing fails or TTL is empty.
func (c *Config) TTLDuration() time.Duration {
   if c.Cache.TTL == "" {
       return 0
   }
   d, err := time.ParseDuration(c.Cache.TTL)
   if err != nil {
       return 0
   }
   return d
}