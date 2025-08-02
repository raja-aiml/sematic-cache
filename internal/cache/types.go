package cache

// QueryResult holds a single match from a similarity search
type QueryResult struct {
	Prompt     string
	Answer     string
	Similarity float64
	ModelName  string
	ModelID    string
}

// EmbeddingFunc converts text into an embedding vector
type EmbeddingFunc func(text string) ([]float32, error)

// API Request/Response structures
type GetRequest struct {
	Prompt string `json:"prompt" binding:"required"`
}

type SetRequest struct {
	Prompt    string    `json:"prompt" binding:"required"`
	Answer    string    `json:"answer" binding:"required"`
	ModelName string    `json:"model_name,omitempty"`
	ModelID   string    `json:"model_id,omitempty"`
	Embedding []float32 `json:"embedding,omitempty"`
}

type SimilarRequest struct {
	Prompt    string    `json:"prompt"`
	Embedding []float32 `json:"embedding,omitempty"`
	TopK      int       `json:"top_k"`
	Threshold float64   `json:"threshold,omitempty"`
}

type CacheResponse struct {
	Prompt    string `json:"prompt"`
	Answer    string `json:"answer"`
	ModelName string `json:"model_name,omitempty"`
	ModelID   string `json:"model_id,omitempty"`
	Found     bool   `json:"found"`
}

type SimilarResponse struct {
	Prompt  string        `json:"prompt"`
	Results []QueryResult `json:"results"`
}

type StatsResponse struct {
	Hits    uint64  `json:"hits"`
	Misses  uint64  `json:"misses"`
	HitRate float64 `json:"hit_rate"`
}
