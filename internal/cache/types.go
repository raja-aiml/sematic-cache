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
