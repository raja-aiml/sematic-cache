-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create cache entries table with vector column
CREATE TABLE IF NOT EXISTS cache_entries (
    id SERIAL PRIMARY KEY,
    prompt TEXT UNIQUE NOT NULL,
    embedding vector(1536),  -- OpenAI embedding dimensions
    answer TEXT NOT NULL,
    model_name VARCHAR(255),
    model_id VARCHAR(255),
    similarity_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_cache_entries_prompt ON cache_entries(prompt);
CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at ON cache_entries(created_at);

-- Create vector similarity search index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_cache_entries_embedding_ivfflat 
ON cache_entries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Alternative: HNSW index (better recall, slower build)
-- CREATE INDEX IF NOT EXISTS idx_cache_entries_embedding_hnsw 
-- ON cache_entries USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- Function to search similar embeddings
CREATE OR REPLACE FUNCTION search_similar_embeddings(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.8,
    max_results int DEFAULT 10
)
RETURNS TABLE (
    prompt TEXT,
    answer TEXT,
    similarity float,
    model_name VARCHAR(255)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.prompt,
        ce.answer,
        1 - (ce.embedding <=> query_embedding) as similarity,
        ce.model_name
    FROM cache_entries ce
    WHERE 1 - (ce.embedding <=> query_embedding) > similarity_threshold
    ORDER BY ce.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Update access tracking
CREATE OR REPLACE FUNCTION update_access_tracking()
RETURNS TRIGGER AS $$
BEGIN
    NEW.access_count = OLD.access_count + 1;
    NEW.last_accessed = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for access tracking
DROP TRIGGER IF EXISTS update_cache_access ON cache_entries;
CREATE TRIGGER update_cache_access
    BEFORE UPDATE ON cache_entries
    FOR EACH ROW
    WHEN (OLD.prompt IS DISTINCT FROM NEW.prompt)
    EXECUTE FUNCTION update_access_tracking();