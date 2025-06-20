-- Test queries for PostgreSQL validation

-- Basic connectivity test
SELECT version();

-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Test semantic_cache schema
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

-- Count records in cache_entries
SELECT COUNT(*) as total_entries FROM cache_entries;

-- Sample query with vector similarity
-- This assumes you have some data with embeddings
-- SELECT 
--     query,
--     response,
--     1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) as similarity
-- FROM cache_entries
-- ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'::vector
-- LIMIT 5;

-- Performance test query
EXPLAIN ANALYZE
SELECT query, response
FROM cache_entries
WHERE created_at > NOW() - INTERVAL '1 hour'
LIMIT 100;

-- Check indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;