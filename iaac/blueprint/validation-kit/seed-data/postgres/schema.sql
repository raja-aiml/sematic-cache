-- PostgreSQL Schema for Testing
-- K3D Blueprint Validation Kit

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create test schema
CREATE SCHEMA IF NOT EXISTS test_schema;

-- Sample table for basic operations
CREATE TABLE test_schema.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample table with JSON data
CREATE TABLE test_schema.events (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES test_schema.users(id),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample table for vector operations
CREATE TABLE test_schema.embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance testing table
CREATE TABLE test_schema.performance_test (
    id SERIAL PRIMARY KEY,
    data TEXT,
    numeric_value NUMERIC(10, 2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[]
);

-- Indexes
CREATE INDEX idx_users_email ON test_schema.users(email);
CREATE INDEX idx_events_user_id ON test_schema.events(user_id);
CREATE INDEX idx_events_timestamp ON test_schema.events(timestamp);
CREATE INDEX idx_embeddings_vector ON test_schema.embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_performance_timestamp ON test_schema.performance_test(timestamp);
CREATE INDEX idx_performance_tags ON test_schema.performance_test USING GIN(tags);

-- Functions
CREATE OR REPLACE FUNCTION test_schema.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON test_schema.users
    FOR EACH ROW
    EXECUTE FUNCTION test_schema.update_updated_at();

-- Views
CREATE VIEW test_schema.user_activity AS
SELECT 
    u.username,
    u.email,
    COUNT(e.id) as event_count,
    MAX(e.timestamp) as last_activity
FROM test_schema.users u
LEFT JOIN test_schema.events e ON u.id = e.user_id
GROUP BY u.username, u.email;