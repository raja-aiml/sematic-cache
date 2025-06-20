-- Sample Data for PostgreSQL Testing
-- K3D Blueprint Validation Kit

-- Insert test users
INSERT INTO test_schema.users (username, email) VALUES
    ('john_doe', 'john@example.com'),
    ('jane_smith', 'jane@example.com'),
    ('bob_wilson', 'bob@example.com'),
    ('alice_johnson', 'alice@example.com'),
    ('charlie_brown', 'charlie@example.com');

-- Insert test events
INSERT INTO test_schema.events (user_id, event_type, event_data)
SELECT 
    u.id,
    event_types.type,
    jsonb_build_object(
        'action', event_types.type,
        'details', 'Test event for ' || u.username,
        'metadata', jsonb_build_object(
            'ip', '192.168.1.' || (random() * 254 + 1)::int,
            'user_agent', 'Mozilla/5.0 Test Browser'
        )
    )
FROM test_schema.users u
CROSS JOIN (
    VALUES ('login'), ('logout'), ('view_page'), ('click_button'), ('submit_form')
) AS event_types(type)
WHERE random() < 0.7;  -- 70% chance of each event

-- Insert sample embeddings (simulated)
INSERT INTO test_schema.embeddings (content, embedding, metadata)
VALUES
    ('The quick brown fox jumps over the lazy dog', 
     '[' || array_to_string(ARRAY(SELECT random() FROM generate_series(1, 384)), ',') || ']'::vector,
     '{"category": "text", "language": "en"}'),
    ('PostgreSQL is a powerful open source database', 
     '[' || array_to_string(ARRAY(SELECT random() FROM generate_series(1, 384)), ',') || ']'::vector,
     '{"category": "database", "language": "en"}'),
    ('Machine learning embeddings for semantic search', 
     '[' || array_to_string(ARRAY(SELECT random() FROM generate_series(1, 384)), ',') || ']'::vector,
     '{"category": "ml", "language": "en"}');

-- Insert performance test data
INSERT INTO test_schema.performance_test (data, numeric_value, tags)
SELECT 
    'Test data ' || i,
    random() * 1000,
    ARRAY['test', 'performance', 'batch' || (i / 100)]::TEXT[]
FROM generate_series(1, 10000) i;

-- Create materialized view for testing
CREATE MATERIALIZED VIEW test_schema.user_statistics AS
SELECT 
    u.username,
    COUNT(DISTINCT e.event_type) as unique_events,
    COUNT(e.id) as total_events,
    AVG(CASE WHEN e.event_type = 'login' THEN 1 ELSE 0 END)::NUMERIC(5,2) as login_rate
FROM test_schema.users u
LEFT JOIN test_schema.events e ON u.id = e.user_id
GROUP BY u.username;

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW test_schema.user_statistics;