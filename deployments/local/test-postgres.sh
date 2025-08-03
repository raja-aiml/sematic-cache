#!/bin/bash

echo "=== PostgreSQL Connection Tests ==="
echo

echo "1. Test from host machine (localhost:5432):"
PGPASSWORD=cache_pass psql -h localhost -p 5432 -U cache_user -d semantic_cache -c "SELECT 'Host connection successful' as status;" 2>/dev/null || echo "   ❌ Failed to connect from host"

echo
echo "2. Test from within Docker network:"
docker exec local-postgres psql -U cache_user -d semantic_cache -c "SELECT 'Docker connection successful' as status;"

echo
echo "3. Check pgvector extension:"
docker exec local-postgres psql -U cache_user -d semantic_cache -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

echo
echo "4. Test vector operations:"
docker exec local-postgres psql -U cache_user -d semantic_cache -c "SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance;"

echo
echo "5. Database info:"
docker exec local-postgres psql -U cache_user -d semantic_cache -c "SELECT current_database(), current_user, version();"

echo
echo "=== Connection Examples ==="
echo
echo "From host machine:"
echo "  psql -h localhost -p 5432 -U cache_user -d semantic_cache"
echo "  Password: cache_pass"
echo
echo "From application (Docker network):"
echo "  postgresql://cache_user:cache_pass@postgres:5432/semantic_cache?sslmode=disable"
echo
echo "Using environment variable:"
echo "  export DATABASE_URL='postgresql://cache_user:cache_pass@localhost:5432/semantic_cache?sslmode=disable'"
echo "  psql \$DATABASE_URL"