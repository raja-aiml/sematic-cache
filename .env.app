# Application Configuration (Non-Secrets)
# This file contains default application settings that are NOT sensitive
# Can be committed to version control

# Server Configuration
PORT=8080
SERVER_PORT=8080

# Database Configuration (non-sensitive defaults)
DATABASE_MAX_CONNECTIONS=25
DATABASE_MAX_IDLE_CONNECTIONS=5
DATABASE_CONNECTION_MAX_LIFETIME=300

# Vector Search Configuration
SIMILARITY_THRESHOLD=0.8
VECTOR_INDEX_LISTS=100
SEARCH_LIMIT=10

# OpenAI Configuration (non-sensitive)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=text-embedding-3-small

# Observability
LOG_LEVEL=info
LOG_FORMAT=json

# Telemetry
OTEL_SERVICE_NAME=semantic-cache

# Health Checks
HEALTH_CHECK_PATH=/health
READINESS_CHECK_PATH=/ready

# Graceful Shutdown
SHUTDOWN_TIMEOUT=30

# Gin Mode (production, debug, release)
GIN_MODE=release

# Database Connection (default for local development)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/semantic_cache?sslmode=disable

# Telemetry Endpoint (empty by default, can be overridden)
OTEL_EXPORTER_OTLP_ENDPOINT=