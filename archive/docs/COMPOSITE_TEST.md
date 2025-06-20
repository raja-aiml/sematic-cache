# Composite Backend Testing with Cluster Services

The `composite-test.sh` script provides automated testing of the three-tier composite backend using services running in the k3d cluster.

## Overview

This script is specifically designed to:
- Test the composite backend (Memory → PostgreSQL) against real cluster services
- Set up automatic port forwarding for PostgreSQL and Redis
- Run integration tests with the composite backend
- Provide a development server connected to cluster backends

## Prerequisites

1. **k3d cluster must be running**:
   ```bash
   ./deploy/cluster.sh up
   ```

2. **Required tools**:
   - kubectl
   - Go 1.21+
   - nc (netcat) for port testing

## Usage

### Run All Tests (Default)
```bash
./deploy/composite-test.sh test
```

This will:
1. Check prerequisites
2. Set up port forwarding (PostgreSQL: 5432, Redis: 6379)
3. Initialize database with pgvector extension
4. Run composite backend demo
5. Start server and test API endpoints
6. Display metrics

### Run Demo Only
```bash
# With OpenAI API key
OPENAI_API_KEY="your-key" ./deploy/composite-test.sh demo

# Without API key (exact match only)
./deploy/composite-test.sh demo
```

### Start Development Server
```bash
./deploy/composite-test.sh server
```

Starts the server on port 8090 with composite backend connected to cluster services.

### Port Forwarding Only
```bash
./deploy/composite-test.sh port-forward
```

Sets up port forwarding and keeps it running for manual testing.

## What Gets Tested

### 1. **Infrastructure**
- Cluster connectivity
- PostgreSQL with pgvector extension
- Redis connectivity (Note: single-node Redis only)

### 2. **Composite Backend**
- Memory L1 tier (ultra-fast, <1μs)
- PostgreSQL L2 tier (persistent, vector search)
- Automatic fallback between tiers
- Cache promotion on hits

### 3. **API Endpoints**
- `/health` - Service health check
- `/set` - Store cache entries
- `/get` - Retrieve cached data
- `/metrics` - Cache statistics

## Expected Output

```bash
$ ./deploy/composite-test.sh test

ℹ️  Starting composite backend test with cluster services
✅ Prerequisites check passed
ℹ️  Using namespace: infra
ℹ️  Setting up port forwarding from namespace: infra
✅ PostgreSQL port forward established
✅ Redis port forward established
ℹ️  Database URL: postgres://cache:cache@localhost:5432/cache?sslmode=disable
✅ Database initialized
✅ Configuration created: config/composite-cluster-no-redis.yml
ℹ️  Running composite backend demo...
=== Example 1: Basic Caching ===
Cached: What is machine learning?
...
✅ Demo completed

ℹ️  Starting semantic cache server on port 8090...
✅ Server started on port 8090
✅ Health check passed
✅ Cache operations working
Cache Metrics: {"hitRate":1,"hits":1,"misses":0}
✅ Server tests completed

✅ All tests completed successfully!
```

## Configuration

The script automatically creates a composite configuration at:
`config/composite-cluster-no-redis.yml`

This configuration uses:
- Memory L1 tier (1000 capacity, LRU eviction)
- PostgreSQL L2 tier (unlimited capacity, persistent)

## Troubleshooting

### Port Forward Issues
```bash
# Check if ports are already in use
lsof -i :5432
lsof -i :6379

# Kill existing port forwards
pkill -f "kubectl port-forward"
```

### Database Connection Issues
```bash
# Check PostgreSQL pod
kubectl get pods -n infra -l app=postgres

# Check logs
kubectl logs -n infra -l app=postgres
```

### OpenAI API Key
Without a valid OpenAI API key:
- Embedding generation will fail
- Exact match caching will still work
- Vector similarity search won't function

## Differences from Other Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `cluster.sh` | Manage k3d cluster lifecycle | Create/destroy cluster |
| `dev.sh` | Deploy application to cluster | Build and deploy to k8s |
| `composite-test.sh` | Test composite backend locally | Development and testing |

## Advanced Usage

### Custom Database URL
```bash
DATABASE_URL="postgres://user:pass@host:5432/db" ./deploy/composite-test.sh test
```

### Custom Server Port
```bash
# Edit the script to change SERVER_PORT
SERVER_PORT=9090
```

### Debug Mode
```bash
# Run with bash debug mode
bash -x ./deploy/composite-test.sh test
```

## Integration with CI/CD

The script can be used in CI pipelines:

```yaml
# Example GitHub Actions
- name: Start k3d cluster
  run: ./deploy/cluster.sh up

- name: Test composite backend
  run: ./deploy/composite-test.sh test
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Cleanup
  run: ./deploy/cluster.sh down
```