# SDK Migration Summary

This document summarizes the migration from CLI command execution to Go SDK usage in the deploy/local tool.

## Overview

We've successfully replaced most external command executions with native Go SDKs, improving performance, error handling, and maintainability.

## Changes Made

### 1. Docker Commands → Docker Go SDK ✅

**File**: `pkg/docker/builder.go`
- Added Docker SDK support with automatic fallback to CLI
- Created `pkg/docker/builder_sdk.go` with full SDK implementation
- Benefits:
  - Better error handling
  - Native Go types for configuration
  - Streaming build output
  - No shell escaping issues

**Dependencies Added**:
```go
github.com/docker/docker v28.2.2+incompatible
github.com/docker/go-connections v0.5.0
```

### 2. kubectl Commands → client-go ✅

**Files**: 
- `pkg/kubernetes/apply.go` 
- `pkg/kubernetes/apply_sdk.go`

**Changes**:
- Replaced `kubectl apply -k` with Kustomize SDK + client-go dynamic client
- Replaced `kubectl delete -k` with SDK-based deletion
- Native Kubernetes resource management

**Dependencies Added**:
```go
k8s.io/client-go v0.30.2
k8s.io/apimachinery v0.30.2
k8s.io/api v0.30.2
sigs.k8s.io/kustomize/api v0.19.0
sigs.k8s.io/kustomize/kyaml v0.19.0
```

### 3. k3d Commands → CLI (Decision) ⚠️

**File**: `pkg/k3d/cluster.go`

**Decision**: Kept CLI-based implementation
**Reasons**:
- k3d v5 SDK adds significant dependencies (150+ transitive)
- Potential version conflicts with existing dependencies
- CLI interface is stable and well-maintained
- SDK documentation is limited

### 4. Kustomize Commands → Kustomize SDK ✅

**File**: `pkg/kubernetes/apply_sdk.go`

**Changes**:
- Replaced `kustomize build` with SDK-based build
- Direct YAML generation without temp files
- Better error handling for invalid manifests

### 5. psql Commands → pgx Go Driver ✅

**Files**: 
- `cmd/composite.go`
- `pkg/database/postgres.go` (new)

**Changes**:
- Created PostgreSQL manager using pgx v5
- Replaced `psql -c` with native Go database operations
- Added connection pooling and retry logic
- Better error handling and logging

**Dependencies Added**:
```go
github.com/jackc/pgx/v5 v5.7.5
```

## Benefits

1. **Performance**: 
   - No process spawning overhead
   - Connection reuse (Docker, PostgreSQL)
   - Parallel operations where supported

2. **Error Handling**:
   - Structured errors instead of parsing stderr
   - Type-safe configurations
   - Better context propagation

3. **Security**:
   - No shell injection risks
   - Proper credential handling
   - No temp file creation for sensitive data

4. **Maintainability**:
   - IDE support and type checking
   - Easier testing with interfaces
   - No platform-specific command variations

## Fallback Strategy

All SDK implementations include CLI fallbacks:
- Docker: Falls back if SDK connection fails
- Kubernetes: Falls back if kubeconfig issues
- PostgreSQL: Falls back to kubectl exec for pod access

## Testing Recommendations

1. **Unit Tests**: Mock SDK interfaces for isolated testing
2. **Integration Tests**: Test both SDK and CLI paths
3. **Performance Tests**: Measure improvement over CLI approach
4. **Error Scenarios**: Test fallback mechanisms

## Future Improvements

1. **Connection Pooling**: Implement for Kubernetes client
2. **Retry Logic**: Add exponential backoff for all SDKs
3. **Metrics**: Add performance metrics for SDK vs CLI
4. **Configuration**: Make SDK/CLI choice configurable