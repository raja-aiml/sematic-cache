# Scripts Directory

This directory contains automation and operational tools for managing the K3D blueprint.

## Directory Structure

- `lib/` - Shared shell utilities and helper functions
- `cluster/` - Cluster lifecycle management scripts
- `dev.sh` - Main development script for common operations

## Usage

### Main Development Script
```bash
./scripts/dev.sh --help
```

### Cluster Management
```bash
./scripts/cluster/cluster.sh create
./scripts/cluster/cluster.sh destroy
```

## Library Functions

The `lib/` directory contains reusable shell functions:
- `common.sh` - General helper functions
- `k8s.sh` - Kubernetes/kubectl helpers
- `logging.sh` - Logging format helpers
- `validation.sh` - Testing helper functions
- `istio.sh` - Istio-specific helpers
- `monitoring.sh` - Monitoring setup helpers