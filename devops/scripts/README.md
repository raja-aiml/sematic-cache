# DevOps Scripts

This directory contains reusable shell scripts for DevOps operations.

## Structure

```
scripts/
├── lib/
│   └── common.sh       # Common functions library
├── setup/
│   └── install-tools.sh # Install development tools
└── README.md
```

## Common Functions Library

The `lib/common.sh` file provides reusable functions for shell scripts.

### Usage

```bash
#!/usr/bin/env bash
source "$(dirname "$0")/lib/common.sh"

# Now you can use all common functions
log_info "Starting deployment..."
verify_tools docker kubectl helm
wait_for_service localhost 8080
```

### Available Functions

#### Logging
- `log_info` - Information messages
- `log_warn` - Warning messages  
- `log_error` - Error messages
- `log_debug` - Debug messages (when DEBUG=true)
- `log_success` - Success messages with ✅
- `log_failure` - Failure messages with ❌

#### Error Handling
- `die` - Log error and exit
- `trap_error` - Automatic error line reporting

#### Utilities
- `command_exists` - Check if command is available
- `verify_tools` - Verify required tools are installed
- `get_os` - Get operating system (linux/darwin/windows)
- `get_arch` - Get architecture (amd64/arm64/386)

#### Docker
- `is_docker_running` - Check if Docker daemon is running
- `wait_for_container` - Wait for container to be healthy

#### Kubernetes
- `kubectl_context_exists` - Check if kubectl context exists
- `wait_for_deployment` - Wait for deployment to be ready
- `wait_for_service` - Wait for TCP service to be available

#### HTTP
- `wait_for_http` - Wait for HTTP endpoint to respond
- `http_retry` - Make HTTP request with retries

#### Interactive
- `confirm` - Ask for user confirmation
- `measure_time` - Measure command execution time

#### Cleanup
- `register_cleanup` - Register cleanup function
- `execute_cleanup` - Run all cleanup functions

## Setup Scripts

### install-tools.sh

Installs required development tools:
- Task (build automation)
- golangci-lint (Go linter)
- k3d (local Kubernetes)
- Helm (package manager)
- Kustomize (config management)

#### Usage

```bash
# Interactive installation
./scripts/setup/install-tools.sh

# Skip confirmation
./scripts/setup/install-tools.sh --skip-confirmation
```

## Best Practices

1. **Always source common.sh** for consistent functions
2. **Use set -euo pipefail** for safer scripts
3. **Add proper error handling** with trap
4. **Log actions** for visibility
5. **Make scripts idempotent** - safe to run multiple times
6. **Add confirmation prompts** for destructive actions
7. **Use cleanup functions** for resource cleanup

## Example Script

```bash
#!/usr/bin/env bash
# Example deployment script

set -euo pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# Configuration
CLUSTER_NAME=${1:-"dev-cluster"}
NAMESPACE="app"

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    kubectl delete namespace "$NAMESPACE" --ignore-not-found || true
}

# Main function
main() {
    log_info "Starting deployment to $CLUSTER_NAME"
    
    # Verify tools
    verify_tools kubectl helm docker || die "Missing required tools"
    
    # Register cleanup
    register_cleanup cleanup
    
    # Check cluster
    if ! kubectl_context_exists "k3d-$CLUSTER_NAME"; then
        die "Cluster $CLUSTER_NAME not found"
    fi
    
    # Create namespace
    log_info "Creating namespace $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy application
    log_info "Deploying application"
    measure_time kubectl apply -k ./k8s -n "$NAMESPACE"
    
    # Wait for deployment
    wait_for_deployment "app" "$NAMESPACE" 300
    
    # Wait for service
    wait_for_http "http://localhost:8080/health" 60
    
    log_success "Deployment completed successfully!"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```