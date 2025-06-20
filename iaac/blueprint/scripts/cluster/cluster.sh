#!/bin/bash

# Blueprint Cluster Management Script
# Usage: ./cluster.sh <create|delete> <cluster_name> [registry_name] [registry_port]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required tools
check_prerequisites() {
    local missing_tools=()
    
    if ! command -v k3d &> /dev/null; then
        missing_tools+=("k3d")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
}

# Create k3d cluster with registry
create_cluster() {
    local cluster_name="${1:-blueprint}"
    local registry_name="${2:-blueprint-registry}"
    local registry_port="${3:-5000}"
    
    log_info "Creating k3d cluster: ${cluster_name}"
    log_info "Registry: ${registry_name}:${registry_port}"
    
    # Check if cluster already exists
    if k3d cluster list | grep -q "${cluster_name}"; then
        log_warning "Cluster ${cluster_name} already exists"
        read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            delete_cluster "${cluster_name}"
        else
            log_info "Using existing cluster"
            return 0
        fi
    fi
    
    # Create registry if it doesn't exist
    if ! k3d registry list | grep -q "${registry_name}"; then
        log_info "Creating registry: ${registry_name}"
        k3d registry create "${registry_name}" --port "${registry_port}"
    else
        log_info "Registry ${registry_name} already exists"
    fi
    
    # Create cluster configuration
    cat > /tmp/k3d-config.yaml <<EOF
apiVersion: k3d.io/v1alpha4
kind: Simple
metadata:
  name: ${cluster_name}
servers: 1
agents: 2
image: rancher/k3s:v1.28.2-k3s1
ports:
  - port: 8080:80
    nodeFilters:
      - loadbalancer
  - port: 8443:443
    nodeFilters:
      - loadbalancer
  - port: 6443:6443
    nodeFilters:
      - server:*
registries:
  use:
    - k3d-${registry_name}:${registry_port}
options:
  k3d:
    wait: true
    timeout: "300s"
  k3s:
    extraArgs:
      - arg: --disable=traefik
        nodeFilters:
          - server:*
      - arg: --disable=servicelb
        nodeFilters:
          - server:*
  kubeconfig:
    updateDefaultKubeconfig: true
    switchCurrentContext: true
EOF
    
    # Create the cluster
    log_info "Creating cluster with configuration..."
    k3d cluster create --config /tmp/k3d-config.yaml
    
    # Wait for cluster to be ready
    log_info "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=300s
    
    # Install metrics server for HPA
    log_info "Installing metrics server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Patch metrics server for k3d
    kubectl patch deployment metrics-server -n kube-system --type='json' \
        -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
    
    # Clean up temporary files
    rm -f /tmp/k3d-config.yaml
    
    log_success "Cluster ${cluster_name} created successfully!"
    log_info "Registry available at: localhost:${registry_port}"
    log_info "Cluster API available at: localhost:6443"
    log_info "HTTP ingress available at: localhost:8080"
    log_info "HTTPS ingress available at: localhost:8443"
}

# Delete k3d cluster
delete_cluster() {
    local cluster_name="${1:-blueprint}"
    
    log_info "Deleting k3d cluster: ${cluster_name}"
    
    if k3d cluster list | grep -q "${cluster_name}"; then
        k3d cluster delete "${cluster_name}"
        log_success "Cluster ${cluster_name} deleted successfully!"
    else
        log_warning "Cluster ${cluster_name} does not exist"
    fi
}

# List clusters
list_clusters() {
    log_info "Available k3d clusters:"
    k3d cluster list
    
    log_info "Available k3d registries:"
    k3d registry list
}

# Show cluster info
cluster_info() {
    local cluster_name="${1:-blueprint}"
    
    log_info "Cluster information for: ${cluster_name}"
    
    if k3d cluster list | grep -q "${cluster_name}"; then
        kubectl cluster-info
        echo
        kubectl get nodes -o wide
        echo
        kubectl get namespaces
    else
        log_error "Cluster ${cluster_name} does not exist"
        exit 1
    fi
}

# Main function
main() {
    check_prerequisites
    
    case "${1:-}" in
        "create")
            create_cluster "${2:-}" "${3:-}" "${4:-}"
            ;;
        "delete")
            delete_cluster "${2:-}"
            ;;
        "list")
            list_clusters
            ;;
        "info")
            cluster_info "${2:-}"
            ;;
        *)
            echo "Usage: $0 <create|delete|list|info> [cluster_name] [registry_name] [registry_port]"
            echo
            echo "Commands:"
            echo "  create [cluster_name] [registry_name] [registry_port] - Create k3d cluster with registry"
            echo "  delete [cluster_name]                               - Delete k3d cluster"
            echo "  list                                                - List all clusters and registries"
            echo "  info [cluster_name]                                 - Show cluster information"
            echo
            echo "Examples:"
            echo "  $0 create k3d-blueprint k3d-registry 5000"
            echo "  $0 delete k3d-blueprint"
            echo "  $0 list"
            echo "  $0 info k3d-blueprint"
            exit 1
            ;;
    esac
}

main "$@"