#!/bin/bash
# K3D cluster lifecycle management

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utilities
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/k8s.sh"

# Cluster configuration
CLUSTER_NAME="${CLUSTER_NAME:-k3d-blueprint}"
K3S_VERSION="${K3S_VERSION:-v1.28.5-k3s1}"
AGENTS="${AGENTS:-2}"
SERVERS="${SERVERS:-1}"
API_PORT="${API_PORT:-6443}"
LB_PORT="${LB_PORT:-8080}"
REGISTRY_PORT="${REGISTRY_PORT:-5000}"

# Create K3D cluster
create_cluster() {
    log_info "Creating K3D cluster: $CLUSTER_NAME"
    
    # Check if cluster already exists
    if k3d cluster list | grep -q "$CLUSTER_NAME"; then
        log_warning "Cluster $CLUSTER_NAME already exists"
        return 0
    fi
    
    # Create registry if not exists
    if ! k3d registry list | grep -q "k3d-registry"; then
        log_info "Creating local registry"
        k3d registry create registry --port "$REGISTRY_PORT"
    fi
    
    # Create cluster
    k3d cluster create "$CLUSTER_NAME" \
        --servers "$SERVERS" \
        --agents "$AGENTS" \
        --image "rancher/k3s:$K3S_VERSION" \
        --api-port "$API_PORT" \
        --port "$LB_PORT:80@loadbalancer" \
        --port "443:443@loadbalancer" \
        --registry-use k3d-registry:$REGISTRY_PORT \
        --k3s-arg "--disable=traefik@server:*" \
        --k3s-arg "--disable-network-policy@server:*" \
        --wait
    
    # Update kubeconfig
    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context
    
    log_success "Cluster $CLUSTER_NAME created successfully"
}

# Delete K3D cluster
delete_cluster() {
    log_info "Deleting K3D cluster: $CLUSTER_NAME"
    
    if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
        log_warning "Cluster $CLUSTER_NAME does not exist"
        return 0
    fi
    
    k3d cluster delete "$CLUSTER_NAME"
    log_success "Cluster $CLUSTER_NAME deleted successfully"
}

# Stop K3D cluster
stop_cluster() {
    log_info "Stopping K3D cluster: $CLUSTER_NAME"
    k3d cluster stop "$CLUSTER_NAME"
    log_success "Cluster $CLUSTER_NAME stopped"
}

# Start K3D cluster
start_cluster() {
    log_info "Starting K3D cluster: $CLUSTER_NAME"
    k3d cluster start "$CLUSTER_NAME"
    log_success "Cluster $CLUSTER_NAME started"
}

# Get cluster info
cluster_info() {
    log_info "Cluster information for: $CLUSTER_NAME"
    
    # K3D info
    k3d cluster list | grep "$CLUSTER_NAME" || true
    
    # Kubernetes info
    if kubectl cluster-info >/dev/null 2>&1; then
        echo ""
        kubectl cluster-info
        echo ""
        kubectl get nodes
    else
        log_warning "Unable to connect to cluster"
    fi
}

# Install nginx ingress controller
install_ingress() {
    log_info "Installing NGINX Ingress Controller"
    
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
    
    # Wait for ingress controller
    wait_for_deployment "ingress-nginx-controller" "ingress-nginx" 300
    
    log_success "NGINX Ingress Controller installed"
}

# Main function
main() {
    local action="${1:-}"
    
    case "$action" in
        create)
            create_cluster
            install_ingress
            ;;
        delete|destroy)
            delete_cluster
            ;;
        stop)
            stop_cluster
            ;;
        start)
            start_cluster
            ;;
        info)
            cluster_info
            ;;
        *)
            echo "Usage: $0 {create|delete|destroy|stop|start|info}"
            echo ""
            echo "Environment variables:"
            echo "  CLUSTER_NAME    - Cluster name (default: k3d-blueprint)"
            echo "  K3S_VERSION     - K3s version (default: v1.28.5-k3s1)"
            echo "  AGENTS          - Number of agent nodes (default: 2)"
            echo "  SERVERS         - Number of server nodes (default: 1)"
            echo "  API_PORT        - API server port (default: 6443)"
            echo "  LB_PORT         - Load balancer port (default: 8080)"
            echo "  REGISTRY_PORT   - Registry port (default: 5000)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"