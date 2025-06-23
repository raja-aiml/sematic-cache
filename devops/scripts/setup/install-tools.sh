#!/usr/bin/env bash
# Install required development tools
# Usage: ./install-tools.sh [--skip-confirmation]

set -euo pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# === CONFIGURATION ===
SKIP_CONFIRMATION=${1:-false}

# Tool versions
TASK_VERSION="v3.31.0"
GOLANGCI_LINT_VERSION="v1.55.2"
K3D_VERSION="v5.6.0"
HELM_VERSION="v3.13.2"
KUSTOMIZE_VERSION="v5.3.0"

# === HELPER FUNCTIONS ===
install_task() {
    log_info "Installing Task ${TASK_VERSION}..."
    
    local os=$(get_os)
    local arch=$(get_arch)
    local url="https://github.com/go-task/task/releases/download/${TASK_VERSION}/task_${os}_${arch}.tar.gz"
    
    curl -sL "$url" | tar -xz -C /tmp
    sudo mv /tmp/task /usr/local/bin/
    
    log_success "Task installed successfully"
}

install_golangci_lint() {
    log_info "Installing golangci-lint ${GOLANGCI_LINT_VERSION}..."
    
    curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | \
        sh -s -- -b /usr/local/bin "$GOLANGCI_LINT_VERSION"
    
    log_success "golangci-lint installed successfully"
}

install_k3d() {
    log_info "Installing k3d ${K3D_VERSION}..."
    
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | \
        TAG="$K3D_VERSION" bash
    
    log_success "k3d installed successfully"
}

install_helm() {
    log_info "Installing Helm ${HELM_VERSION}..."
    
    local os=$(get_os)
    local arch=$(get_arch)
    local url="https://get.helm.sh/helm-${HELM_VERSION}-${os}-${arch}.tar.gz"
    
    curl -sL "$url" | tar -xz -C /tmp
    sudo mv /tmp/${os}-${arch}/helm /usr/local/bin/
    
    log_success "Helm installed successfully"
}

install_kustomize() {
    log_info "Installing Kustomize ${KUSTOMIZE_VERSION}..."
    
    local os=$(get_os)
    local arch=$(get_arch)
    local url="https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_${os}_${arch}.tar.gz"
    
    curl -sL "$url" | tar -xz -C /tmp
    sudo mv /tmp/kustomize /usr/local/bin/
    
    log_success "Kustomize installed successfully"
}

# === MAIN ===
main() {
    log_info "Development Tools Installer"
    log_info "=========================="
    
    # Check for required tools
    verify_tools curl tar sudo || die "Missing required tools"
    
    # Detect OS
    local os=$(get_os)
    local arch=$(get_arch)
    log_info "Detected OS: $os/$arch"
    
    # Tools to install
    local tools=(
        "task:Task (build automation)"
        "golangci-lint:golangci-lint (Go linter)"
        "k3d:k3d (local Kubernetes)"
        "helm:Helm (package manager)"
        "kustomize:Kustomize (config management)"
    )
    
    # Check what's already installed
    log_info "Checking installed tools..."
    local missing=()
    
    for tool_desc in "${tools[@]}"; do
        local tool="${tool_desc%%:*}"
        local desc="${tool_desc#*:}"
        
        if command_exists "$tool"; then
            local version=$($tool version 2>/dev/null | head -1 || echo "unknown")
            log_success "$desc is already installed: $version"
        else
            missing+=("$tool_desc")
        fi
    done
    
    # Exit if all tools are installed
    if [ ${#missing[@]} -eq 0 ]; then
        log_success "All tools are already installed!"
        return 0
    fi
    
    # Show tools to install
    log_info "The following tools will be installed:"
    for tool_desc in "${missing[@]}"; do
        local desc="${tool_desc#*:}"
        echo "  - $desc"
    done
    
    # Confirm installation
    if [[ "$SKIP_CONFIRMATION" != "--skip-confirmation" ]]; then
        confirm "Proceed with installation?" || die "Installation cancelled"
    fi
    
    # Install missing tools
    for tool_desc in "${missing[@]}"; do
        local tool="${tool_desc%%:*}"
        case "$tool" in
            task)
                measure_time install_task
                ;;
            golangci-lint)
                measure_time install_golangci_lint
                ;;
            k3d)
                measure_time install_k3d
                ;;
            helm)
                measure_time install_helm
                ;;
            kustomize)
                measure_time install_kustomize
                ;;
        esac
    done
    
    log_success "Installation completed successfully!"
    
    # Verify installations
    log_info "Verifying installations..."
    for tool_desc in "${missing[@]}"; do
        local tool="${tool_desc%%:*}"
        if command_exists "$tool"; then
            local version=$($tool version 2>/dev/null | head -1 || echo "unknown")
            log_success "$tool: $version"
        else
            log_error "$tool installation failed"
        fi
    done
}

# Run main function
main "$@"