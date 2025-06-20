#!/bin/bash
set -euo pipefail

# Update Dependencies Script
# Updates all dependencies in the K3D Blueprint project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔄 Updating K3D Blueprint Dependencies"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check and update tool
check_update_tool() {
    local tool=$1
    local current_version=$2
    local latest_version_cmd=$3
    local update_files=$4
    
    echo -n "Checking $tool... "
    latest_version=$(eval $latest_version_cmd)
    
    if [ "$current_version" = "$latest_version" ]; then
        echo -e "${GREEN}✓${NC} Up to date ($current_version)"
    else
        echo -e "${YELLOW}⚠${NC} Update available: $current_version → $latest_version"
        
        # Update version in files
        for file in $update_files; do
            if [ -f "$PROJECT_ROOT/$file" ]; then
                sed -i.bak "s/$current_version/$latest_version/g" "$PROJECT_ROOT/$file"
                rm -f "$PROJECT_ROOT/$file.bak"
                echo "  Updated: $file"
            fi
        done
    fi
}

# Update Kubernetes components
echo -e "\n📦 Kubernetes Components:"
echo "------------------------"

# PostgreSQL
POSTGRES_VERSION=$(grep -o 'postgres:[0-9.]*-alpine' "$PROJECT_ROOT/infra/base/postgres/deployment.yaml" | head -1 | cut -d: -f2)
POSTGRES_LATEST=$(curl -s https://registry.hub.docker.com/v2/repositories/library/postgres/tags | jq -r '.results[] | select(.name | test("^[0-9.]*-alpine$")) | .name' | sort -V | tail -1)
check_update_tool "PostgreSQL" "$POSTGRES_VERSION" "echo $POSTGRES_LATEST" "infra/base/postgres/deployment.yaml"

# Redis
REDIS_VERSION=$(grep -o 'redis:[0-9.]*-alpine' "$PROJECT_ROOT/infra/base/redis/deployment.yaml" | head -1 | cut -d: -f2)
REDIS_LATEST=$(curl -s https://registry.hub.docker.com/v2/repositories/library/redis/tags | jq -r '.results[] | select(.name | test("^[0-9.]*-alpine$")) | .name' | sort -V | tail -1)
check_update_tool "Redis" "$REDIS_VERSION" "echo $REDIS_LATEST" "infra/base/redis/deployment.yaml"

# Update monitoring components
echo -e "\n📊 Monitoring Components:"
echo "-------------------------"

# Prometheus
PROM_VERSION=$(grep -o 'prom/prometheus:v[0-9.]*' "$PROJECT_ROOT/infra/modules/observability/prometheus/prometheus.yaml" | head -1 | cut -d: -f2)
PROM_LATEST=$(curl -s https://api.github.com/repos/prometheus/prometheus/releases/latest | jq -r '.tag_name')
check_update_tool "Prometheus" "$PROM_VERSION" "echo $PROM_LATEST" "infra/modules/observability/prometheus/prometheus.yaml"

# Grafana
GRAFANA_VERSION=$(grep -o 'grafana/grafana:[0-9.]*' "$PROJECT_ROOT/infra/modules/observability/grafana/deployment.yaml" | head -1 | cut -d: -f2)
GRAFANA_LATEST=$(curl -s https://api.github.com/repos/grafana/grafana/releases/latest | jq -r '.tag_name' | sed 's/v//')
check_update_tool "Grafana" "$GRAFANA_VERSION" "echo $GRAFANA_LATEST" "infra/modules/observability/grafana/deployment.yaml"

# Loki
LOKI_VERSION=$(grep -o 'grafana/loki:[0-9.]*' "$PROJECT_ROOT/infra/modules/observability/logging/loki/deployment.yaml" | head -1 | cut -d: -f2)
LOKI_LATEST=$(curl -s https://api.github.com/repos/grafana/loki/releases/latest | jq -r '.tag_name' | sed 's/v//')
check_update_tool "Loki" "$LOKI_VERSION" "echo $LOKI_LATEST" "infra/modules/observability/logging/loki/deployment.yaml"

# Update Istio
echo -e "\n🕸️  Service Mesh:"
echo "-----------------"

ISTIO_VERSION=$(grep -o 'ISTIO_VERSION=.*' "$PROJECT_ROOT/.env.example" | cut -d= -f2)
ISTIO_LATEST=$(curl -s https://api.github.com/repos/istio/istio/releases/latest | jq -r '.tag_name')
check_update_tool "Istio" "$ISTIO_VERSION" "echo $ISTIO_LATEST" ".env.example"

# Update K3D/K3s
echo -e "\n☸️  K3D/K3s:"
echo "------------"

K3D_VERSION=$(grep -o 'K3D_VERSION=.*' "$PROJECT_ROOT/.env.example" | cut -d= -f2)
K3D_LATEST=$(curl -s https://api.github.com/repos/k3d-io/k3d/releases/latest | jq -r '.tag_name')
check_update_tool "K3D" "$K3D_VERSION" "echo $K3D_LATEST" ".env.example"

K3S_VERSION=$(grep -o 'K3S_VERSION=.*' "$PROJECT_ROOT/.env.example" | cut -d= -f2)
K3S_LATEST=$(curl -s https://api.github.com/repos/k3s-io/k3s/releases/latest | jq -r '.tag_name' | sed 's/+/-/')
check_update_tool "K3s" "$K3S_VERSION" "echo $K3S_LATEST" ".env.example"

# Generate update report
echo -e "\n📋 Update Report"
echo "================"

if git diff --quiet; then
    echo -e "${GREEN}✓${NC} All dependencies are up to date!"
else
    echo -e "${YELLOW}⚠${NC} Updates available. Review changes with: git diff"
    echo -e "\nTo apply updates:"
    echo "  git add -A"
    echo "  git commit -m 'chore: update dependencies'"
fi

echo -e "\n✅ Dependency check complete!"