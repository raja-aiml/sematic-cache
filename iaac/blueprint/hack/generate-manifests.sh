#!/bin/bash
set -euo pipefail

# Generate Manifests Script
# Renders all Kustomize manifests for review or CI/CD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/generated-manifests"

echo "🏗️  Generating Kubernetes Manifests"
echo "=================================="

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to generate manifests
generate_manifests() {
    local name=$1
    local path=$2
    local output=$3
    
    echo -n "Generating $name... "
    
    if kubectl kustomize "$path" > "$output" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $(wc -l < "$output") lines"
    else
        echo -e "${RED}✗${NC} Failed"
        return 1
    fi
}

# Generate base manifests
echo -e "\n📁 Base Manifests:"
echo "------------------"
generate_manifests "Infrastructure Base" "$PROJECT_ROOT/infra/base" "$OUTPUT_DIR/base-infra.yaml"

# Generate overlay manifests
echo -e "\n🔀 Overlay Manifests:"
echo "--------------------"
mkdir -p "$OUTPUT_DIR/overlays"
generate_manifests "Local Overlay" "$PROJECT_ROOT/infra/overlays/local" "$OUTPUT_DIR/overlays/local.yaml"
generate_manifests "Dev Overlay" "$PROJECT_ROOT/infra/overlays/dev" "$OUTPUT_DIR/overlays/dev.yaml"

# Generate module manifests
echo -e "\n🧩 Module Manifests:"
echo "-------------------"
mkdir -p "$OUTPUT_DIR/modules"
generate_manifests "Istio Module" "$PROJECT_ROOT/infra/modules/istio" "$OUTPUT_DIR/modules/istio.yaml"
generate_manifests "Observability Module" "$PROJECT_ROOT/infra/modules/observability" "$OUTPUT_DIR/modules/observability.yaml"
generate_manifests "Security Module" "$PROJECT_ROOT/infra/modules/security" "$OUTPUT_DIR/modules/security.yaml"

# Generate scenario manifests
echo -e "\n🎯 Scenario Manifests:"
echo "---------------------"
mkdir -p "$OUTPUT_DIR/scenarios"
generate_manifests "Minimal Scenario" "$PROJECT_ROOT/scenarios/minimal" "$OUTPUT_DIR/scenarios/minimal.yaml"
generate_manifests "Development Scenario" "$PROJECT_ROOT/scenarios/development" "$OUTPUT_DIR/scenarios/development.yaml"
generate_manifests "Service Mesh Scenario" "$PROJECT_ROOT/scenarios/service-mesh" "$OUTPUT_DIR/scenarios/service-mesh.yaml"
generate_manifests "Monitoring Only" "$PROJECT_ROOT/scenarios/monitoring-only" "$OUTPUT_DIR/scenarios/monitoring-only.yaml"
generate_manifests "Full Stack" "$PROJECT_ROOT/scenarios/full-stack" "$OUTPUT_DIR/scenarios/full-stack.yaml"

# Generate summary
echo -e "\n📊 Summary:"
echo "-----------"
echo "Total files generated: $(find "$OUTPUT_DIR" -name "*.yaml" | wc -l)"
echo "Total YAML lines: $(find "$OUTPUT_DIR" -name "*.yaml" -exec wc -l {} + | tail -1 | awk '{print $1}')"

# Validate manifests
echo -e "\n🔍 Validating Manifests:"
echo "-----------------------"

validation_errors=0
for file in $(find "$OUTPUT_DIR" -name "*.yaml"); do
    echo -n "Validating $(basename "$file")... "
    if kubectl --dry-run=client apply -f "$file" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        ((validation_errors++))
    fi
done

if [ $validation_errors -eq 0 ]; then
    echo -e "\n${GREEN}✅ All manifests validated successfully!${NC}"
else
    echo -e "\n${RED}❌ $validation_errors manifest(s) failed validation${NC}"
    exit 1
fi

echo -e "\n📁 Manifests generated in: $OUTPUT_DIR"