#!/bin/bash
set -euo pipefail

# Verify Blueprint Script
# Validates the overall K3D Blueprint project structure and integrity

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Verifying K3D Blueprint Structure"
echo "==================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Track issues
ISSUES=0

# Function to check directory
check_directory() {
    local path=$1
    local description=$2
    
    echo -n "Checking $description... "
    if [ -d "$PROJECT_ROOT/$path" ]; then
        echo -e "${GREEN}✓${NC} Found"
    else
        echo -e "${RED}✗${NC} Missing"
        ((ISSUES++))
    fi
}

# Function to check file
check_file() {
    local path=$1
    local description=$2
    
    echo -n "Checking $description... "
    if [ -f "$PROJECT_ROOT/$path" ]; then
        echo -e "${GREEN}✓${NC} Found"
    else
        echo -e "${RED}✗${NC} Missing"
        ((ISSUES++))
    fi
}

# Check root structure
echo "📁 Root Structure:"
echo "-----------------"
check_file ".gitignore" ".gitignore"
check_file ".gitattributes" ".gitattributes"
check_file ".env.example" ".env.example"
check_file "LICENSE" "LICENSE"
check_file "README.md" "README.md"
check_file "CHANGELOG.md" "CHANGELOG.md"
check_file "Taskfile.yaml" "Taskfile.yaml"

# Check infrastructure
echo -e "\n🏗️  Infrastructure:"
echo "------------------"
check_directory "infra" "infra directory"
check_directory "infra/base" "base configurations"
check_directory "infra/base/postgres" "PostgreSQL base"
check_directory "infra/base/redis" "Redis base"
check_directory "infra/overlays" "overlays"
check_directory "infra/overlays/local" "local overlay"
check_directory "infra/overlays/dev" "dev overlay"
check_directory "infra/modules" "modules"
check_directory "infra/modules/istio" "Istio module"
check_directory "infra/modules/observability" "observability module"
check_directory "infra/modules/security" "security module"

# Check observability components
echo -e "\n📊 Observability Components:"
echo "---------------------------"
check_directory "infra/modules/observability/prometheus" "Prometheus"
check_directory "infra/modules/observability/grafana" "Grafana"
check_directory "infra/modules/observability/logging" "logging stack"
check_directory "infra/modules/observability/tracing" "tracing stack"

# Check scenarios
echo -e "\n🎯 Scenarios:"
echo "-------------"
check_directory "scenarios" "scenarios directory"
check_directory "scenarios/minimal" "minimal scenario"
check_directory "scenarios/development" "development scenario"
check_directory "scenarios/service-mesh" "service-mesh scenario"
check_directory "scenarios/monitoring-only" "monitoring-only scenario"
check_directory "scenarios/full-stack" "full-stack scenario"

# Check validation kit
echo -e "\n✅ Validation Kit:"
echo "-----------------"
check_directory "validation-kit" "validation-kit directory"
check_directory "validation-kit/tests" "test definitions"
check_directory "validation-kit/scripts" "test scripts"
check_directory "validation-kit/seed-data" "seed data"

# Check hack scripts
echo -e "\n🔧 Hack Scripts:"
echo "---------------"
check_directory "hack" "hack directory"
check_file "hack/update-dependencies.sh" "update-dependencies.sh"
check_file "hack/generate-manifests.sh" "generate-manifests.sh"
check_file "hack/lint-all.sh" "lint-all.sh"
check_file "hack/test-all.sh" "test-all.sh"
check_file "hack/verify-blueprint.sh" "verify-blueprint.sh"
check_file "hack/benchmark.sh" "benchmark.sh"
check_file "hack/release.sh" "release.sh"

# Check kustomization files
echo -e "\n🔧 Kustomization Files:"
echo "---------------------"
for dir in $(find "$PROJECT_ROOT" -type d -name "base" -o -name "overlays" -o -name "modules" -o -name "scenarios" | grep -v generated-manifests); do
    if [ -d "$dir" ]; then
        for subdir in $(find "$dir" -maxdepth 2 -type d ! -path "$dir"); do
            if [ ! -f "$subdir/kustomization.yaml" ]; then
                echo -e "${YELLOW}⚠${NC} Missing kustomization.yaml in $(realpath --relative-to="$PROJECT_ROOT" "$subdir")"
                ((ISSUES++))
            fi
        done
    fi
done

# Check for required labels
echo -e "\n🏷️  Label Verification:"
echo "--------------------"
echo -n "Checking namespace labels... "
namespace_files=$(find "$PROJECT_ROOT" -name "namespace*.yaml" -type f | grep -v generated-manifests)
missing_labels=0
for file in $namespace_files; do
    if grep -q "kind: Namespace" "$file"; then
        if ! grep -q "labels:" "$file"; then
            ((missing_labels++))
        fi
    fi
done
if [ $missing_labels -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All namespaces have labels"
else
    echo -e "${YELLOW}⚠${NC} $missing_labels namespace(s) missing labels"
fi

# Check for README files
echo -e "\n📚 Documentation:"
echo "----------------"
for scenario in minimal development service-mesh monitoring-only full-stack; do
    check_file "scenarios/$scenario/README.md" "$scenario README"
done

# Check file permissions
echo -e "\n🔒 File Permissions:"
echo "-------------------"
echo -n "Checking script permissions... "
scripts_without_exec=$(find "$PROJECT_ROOT" -name "*.sh" -type f ! -perm -u+x | wc -l)
if [ $scripts_without_exec -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All scripts are executable"
else
    echo -e "${YELLOW}⚠${NC} $scripts_without_exec script(s) not executable"
    find "$PROJECT_ROOT" -name "*.sh" -type f ! -perm -u+x -exec chmod +x {} \;
    echo "  Fixed: Made all scripts executable"
fi

# Summary
echo -e "\n📊 Verification Summary"
echo "======================"

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ Blueprint structure is complete!${NC}"
    echo -e "\nThe K3D Blueprint is ready for use."
else
    echo -e "${RED}❌ Found $ISSUES issue(s)${NC}"
    echo -e "\nPlease fix the missing components before proceeding."
    exit 1
fi

# Additional recommendations
echo -e "\n💡 Recommendations:"
echo "------------------"
echo "1. Run './hack/lint-all.sh' to check code quality"
echo "2. Run './hack/test-all.sh' to verify functionality"
echo "3. Run './hack/generate-manifests.sh' to pre-render manifests"
echo "4. Check './scenarios/*/README.md' for usage instructions"