#!/bin/bash
set -euo pipefail

# Lint All Script
# Runs various linters on the K3D Blueprint project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Linting K3D Blueprint"
echo "======================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Track errors
TOTAL_ERRORS=0

# Function to run linter
run_linter() {
    local name=$1
    local command=$2
    
    echo -n "Running $name... "
    
    if eval "$command" > /tmp/lint-output.txt 2>&1; then
        echo -e "${GREEN}✓${NC} Passed"
    else
        echo -e "${RED}✗${NC} Failed"
        echo "  Errors:"
        cat /tmp/lint-output.txt | sed 's/^/    /'
        ((TOTAL_ERRORS++))
    fi
    rm -f /tmp/lint-output.txt
}

# Check for required tools
echo "Checking required tools..."
for tool in yamllint kubeval shellcheck hadolint; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} $tool not found. Installing..."
        case $tool in
            yamllint)
                pip install yamllint
                ;;
            kubeval)
                wget https://github.com/instrumenta/kubeval/releases/latest/download/kubeval-linux-amd64.tar.gz
                tar xf kubeval-linux-amd64.tar.gz
                sudo mv kubeval /usr/local/bin
                rm kubeval-linux-amd64.tar.gz
                ;;
            shellcheck)
                sudo apt-get update && sudo apt-get install -y shellcheck
                ;;
            hadolint)
                wget -O hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
                chmod +x hadolint
                sudo mv hadolint /usr/local/bin
                ;;
        esac
    fi
done

# Lint YAML files
echo -e "\n📝 YAML Files:"
echo "--------------"

# Create yamllint config
cat > /tmp/yamllint-config.yaml <<EOF
extends: relaxed
rules:
  line-length:
    max: 120
  comments:
    min-spaces-from-content: 1
  truthy:
    allowed-values: ['true', 'false', 'yes', 'no', 'on', 'off']
EOF

run_linter "YAML Lint" "yamllint -c /tmp/yamllint-config.yaml $PROJECT_ROOT --exclude $PROJECT_ROOT/generated-manifests"

# Validate Kubernetes manifests
echo -e "\n☸️  Kubernetes Manifests:"
echo "----------------------"

for file in $(find "$PROJECT_ROOT" -name "*.yaml" -o -name "*.yml" | grep -E "(deployment|service|configmap|secret|ingress|networkpolicy)" | grep -v generated-manifests); do
    if grep -q "kind:" "$file"; then
        run_linter "$(basename $file)" "kubeval --ignore-missing-schemas $file"
    fi
done

# Lint shell scripts
echo -e "\n🐚 Shell Scripts:"
echo "----------------"

for script in $(find "$PROJECT_ROOT" -name "*.sh" -type f); do
    run_linter "$(basename $script)" "shellcheck -x $script"
done

# Check Kustomization files
echo -e "\n🔧 Kustomization Files:"
echo "---------------------"

for kustomization in $(find "$PROJECT_ROOT" -name "kustomization.yaml" -type f); do
    dir=$(dirname "$kustomization")
    run_linter "$(basename $dir)" "kubectl kustomize $dir --enable-helm > /dev/null"
done

# Lint Dockerfiles
echo -e "\n🐳 Dockerfiles:"
echo "---------------"

for dockerfile in $(find "$PROJECT_ROOT" -name "Dockerfile*" -type f); do
    run_linter "$(basename $dockerfile)" "hadolint $dockerfile"
done

# Check for security issues
echo -e "\n🔒 Security Checks:"
echo "------------------"

# Check for hardcoded secrets
echo -n "Checking for secrets... "
if grep -r -E "(password|secret|key|token):\s*['\"]?[a-zA-Z0-9]+" "$PROJECT_ROOT" --include="*.yaml" --include="*.yml" --exclude-dir=generated-manifests | grep -v -E "(changeme|example|placeholder|secret-name|secretKeyRef)" > /tmp/secrets-check.txt; then
    if [ -s /tmp/secrets-check.txt ]; then
        echo -e "${RED}✗${NC} Potential secrets found:"
        cat /tmp/secrets-check.txt | sed 's/^/    /'
        ((TOTAL_ERRORS++))
    else
        echo -e "${GREEN}✓${NC} No secrets found"
    fi
else
    echo -e "${GREEN}✓${NC} No secrets found"
fi
rm -f /tmp/secrets-check.txt

# Check for resource limits
echo -n "Checking resource limits... "
missing_limits=$(grep -r "kind: Deployment" "$PROJECT_ROOT" --include="*.yaml" -l | while read file; do
    if ! grep -q "resources:" "$file"; then
        echo "$file"
    fi
done)

if [ -n "$missing_limits" ]; then
    echo -e "${YELLOW}⚠${NC} Deployments missing resource limits:"
    echo "$missing_limits" | sed 's/^/    /'
else
    echo -e "${GREEN}✓${NC} All deployments have resource limits"
fi

# Summary
echo -e "\n📊 Lint Summary"
echo "==============="

if [ $TOTAL_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
else
    echo -e "${RED}❌ $TOTAL_ERRORS check(s) failed${NC}"
    exit 1
fi

# Cleanup
rm -f /tmp/yamllint-config.yaml