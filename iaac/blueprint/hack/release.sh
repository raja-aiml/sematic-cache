#!/bin/bash
set -euo pipefail

# Release Script
# Prepares a new release of the K3D Blueprint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 K3D Blueprint Release Process"
echo "==============================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get current version
CURRENT_VERSION=$(grep -E "^## \[" "$PROJECT_ROOT/CHANGELOG.md" | head -1 | sed 's/.*\[\(.*\)\].*/\1/' || echo "0.0.0")
echo "Current version: $CURRENT_VERSION"

# Prompt for new version
echo -e "\nEnter new version (current: $CURRENT_VERSION):"
read -r NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}❌ Version cannot be empty${NC}"
    exit 1
fi

# Validate version format
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo -e "${RED}❌ Invalid version format. Use semantic versioning (e.g., 1.2.3)${NC}"
    exit 1
fi

# Run pre-release checks
echo -e "\n📋 Running pre-release checks..."

# Check working directory
echo -n "Checking for uncommitted changes... "
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}✗${NC}"
    echo "Please commit or stash your changes before releasing."
    exit 1
else
    echo -e "${GREEN}✓${NC}"
fi

# Run verification
echo -n "Verifying blueprint structure... "
if bash "$SCRIPT_DIR/verify-blueprint.sh" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Blueprint verification failed. Run './hack/verify-blueprint.sh' for details."
    exit 1
fi

# Run linting
echo -n "Running linters... "
if bash "$SCRIPT_DIR/lint-all.sh" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC} Some linting issues found"
fi

# Update files
echo -e "\n📝 Updating version in files..."

# Update CHANGELOG.md
echo -n "Updating CHANGELOG.md... "
TODAY=$(date +%Y-%m-%d)
sed -i.bak "s/## \[Unreleased\]/## [Unreleased]\n\n## [$NEW_VERSION] - $TODAY/" "$PROJECT_ROOT/CHANGELOG.md"
rm -f "$PROJECT_ROOT/CHANGELOG.md.bak"
echo -e "${GREEN}✓${NC}"

# Update version references in documentation
echo -n "Updating documentation... "
find "$PROJECT_ROOT" -name "*.md" -type f -exec sed -i.bak "s/version: $CURRENT_VERSION/version: $NEW_VERSION/g" {} \;
find "$PROJECT_ROOT" -name "*.md.bak" -type f -delete
echo -e "${GREEN}✓${NC}"

# Generate release manifests
echo -e "\n📦 Generating release artifacts..."
mkdir -p "$PROJECT_ROOT/releases/v$NEW_VERSION"

# Generate all manifests
echo "Generating manifests..."
for scenario in minimal development service-mesh monitoring-only full-stack; do
    echo -n "  $scenario... "
    kubectl kustomize "$PROJECT_ROOT/scenarios/$scenario" > "$PROJECT_ROOT/releases/v$NEW_VERSION/$scenario.yaml"
    echo -e "${GREEN}✓${NC}"
done

# Create release tarball
echo -n "Creating release archive... "
cd "$PROJECT_ROOT"
tar -czf "releases/k3d-blueprint-v$NEW_VERSION.tar.gz" \
    --exclude="releases" \
    --exclude=".git" \
    --exclude="generated-manifests" \
    --exclude="*.log" \
    .
cd - > /dev/null
echo -e "${GREEN}✓${NC}"

# Generate release notes
echo -e "\n📋 Generating release notes..."
cat > "$PROJECT_ROOT/releases/v$NEW_VERSION/RELEASE_NOTES.md" <<EOF
# K3D Blueprint v$NEW_VERSION

Released: $TODAY

## What's New

### Features
- [Add feature descriptions here]

### Improvements
- [Add improvement descriptions here]

### Bug Fixes
- [Add bug fix descriptions here]

## Installation

### Quick Start
\`\`\`bash
# Download and extract
wget https://github.com/your-org/k3d-blueprint/releases/download/v$NEW_VERSION/k3d-blueprint-v$NEW_VERSION.tar.gz
tar -xzf k3d-blueprint-v$NEW_VERSION.tar.gz
cd k3d-blueprint

# Deploy minimal scenario
kubectl apply -k scenarios/minimal
\`\`\`

### Pre-rendered Manifests
Pre-rendered manifests for each scenario are available in the release:
- \`minimal.yaml\` - Basic infrastructure only
- \`development.yaml\` - Full development environment
- \`service-mesh.yaml\` - Istio service mesh deployment
- \`monitoring-only.yaml\` - Observability stack only
- \`full-stack.yaml\` - Complete production deployment

## Compatibility
- Kubernetes: 1.28+
- K3D: v5.6.0+
- K3s: v1.28.5+

## Changelog
See [CHANGELOG.md](../../CHANGELOG.md) for detailed changes.
EOF

echo -e "${GREEN}✓${NC} Release notes generated"

# Create git tag
echo -e "\n🏷️  Creating git tag..."
git add -A
git commit -m "chore: release v$NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Summary
echo -e "\n✅ Release v$NEW_VERSION prepared!"
echo -e "\n📦 Release artifacts created in: releases/v$NEW_VERSION/"
echo -e "\n📋 Next steps:"
echo "1. Review the changes: git show"
echo "2. Review release notes: releases/v$NEW_VERSION/RELEASE_NOTES.md"
echo "3. Push changes: git push origin main --tags"
echo "4. Create GitHub release with the generated artifacts"
echo "5. Upload the tarball: releases/k3d-blueprint-v$NEW_VERSION.tar.gz"

# Offer to push
echo -e "\nDo you want to push the release now? (y/N)"
read -r PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "y" ] || [ "$PUSH_CONFIRM" = "Y" ]; then
    echo "Pushing release..."
    git push origin main --tags
    echo -e "${GREEN}✅ Release pushed successfully!${NC}"
    echo -e "\n🌐 Create the GitHub release at:"
    echo "https://github.com/your-org/k3d-blueprint/releases/new?tag=v$NEW_VERSION"
else
    echo -e "\n${YELLOW}⚠${NC} Remember to push when ready:"
    echo "  git push origin main --tags"
fi