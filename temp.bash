#!/bin/bash

set -euo pipefail

echo "🔍 COMPREHENSIVE PROJECT DIAGNOSTICS"
echo "======================================"
echo ""

echo "=== 📂 CURRENT DIRECTORY ==="
pwd
echo ""

echo "=== 📁 DIRECTORY CONTENTS ==="
ls -la
echo ""

echo "=== 🌳 DIRECTORY TREE (depth 3) ==="
if command -v tree &>/dev/null; then
  tree -L 3
else
  find . -type d -maxdepth 3 | head -20
fi
echo ""

echo "=== 🔍 GO FILES ==="
find . -name "*.go" -type f 2>/dev/null || echo "No Go files found."
echo ""

echo "=== 📄 go.mod / go.sum ==="
find . -name "go.mod" -type f 2>/dev/null
find . -name "go.sum" -type f 2>/dev/null
echo ""

echo "=== 🧬 GIT INFO ==="
git remote -v 2>/dev/null || echo "❌ Not a git repository"
git branch 2>/dev/null || echo "❌ No git branches"
git status 2>/dev/null || echo "❌ No git status available"
echo ""

echo "=== 📁 COMMON GO DIRECTORIES ==="
ls -la | grep -E "src|cmd|internal|pkg|app|server" || echo "No common Go directories found"
echo ""

echo "=== 🔎 SEMATIC-CACHE DIRECTORIES ==="
find . -type d \( -name "*sematic*" -o -name "*cache*" \) | head -10
echo ""

echo "=== 🔁 GO FILES IN PARENT DIR ==="
find .. -name "*.go" -type f 2>/dev/null | head -10
echo ""

echo "=== 🔍 SEMATIC-CACHE DIRECTORIES IN HOME ==="
find "$HOME" -type d \( -name "*sematic*" -o -name "*cache*" \) 2>/dev/null | head -5
echo ""

echo "=== 🐳 DOCKERFILES ==="
find . -name "Dockerfile" -type f 2>/dev/null
echo ""

echo "=== 🛠️ MAKEFILES / BUILD SCRIPTS ==="
find . \( -name "Makefile" -o -name "*.sh" -o -name "build.*" \) 2>/dev/null | head -10
echo ""

echo "=== 🧰 GO ENVIRONMENT ==="
go version 2>/dev/null || echo "❌ Go not installed"
go env GOPATH GOROOT 2>/dev/null || echo "❌ Go env not available"
echo ""

echo "=== 🚀 DEPLOYMENT YAML FILES ==="
find . \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | head -10
echo ""

echo "=== 📘 README / DOCS ==="
find . \( -name "README*" -o -name "*.md" \) 2>/dev/null | head -5
echo ""

echo "=== 🗂️ ALL PROJECT FILES (First 50) ==="
find . -type f 2>/dev/null | head -50
echo ""

echo "=== 📍 GIT ROOT DIRECTORY ==="
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
if [[ -n "$GIT_ROOT" ]]; then
  echo "$GIT_ROOT"
else
  echo "❌ Not in a git repository"
fi
echo ""

if [[ -n "$GIT_ROOT" && "$GIT_ROOT" != "$(pwd)" ]]; then
  echo "📌 Git root is different from current directory!"
  echo "Git root: $GIT_ROOT"
  echo "Current:  $(pwd)"
  echo ""
  echo "=== 📁 CONTENTS OF GIT ROOT ==="
  ls -la "$GIT_ROOT" | head -20
  echo ""
  echo "=== 🔍 GO FILES IN GIT ROOT ==="
  find "$GIT_ROOT" -name "*.go" -type f 2>/dev/null | head -10
  echo ""
fi

echo "✅ 🎯 DIAGNOSTICS COMPLETE!"
echo "Please share this output for issue analysis."