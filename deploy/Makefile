# =====================================
# 🚀 SEMANTIC CACHE DEPLOYMENT MAKEFILE
# =====================================

.PHONY: help setup build deploy test quick-test status logs cleanup reset full
.DEFAULT_GOAL := help

# Configuration
CLUSTER_NAME := sematic-cache
WORKFLOW_SCRIPT := ./workflow.sh

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help: ## Show this help message
	@echo "🚀 Semantic Cache Deployment Commands"
	@echo "====================================="
	@echo
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make full          # Complete end-to-end workflow"
	@echo "  make quick         # Quick development workflow"
	@echo "  make status        # Check deployment status"
	@echo "  make reset         # Reset and redeploy everything"

# === CORE WORKFLOWS ===

full: ## Complete production workflow: setup → build → deploy → test
	@echo "🚀 Running complete production workflow..."
	$(WORKFLOW_SCRIPT) full

quick: ## Quick development workflow with essential tests only
	@echo "⚡ Running quick development workflow..."
	$(WORKFLOW_SCRIPT) --quick full

reset: ## Complete reset: cleanup → setup → build → deploy
	@echo "🔄 Resetting entire deployment..."
	$(WORKFLOW_SCRIPT) reset

# === INDIVIDUAL STEPS ===

setup: ## Create k3d cluster and deploy infrastructure
	@echo "🏗️  Setting up cluster and infrastructure..."
	$(WORKFLOW_SCRIPT) setup

build: ## Build Docker image and import to cluster
	@echo "🔨 Building application..."
	$(WORKFLOW_SCRIPT) build

deploy: ## Deploy application to cluster
	@echo "🚀 Deploying application..."
	$(WORKFLOW_SCRIPT) deploy

test: ## Run comprehensive end-to-end tests
	@echo "🧪 Running comprehensive tests..."
	$(WORKFLOW_SCRIPT) test

quick-test: ## Run essential tests only (faster)
	@echo "⚡ Running quick tests..."
	$(WORKFLOW_SCRIPT) quick-test

# === MONITORING & MAINTENANCE ===

status: ## Show current deployment status
	@echo "📊 Checking deployment status..."
	$(WORKFLOW_SCRIPT) status

logs: ## Show application logs
	@echo "📜 Showing application logs..."
	$(WORKFLOW_SCRIPT) logs

health: ## Quick health check of all services
	@echo "❤️  Performing health check..."
	@echo "API Health:"
	@curl -s http://localhost:8080/semantic-cache/health | jq . || echo "❌ API not responding"
	@echo
	@echo "Web Interface:"
	@curl -s http://localhost:8080/web/ | head -1 || echo "❌ Web not responding"

metrics: ## Show API metrics
	@echo "📈 API Metrics:"
	@curl -s http://localhost:8080/semantic-cache/metrics | jq . || echo "❌ Metrics not available"

# === API TESTING ===

api-test: ## Test API endpoints with sample data
	@echo "🔬 Testing API endpoints..."
	@echo "Testing health endpoint:"
	@curl -s http://localhost:8080/semantic-cache/health
	@echo
	@echo "Testing cache SET:"
	@curl -s -X POST http://localhost:8080/semantic-cache/set \
		-H "Content-Type: application/json" \
		-d '{"prompt":"What is Kubernetes?","answer":"Container orchestration platform","modelName":"test"}'
	@echo
	@echo "Testing cache GET:"
	@curl -s -X POST http://localhost:8080/semantic-cache/get \
		-H "Content-Type: application/json" \
		-d '{"prompt":"What is Kubernetes?"}'

# === CLEANUP ===

cleanup: ## Destroy cluster and cleanup all resources
	@echo "🧹 Cleaning up all resources..."
	$(WORKFLOW_SCRIPT) cleanup

clean: cleanup ## Alias for cleanup

# === DEVELOPMENT HELPERS ===

dev: ## Development cycle: build → deploy → quick-test
	@echo "🔧 Running development cycle..."
	$(WORKFLOW_SCRIPT) build deploy quick-test

redeploy: ## Quick redeploy: build → deploy
	@echo "🔄 Redeploying application..."
	$(WORKFLOW_SCRIPT) build deploy

watch-pods: ## Watch pod status in real-time
	@echo "👀 Watching pods..."
	watch kubectl get pods -A

watch-logs: ## Watch application logs in real-time
	@echo "📜 Watching logs..."
	kubectl logs -n app -l app=sematic-cache -f

# === CLUSTER MANAGEMENT ===

cluster-info: ## Show cluster information
	@echo "🏗️  Cluster Information:"
	@k3d cluster list
	@echo
	@kubectl cluster-info
	@echo
	@kubectl get nodes

cluster-events: ## Show recent cluster events
	@echo "📅 Recent Cluster Events:"
	@kubectl get events --all-namespaces --sort-by='.lastTimestamp' | tail -20

# === VERIFICATION ===

verify: ## Verify deployment is working correctly
	@echo "✅ Verifying deployment..."
	@echo "Checking cluster:"
	@k3d cluster list | grep $(CLUSTER_NAME) && echo "✅ Cluster exists" || echo "❌ Cluster missing"
	@echo "Checking pods:"
	@kubectl get pods -A | grep -E "(Running|Ready)" && echo "✅ Pods running" || echo "❌ Pod issues"
	@echo "Checking API:"
	@curl -s --max-time 5 http://localhost:8080/semantic-cache/health >/dev/null && echo "✅ API responding" || echo "❌ API not responding"
	@echo "Checking web:"
	@curl -s --max-time 5 http://localhost:8080/web/ >/dev/null && echo "✅ Web responding" || echo "❌ Web not responding"

# === DOCUMENTATION ===

docs: ## Show deployment documentation
	@echo "📚 Deployment Documentation:"
	@echo "Main README: ../README.md"
	@echo "Deploy README: ./README.md"  
	@echo "Test README: ./tests/README.md"
	@echo
	@echo "Quick reference:"
	@echo "  Cluster: $(CLUSTER_NAME)"
	@echo "  API: http://localhost:8080/semantic-cache/"
	@echo "  Web: http://localhost:8080/web/"
	@echo "  Logs: make logs"
	@echo "  Status: make status"

# === ADVANCED ===

debug: ## Debug deployment issues
	@echo "🐛 Debug Information:"
	@echo "=== Cluster Status ==="
	@k3d cluster list
	@echo
	@echo "=== Pod Status ==="
	@kubectl get pods -A -o wide
	@echo
	@echo "=== Service Status ==="
	@kubectl get svc -A
	@echo
	@echo "=== Recent Events ==="
	@kubectl get events --all-namespaces --sort-by='.lastTimestamp' | tail -10
	@echo
	@echo "=== API Status ==="
	@curl -I http://localhost:8080/semantic-cache/health 2>/dev/null || echo "API not accessible"

benchmark: ## Run performance benchmark
	@echo "⚡ Running performance benchmark..."
	@echo "This will make 50 concurrent requests to test performance"
	@seq 1 50 | xargs -P 10 -I {} curl -s -w "%{time_total}s\n" -o /dev/null http://localhost:8080/semantic-cache/health | sort -n | tail -10

# === CI/CD HELPERS ===

ci-test: ## Run CI-friendly tests (no interactive prompts)
	@echo "🤖 Running CI tests..."
	$(WORKFLOW_SCRIPT) --quick --skip-cleanup full

validate: ## Validate configuration files
	@echo "✅ Validating configuration..."
	@kubectl kustomize . >/dev/null && echo "✅ Kustomization valid" || echo "❌ Kustomization invalid"
	@kubectl kustomize app/ >/dev/null && echo "✅ App kustomization valid" || echo "❌ App kustomization invalid"
	@kubectl kustomize infra/ >/dev/null && echo "✅ Infra kustomization valid" || echo "❌ Infra kustomization invalid"