# Semantic Cache CLI Tools Build System
# Simple command structure following KISS principle

# Configuration
BINARY_NAME := semantic-cache-tool
BINARY_PATH := bin/$(BINARY_NAME)
GO_FILES := $(shell find . -type f -name '*.go' -not -path "./vendor/*" -not -path "./bin/*")
COVERAGE_FILE := coverage.out
CONFIG_PATH := ../.env.app

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Default
.DEFAULT_GOAL := help

##@ Core Commands

help: ## Show this help
	@echo "$(GREEN)Semantic Cache CLI Tools$(RESET)"
	@echo "$(YELLOW)Usage: make <command> [<subcommand>]$(RESET)"
	@echo ""
	@echo ""
	@echo "$(GREEN)Core Commands$(RESET)"
	@echo "  $(CYAN)help$(RESET)                  Show this help"
	@echo "  $(CYAN)build$(RESET)                 Build CLI binary"
	@echo "  $(CYAN)install$(RESET)               Install CLI globally"
	@echo "  $(CYAN)clean$(RESET)                 Clean artifacts"
	@echo ""
	@echo "$(GREEN)Code Quality$(RESET)"
	@echo "  $(CYAN)fmt$(RESET)                   Format code"
	@echo "  $(CYAN)lint$(RESET)                  Run linters"
	@echo "  $(CYAN)check$(RESET)                 Run all checks"
	@echo ""
	@echo "$(GREEN)Workflows$(RESET)"
	@echo "  $(CYAN)all$(RESET)                   Full build (clean → format → lint → test → build)"
	@echo "  $(CYAN)ci$(RESET)                    CI pipeline (deps → checks → build)"

build: ## Build CLI binary
	@echo "$(GREEN)Building CLI...$(RESET)"
	@go build -o $(BINARY_PATH) -ldflags="-s -w" .
	@echo "$(GREEN)✓ Built: $(BINARY_PATH)$(RESET)"

install: build ## Install CLI globally
	@echo "$(GREEN)Installing CLI...$(RESET)"
	@go install .
	@echo "$(GREEN)✓ Installed: $(BINARY_NAME)$(RESET)"

clean: ## Clean artifacts
	@echo "$(YELLOW)Cleaning...$(RESET)"
	@rm -rf bin/ $(COVERAGE_FILE)
	@go clean -cache
	@echo "$(GREEN)✓ Cleaned$(RESET)"

##@ Code Quality

fmt: ## Format code
	@echo "$(CYAN)Formatting...$(RESET)"
	@gofmt -w -s $(GO_FILES)
	@go mod tidy
	@echo "$(GREEN)✓ Formatted$(RESET)"

lint: ## Run linters
	@echo "$(CYAN)Linting...$(RESET)"
	@go vet ./...
	@echo "$(GREEN)✓ Linted$(RESET)"

check: fmt lint test ## Run all checks
	@echo "$(GREEN)✓ All checks passed$(RESET)"

##@ Testing

test: ## Run tests
	@echo "$(CYAN)Testing...$(RESET)"
	@go test -v -cover -race ./...

# Subcommands for test
cover: ## Coverage report
	@echo "$(CYAN)Generating coverage report...$(RESET)"
	@go test -coverprofile=$(COVERAGE_FILE) ./...
	@go tool cover -html=$(COVERAGE_FILE)

watch: ## Watch mode
	@which gotestsum > /dev/null || go install gotest.tools/gotestsum@latest
	@gotestsum --watch

##@ CLI Commands (for testing)

# Cache commands
cache-get: build ## Test cache get
	@echo "$(CYAN)Testing cache get...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) cache get test-key

cache-set: build ## Test cache set
	@echo "$(CYAN)Testing cache set...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) cache set test-key "test-value"

cache-clear: build ## Test cache clear
	@echo "$(CYAN)Testing cache clear...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) cache clear

# Health commands
health: build ## Check health
	@echo "$(CYAN)Checking health...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) health check

health-live: build ## Liveness check
	@echo "$(CYAN)Checking liveness...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) health live

health-ready: build ## Readiness check
	@echo "$(CYAN)Checking readiness...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) health ready

# Stats command
stats: build ## Get statistics
	@echo "$(CYAN)Getting statistics...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) stats

# Database commands
db-status: build ## Database status
	@echo "$(CYAN)Checking database status...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) database status

db-migrate: build ## Run migrations
	@echo "$(CYAN)Running migrations...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) database migrate

db-reset: build ## Reset database
	@echo "$(RED)Resetting database...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) database reset

# Search command
search: build ## Search similar entries
	@echo "$(CYAN)Searching similar entries...$(RESET)"
	@./$(BINARY_PATH) --config-path $(CONFIG_PATH) search "test query"

##@ Dependencies

deps: ## Install dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	@go mod download
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

# Subcommands for deps
update: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	@go get -u ./...
	@go mod tidy
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

verify: ## Verify dependencies
	@echo "$(CYAN)Verifying dependencies...$(RESET)"
	@go mod verify
	@echo "$(GREEN)✓ Dependencies verified$(RESET)"

##@ Workflows

all: clean fmt lint test build ## Full build
	@echo "$(GREEN)✓ Full build complete$(RESET)"

ci: deps check build ## CI pipeline
	@echo "$(GREEN)✓ CI pipeline complete$(RESET)"

quick: fmt build ## Quick build (format + build)
	@echo "$(GREEN)✓ Quick build complete$(RESET)"

.PHONY: help build install clean \
        fmt lint check \
        test cover watch \
        cache-get cache-set cache-clear \
        health health-live health-ready \
        stats \
        db-status db-migrate db-reset \
        search \
        deps update verify \
        all ci quick