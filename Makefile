# Semantic Cache Build System
# Simple command structure following KISS principle

# Configuration
BINARY_NAME := semantic-cache
BINARY_PATH := bin/$(BINARY_NAME)
DOCKER_IMAGE := semantic-cache:latest
DOCKER_COMPOSE_DIR := deployments/local
GO_FILES := $(shell find . -type f -name '*.go' -not -path "./vendor/*")
COVERAGE_FILE := coverage.out

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# Default
.DEFAULT_GOAL := help

##@ Core Commands

help: ## Show this help
	@echo "$(GREEN)Semantic Cache$(RESET)"
	@echo "$(YELLOW)Usage: make <command> [<subcommand>]$(RESET)"
	@echo ""
	@echo ""
	@echo "$(GREEN)Core Commands$(RESET)"
	@echo "  $(CYAN)help$(RESET)                  Show this help"
	@echo "  $(CYAN)build$(RESET)                 Build binary"
	@echo "  $(CYAN)run$(RESET)                   Run application"
	@echo "  $(CYAN)clean$(RESET)                 Clean artifacts"
	@echo "  $(CYAN)dev$(RESET)                   Development mode (hot reload)"
	@echo ""
	@echo "$(GREEN)Code Quality$(RESET)"
	@echo "  $(CYAN)fmt$(RESET)                   Format code"
	@echo "  $(CYAN)lint$(RESET)                  Run linters"
	@echo "  $(CYAN)check$(RESET)                 Run all checks"
	@echo ""
	@echo "$(GREEN)Testing$(RESET)"
	@echo "  $(CYAN)test$(RESET)                  Run tests"
	@echo "  $(CYAN)test cover$(RESET)            Coverage report"
	@echo "  $(CYAN)test bench$(RESET)            Run benchmarks"
	@echo "  $(CYAN)test watch$(RESET)            Watch mode"
	@echo ""
	@echo "$(GREEN)Docker$(RESET)"
	@echo "  $(CYAN)docker$(RESET)                Build image"
	@echo "  $(CYAN)docker up$(RESET)             Start services"
	@echo "  $(CYAN)docker down$(RESET)           Stop services"
	@echo "  $(CYAN)docker logs$(RESET)           View logs"
	@echo "  $(CYAN)docker restart$(RESET)        Restart services"
	@echo ""
	@echo "$(GREEN)Database$(RESET)"
	@echo "  $(CYAN)db migrate$(RESET)            Run migrations"
	@echo "  $(CYAN)db reset$(RESET)              Reset database"
	@echo ""
	@echo "$(GREEN)Dependencies$(RESET)"
	@echo "  $(CYAN)deps$(RESET)                  Install dependencies"
	@echo "  $(CYAN)deps update$(RESET)           Update dependencies"
	@echo "  $(CYAN)deps verify$(RESET)           Verify dependencies"
	@echo ""
	@echo "$(GREEN)Workflows$(RESET)"
	@echo "  $(CYAN)all$(RESET)                   Full build (clean → format → lint → test → build)"
	@echo "  $(CYAN)ci$(RESET)                    CI pipeline (deps → checks → build)"
	@echo "  $(CYAN)deploy$(RESET)                Deploy locally (docker → start services)"
	@echo ""
	@echo "$(GREEN)Stack Management$(RESET)"
	@echo "  $(CYAN)start$(RESET)                 Start full stack (builds infra + app + tools)"
	@echo "  $(CYAN)stop$(RESET)                  Stop full stack (removes containers + binaries)"
	@echo "  $(CYAN)status$(RESET)                Check stack status"
	@echo "  $(CYAN)logs$(RESET)                  View application logs"

build: ## Build binary
	@echo "Building..."
	@go build -o $(BINARY_PATH) -ldflags="-s -w" .

run: ## Run application
	@go build -o $(BINARY_PATH) -ldflags="-s -w" .
	@./$(BINARY_PATH)

clean: ## Clean artifacts
	@echo "Cleaning..."
	@rm -rf bin/ $(COVERAGE_FILE)
	@go clean -cache

dev: ## Development mode (hot reload)
	@which air > /dev/null || go install github.com/cosmtrek/air@latest
	@air

##@ Code Quality

fmt: ## Format code
	@echo "Formatting..."
	@gofmt -w -s $(GO_FILES)
	@go mod tidy

lint: ## Run linters
	@echo "Linting..."
	@go vet ./...

check: fmt lint test ## Run all checks

##@ Testing

test: ## Run tests
	@go test -v -cover -race ./...

# Subcommands for test
cover: ## Coverage report
	@go test -coverprofile=$(COVERAGE_FILE) ./...
	@go tool cover -html=$(COVERAGE_FILE)

bench: ## Run benchmarks
	@go test -bench=. -benchmem ./...

watch: ## Watch mode
	@which gotestsum > /dev/null || go install gotest.tools/gotestsum@latest
	@gotestsum --watch

##@ Docker

docker: ## Build image
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_IMAGE) -f deployments/docker/Dockerfile .

# Subcommands for docker
up: ## Start services
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose up -d

down: ## Stop services
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose down

logs: ## View application logs
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose logs -f semantic-cache

logs-all: ## View all service logs
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose logs -f

restart: down up ## Restart services

##@ Database

# Subcommands for db
migrate: ## Run migrations
	@echo "Running migrations..."
	@go run cmd/migrate/main.go up

reset: ## Reset database
	@echo "Resetting database..."
	@go run cmd/migrate/main.go down
	@go run cmd/migrate/main.go up

##@ Dependencies

deps: ## Install dependencies
	@go mod download

# Subcommands for deps
update: ## Update dependencies
	@go get -u ./...
	@go mod tidy

verify: ## Verify dependencies
	@go mod verify

##@ Workflows

all: clean fmt lint test build ## Full build
ci: deps check build ## CI pipeline
deploy: docker up ## Deploy locally

##@ Stack Management (Full Workflow)

start: ## Start full stack (builds infra + app + tools)
	@echo "$(GREEN)🚀 Starting full stack...$(RESET)"
	@if [ ! -f .env ]; then \
		echo "$(RED)⚠️  Missing .env file$(RESET)"; \
		echo "$(YELLOW)Please copy .env.example to .env and add your OPENAI_API_KEY$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)1. Building and starting all services (including app)...$(RESET)"
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose up -d --build
	@echo "$(CYAN)2. Waiting for services to be healthy...$(RESET)"
	@sleep 10
	@echo "$(CYAN)3. Building CLI tools...$(RESET)"
	@cd tools && make build
	@echo "$(GREEN)✅ Stack ready!$(RESET)"
	@echo ""
	@echo "$(YELLOW)Services:$(RESET)"
	@echo "  • PostgreSQL:    localhost:5432"
	@echo "  • Semantic Cache: http://localhost:8080"
	@echo "  • Jaeger UI:     http://localhost:16686"
	@echo "  • Prometheus:    http://localhost:9090"
	@echo ""
	@echo "$(YELLOW)CLI Tool:$(RESET)"
	@echo "  • ./tools/bin/semantic-cache-tool"
	@echo ""
	@echo "$(CYAN)The application is running in Docker. Check status with 'make status'$(RESET)"

stop: ## Stop full stack (removes containers + binaries)
	@echo "$(RED)🛑 Stopping full stack...$(RESET)"
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose down
	@echo "$(YELLOW)Cleaning CLI tools binaries...$(RESET)"
	@rm -rf tools/bin/
	@echo "$(GREEN)✅ Stack stopped and tools cleaned$(RESET)"

status: ## Check stack status
	@echo "$(CYAN)📊 Stack Status$(RESET)"
	@echo ""
	@echo "$(YELLOW)Docker Services:$(RESET)"
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose ps
	@echo ""
	@echo "$(YELLOW)Service Health:$(RESET)"
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose ps | grep -E "(healthy|unhealthy|starting)" || true
	@echo ""
	@echo "$(YELLOW)CLI Tool:$(RESET)"
	@if [ -f tools/bin/semantic-cache-tool ]; then \
		echo "  ✅ tools/bin/semantic-cache-tool ready"; \
	else \
		echo "  ❌ tools/bin/semantic-cache-tool not built"; \
	fi
	@echo ""
	@echo "$(YELLOW)Application Logs (last 5 lines):$(RESET)"
	@cd $(DOCKER_COMPOSE_DIR) && docker-compose logs --tail=5 semantic-cache 2>/dev/null || echo "  Application not running"


.PHONY: help build run clean dev fmt lint check \
        test cover bench watch \
        docker up down logs logs-all restart \
        migrate reset \
        deps update verify \
        all ci deploy \
        start stop status