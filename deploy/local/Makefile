# Semantic Cache Deploy - Makefile
BINARY_NAME := semantic-cache-deploy
BINARY_DIR := bin
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)"

# Go parameters
GOCMD := go
GOBUILD := $(GOCMD) build
GOCLEAN := $(GOCMD) clean
GOTEST := $(GOCMD) test
GOGET := $(GOCMD) get
GOMOD := $(GOCMD) mod
GOFMT := gofmt
GOVET := $(GOCMD) vet
GOLINT := golangci-lint

# Ensure bin directory exists
$(shell mkdir -p $(BINARY_DIR))

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Display this help message
	@echo "Semantic Cache Deploy - Makefile Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the binary
	$(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME) .
	@echo "Binary built: $(BINARY_DIR)/$(BINARY_NAME)"

.PHONY: build-all
build-all: ## Build for all platforms
	@echo "Building for multiple platforms..."
	GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME)-darwin-amd64 .
	GOOS=darwin GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME)-darwin-arm64 .
	GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME)-linux-amd64 .
	GOOS=linux GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME)-linux-arm64 .
	GOOS=windows GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(BINARY_DIR)/$(BINARY_NAME)-windows-amd64.exe .

.PHONY: install
install: build ## Install the binary to GOPATH/bin
	cp $(BINARY_DIR)/$(BINARY_NAME) $(GOPATH)/bin/

.PHONY: clean
clean: ## Clean build artifacts
	$(GOCLEAN)
	rm -rf $(BINARY_DIR)

.PHONY: test
test: ## Run tests
	$(GOTEST) -v -race -coverprofile=coverage.out ./...

.PHONY: test-coverage
test-coverage: test ## Run tests with coverage report
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

.PHONY: benchmark
benchmark: ## Run benchmarks
	$(GOTEST) -bench=. -benchmem ./...

.PHONY: lint
lint: ## Run linter
	$(GOLINT) run ./...

.PHONY: fmt
fmt: ## Format code
	$(GOFMT) -w .

.PHONY: fmt-check
fmt-check: ## Check if code is formatted
	@if [ -n "$$($(GOFMT) -l .)" ]; then \
		echo "Go code is not formatted. Run 'make fmt'"; \
		$(GOFMT) -d .; \
		exit 1; \
	fi

.PHONY: vet
vet: ## Run go vet
	$(GOVET) ./...

.PHONY: mod-tidy
mod-tidy: ## Tidy go modules
	$(GOMOD) tidy

.PHONY: mod-download
mod-download: ## Download go modules
	$(GOMOD) download

.PHONY: deps
deps: mod-download ## Install dependencies

.PHONY: check
check: fmt-check vet lint test ## Run all checks

.PHONY: ci
ci: deps check build ## Run CI pipeline

.PHONY: run
run: build ## Build and run
	./$(BINARY_DIR)/$(BINARY_NAME)

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t $(BINARY_NAME):$(VERSION) .

.PHONY: release
release: clean test build-all ## Create release artifacts
	@echo "Creating release $(VERSION)..."
	mkdir -p releases/$(VERSION)
	cp $(BINARY_DIR)/* releases/$(VERSION)/
	tar -czf releases/$(BINARY_NAME)-$(VERSION).tar.gz -C releases/$(VERSION) .
	@echo "Release created: releases/$(BINARY_NAME)-$(VERSION).tar.gz"

# Development shortcuts
.PHONY: dev
dev: ## Run development workflow
	./$(BINARY_DIR)/$(BINARY_NAME) workflow full

.PHONY: dev-up
dev-up: build ## Start development cluster
	./$(BINARY_DIR)/$(BINARY_NAME) cluster up

.PHONY: dev-down
dev-down: ## Stop development cluster
	./$(BINARY_DIR)/$(BINARY_NAME) cluster down

.PHONY: dev-logs
dev-logs: ## Show application logs
	./$(BINARY_DIR)/$(BINARY_NAME) dev logs -f

.PHONY: dev-test
dev-test: ## Run application tests
	./$(BINARY_DIR)/$(BINARY_NAME) dev test