# Semantic Cache - Simple Makefile
.PHONY: help build run test clean fmt lint docker-build docker-up docker-down

# Variables
BINARY_NAME=semantic-cache
BINARY_PATH=bin/$(BINARY_NAME)
DOCKER_IMAGE=semantic-cache:latest

# Default target
help:
	@echo "Semantic Cache - Available commands:"
	@echo ""
	@echo "  make build    - Build the binary"
	@echo "  make run      - Run the application"
	@echo "  make test     - Run tests"
	@echo "  make fmt      - Format code"
	@echo "  make lint     - Run linters"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start with docker-compose"
	@echo "  make docker-down   - Stop docker-compose"

# Build the binary
build:
	@echo "Building $(BINARY_NAME)..."
	@go build -o $(BINARY_PATH) .

# Run the application
run: build
	@echo "Running $(BINARY_NAME)..."
	@./$(BINARY_PATH)

# Run tests
test:
	@echo "Running tests..."
	@go test -v -cover ./...

# Format code
fmt:
	@echo "Formatting code..."
	@gofmt -w .
	@go mod tidy

# Run linters
lint:
	@echo "Running go vet..."
	@go vet ./...

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/
	@go clean -cache

# Docker commands
docker-build:
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_IMAGE) -f deployments/docker/Dockerfile .

docker-up:
	@echo "Starting services..."
	@cd deployments/local && docker-compose up -d

docker-down:
	@echo "Stopping services..."
	@cd deployments/local && docker-compose down

# Quick commands for development
.PHONY: dev test-watch

# Development mode - run with hot reload (requires air)
dev:
	@which air > /dev/null || go install github.com/cosmtrek/air@latest
	@air

# Watch tests (requires gotestsum)
test-watch:
	@which gotestsum > /dev/null || go install gotest.tools/gotestsum@latest
	@gotestsum --watch