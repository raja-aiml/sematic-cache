# Go project Makefile
.PHONY: all build test test-coverage test-integration lint fmt clean run docker-build docker-run help

# Variables
BINARY_NAME=server
BINARY_PATH=bin/$(BINARY_NAME)
MAIN_PATH=cmd/server/main.go
DOCKER_IMAGE=sematic-cache
DOCKER_TAG=latest

# Default target
all: fmt lint test build

# Build the binary
build:
	@echo "Building binary..."
	@go build -o $(BINARY_PATH) $(MAIN_PATH)
	@echo "Binary built at $(BINARY_PATH)"

# Run the application
run: build
	@echo "Running application..."
	@./$(BINARY_PATH) -config config.yml

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	@go test -race -coverprofile=coverage.txt -covermode=atomic ./...
	@go tool cover -html=coverage.txt -o coverage.html
	@echo "Coverage report generated at coverage.html"

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	@go test -tags=integration -v ./test/integration/...

# Run linter
lint:
	@echo "Running linter..."
	@if command -v golangci-lint > /dev/null; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed. Running go vet instead..."; \
		go vet ./...; \
	fi

# Format code
fmt:
	@echo "Formatting code..."
	@gofmt -w .
	@go mod tidy
	@echo "Code formatted"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/ coverage.txt coverage.html
	@echo "Clean complete"

# Docker build
docker-build:
	@echo "Building Docker image..."
	@docker build -f deployments/docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# Docker run
docker-run:
	@echo "Running Docker container..."
	@docker run -p 8080:8080 -v $(PWD)/config.yml:/app/config.yml $(DOCKER_IMAGE):$(DOCKER_TAG)

# Docker compose up
compose-up:
	@echo "Starting services with docker-compose..."
	@cd deployments/local && docker-compose up -d

# Docker compose down
compose-down:
	@echo "Stopping services..."
	@cd deployments/local && docker-compose down

# Help
help:
	@echo "Available targets:"
	@echo "  make build         - Build the binary"
	@echo "  make run           - Build and run the application"
	@echo "  make test          - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-integration - Run integration tests"
	@echo "  make lint          - Run linter"
	@echo "  make fmt           - Format code and tidy dependencies"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make compose-up    - Start services with docker-compose"
	@echo "  make compose-down  - Stop docker-compose services"
	@echo "  make all           - Format, lint, test, and build"
	@echo "  make help          - Show this help message"