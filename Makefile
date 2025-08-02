# Twelve-Factor App Makefile
# Factor XII: Admin processes - One-off administrative tasks

.PHONY: help build run test migrate clean docker-build docker-run docker-stop

# Default target
help:
	@echo "Twelve-Factor App - Semantic Cache"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the application binary"
	@echo "  make run            - Run the application locally"
	@echo "  make test           - Run all tests"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run with docker-compose"
	@echo "  make docker-stop    - Stop docker-compose services"
	@echo "  make migrate        - Run database migrations"
	@echo "  make seed           - Seed database with sample data"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make lint           - Run linters"
	@echo "  make fmt            - Format code"

# Factor V: Build, release, run
build:
	@echo "Building semantic-cache..."
	@go build -o bin/semantic-cache ./cmd/server/main.go

# Run locally (Factor VI: Processes)
run: build
	@echo "Running semantic-cache..."
	@./bin/semantic-cache

# Testing
test:
	@echo "Running tests..."
	@go test -v -cover ./...

# Docker operations (Factor V: Build, release, run)
docker-build:
	@echo "Building Docker image..."
	@docker build -t semantic-cache:latest .

docker-run:
	@echo "Starting services with docker-compose..."
	@docker-compose up -d

docker-stop:
	@echo "Stopping services..."
	@docker-compose down

# Factor XII: Admin processes
migrate:
	@echo "Running database migrations..."
	@go run ./cmd/migrate/main.go up

migrate-down:
	@echo "Rolling back database migrations..."
	@go run ./cmd/migrate/main.go down

seed:
	@echo "Seeding database..."
	@go run ./cmd/seed/main.go

# Database operations
db-console:
	@echo "Connecting to database..."
	@docker-compose exec postgres psql -U postgres -d semantic_cache

db-backup:
	@echo "Backing up database..."
	@docker-compose exec postgres pg_dump -U postgres semantic_cache > backup_$$(date +%Y%m%d_%H%M%S).sql

db-restore:
	@echo "Restoring database from backup..."
	@docker-compose exec -T postgres psql -U postgres semantic_cache < $(FILE)

# Code quality
lint:
	@echo "Running linters..."
	@golangci-lint run ./...
	@go vet ./...

fmt:
	@echo "Formatting code..."
	@gofmt -w .
	@go mod tidy

# Clean up
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf bin/
	@go clean -cache

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	@cp .env.example .env
	@docker-compose up -d postgres
	@sleep 5
	@$(MAKE) migrate
	@echo "Development environment ready!"

# Production deployment (Factor V: Build, release, run)
deploy:
	@echo "Deploying to production..."
	@docker build -t semantic-cache:$(VERSION) .
	@docker tag semantic-cache:$(VERSION) semantic-cache:latest
	@echo "Tagged as semantic-cache:$(VERSION)"

# Monitoring and debugging
logs:
	@docker-compose logs -f app

logs-db:
	@docker-compose logs -f postgres

stats:
	@curl -s http://localhost:$${PORT:-8080}/api/v1/stats | jq .

health:
	@curl -s http://localhost:$${PORT:-8080}/health | jq .

ready:
	@curl -s http://localhost:$${PORT:-8080}/ready | jq .