# Development Guide

This guide covers the development workflow, testing procedures, and best practices for contributing to the Semantic Cache Infrastructure project.

## Development Workflow

### Prerequisites

Ensure you have the following tools installed:
- Go 1.21+
- Docker
- k3d v5.x
- kubectl
- Task (build tool)

### Standard Development Cycle

```bash
# 1. Start development environment
task dev-up

# 2. Make changes to code
# Edit files as needed

# 3. Run quality checks
task check

# 4. Build and test
task build test

# 5. Deploy changes
task dev

# 6. View logs
task dev-logs

# 7. Clean up when done
task dev-down
```

### Git Workflow

Follow conventional commits for clear history:

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
task check

# Commit with conventional commits
git commit -m "feat: add new functionality"
git commit -m "fix: resolve issue with..."
git commit -m "docs: update README"

# Push and create PR
git push origin feature/my-feature
```

## Testing

### Running Tests

#### Using Task (Recommended)

```bash
# Run all tests (filters macOS warnings automatically)
task test

# Run tests quietly
task test-quiet

# Generate coverage report
task test-coverage
```

#### Using Test Script

The test script provides additional options:

```bash
# Basic test run
./scripts/test.sh

# With coverage
./scripts/test.sh --coverage

# Quiet mode with coverage
./scripts/test.sh --quiet --coverage

# Detailed per-package coverage
./scripts/test.sh --detailed
```

#### Direct Go Commands

```bash
# Basic test
go test ./...

# With race detection and coverage
go test -v -race -coverprofile=coverage.out ./...

# Filter macOS warnings manually
go test -v ./... 2>&1 | grep -v "ld: warning"
```

### Coverage Reports

```bash
# View HTML coverage report
task test-coverage
open coverage.html

# View coverage summary
go tool cover -func=coverage.out

# Check specific package coverage
go tool cover -func=coverage.out | grep "pkg/kubernetes"
```

### Writing Tests

Follow these guidelines when writing tests:

1. Use table-driven tests for comprehensive coverage
2. Mock external dependencies (Docker, Kubernetes clients)
3. Ensure tests are deterministic
4. Clean up resources after tests

Example test structure:

```go
func TestClusterManager_CreateCluster(t *testing.T) {
    tests := []struct {
        name    string
        setup   func(*mocks.MockClient)
        wantErr bool
    }{
        {
            name: "successful creation",
            setup: func(m *mocks.MockClient) {
                m.On("CreateCluster", mock.Anything).Return(nil)
            },
            wantErr: false,
        },
        {
            name: "creation fails",
            setup: func(m *mocks.MockClient) {
                m.On("CreateCluster", mock.Anything).Return(errors.New("failed"))
            },
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

## Build System (Task)

### Installation

```bash
# macOS
brew install go-task/tap/go-task

# Linux
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin

# Windows
scoop install task
```

### Common Task Commands

| Command | Description |
|---------|-------------|
| `task` | List all available tasks |
| `task build` | Build the binary |
| `task test` | Run tests |
| `task test-coverage` | Run tests with coverage |
| `task lint` | Run golangci-lint |
| `task fmt` | Format code with gofmt |
| `task check` | Run all quality checks |
| `task ci` | Run full CI pipeline |
| `task dev-up` | Start development cluster |
| `task dev-down` | Stop development cluster |
| `task clean` | Clean build artifacts |

### Build Examples

```bash
# Standard build
task build

# Build for all platforms
task build-all

# Run CI pipeline (deps + check + build)
task ci

# Development workflow
task dev-up      # Start cluster
task dev         # Deploy app
task dev-logs    # View logs
task dev-down    # Clean up
```

## Code Quality

### Pre-commit Checks

Always run these before committing:

```bash
# Run all checks
task check

# Individual checks
task fmt         # Format code
task fmt-check   # Check formatting
task lint        # Run linter
task vet         # Run go vet
task test        # Run tests
```

### Code Standards

1. Follow Go conventions and idioms
2. Keep functions small and focused
3. Document exported functions and types
4. Handle errors explicitly
5. Use meaningful variable names
6. Avoid global state

## Debugging

### Enable Debug Mode

```bash
# Set debug environment variables
export DEBUG=1
export SC_DEPLOY_DEBUG=true

# Run with verbose flag
./bin/iaac -v cluster up
```

### Debug Commands

```bash
# Analyze deployment
iaac debug analyze full

# View secrets
iaac debug secrets view

# Test API endpoints
iaac debug test detailed
```

### Common Issues

#### macOS Linker Warnings

Warnings like `ld: warning: -bind_at_load is deprecated` are harmless and automatically filtered by our test tools.

#### Test Timeouts

Increase timeout for long-running tests:
```bash
go test -timeout 30m ./...
```

#### Build Failures

```bash
# Clean and rebuild
task clean
task deps
task build

# Verify dependencies
go mod tidy
go mod verify
```

#### Deployment Issues

```bash
# Check cluster status
iaac cluster ps

# View pod logs
kubectl logs -n app -l app=semantic-cache --tail=100

# Debug deployment
iaac debug analyze full
```

## Performance

### Running Benchmarks

```bash
# Run all benchmarks
go test -bench=. ./...

# Run specific benchmark
go test -bench=BenchmarkClusterCreate ./pkg/k3d

# With memory profiling
go test -bench=. -benchmem ./...
```

### Profiling

```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=.
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof -bench=.
go tool pprof mem.prof
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass
5. Run `task check` for code quality
6. Submit a pull request

### Pull Request Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted (`task fmt`)
- [ ] Linter passes (`task lint`)
- [ ] Tests pass (`task test`)
- [ ] Conventional commit messages used