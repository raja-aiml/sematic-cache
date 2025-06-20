# Semantic Cache Deploy - Go Implementation

A Go-based CLI tool for managing semantic cache deployments in local Kubernetes environments using k3d. This tool replaces shell scripts with a robust Go implementation using the k3d SDK and Cobra CLI framework.

## Features

- **Cluster Management**: Create and manage k3d clusters with pre-configured infrastructure
- **Application Lifecycle**: Build, deploy, and manage the semantic cache application
- **Workflow Orchestration**: End-to-end deployment workflows with automated testing
- **Composite Backend Testing**: Test three-tier cache architecture (memory + Redis + PostgreSQL)
- **Debugging Tools**: Comprehensive debugging, secret management, and API testing

## Installation

```bash
# From the project root
cd deploy/local
go build -o semantic-cache-deploy .

# Or install globally
go install .
```

## Usage

### Cluster Management

```bash
# Create k3d cluster and deploy infrastructure
semantic-cache-deploy cluster up

# Destroy cluster
semantic-cache-deploy cluster down

# Show cluster status
semantic-cache-deploy cluster ps

# View pod logs
semantic-cache-deploy cluster logs -n app -l app=semantic-cache

# Verify deployment health
semantic-cache-deploy cluster test
```

### Development Workflow

```bash
# Build Docker image and import to k3d
semantic-cache-deploy dev build

# Deploy application with secrets
semantic-cache-deploy dev deploy

# Test application endpoints
semantic-cache-deploy dev test

# View application logs
semantic-cache-deploy dev logs

# Check deployment status
semantic-cache-deploy dev status

# Remove application
semantic-cache-deploy dev remove
```

### End-to-End Workflows

```bash
# Run complete workflow (setup, build, deploy, test)
semantic-cache-deploy workflow full

# Run individual workflow phases
semantic-cache-deploy workflow setup
semantic-cache-deploy workflow build
semantic-cache-deploy workflow deploy
semantic-cache-deploy workflow test

# Clean up application
semantic-cache-deploy workflow cleanup

# Reset everything
semantic-cache-deploy workflow reset
```

### Composite Backend Testing

```bash
# Test three-tier cache architecture with cluster services
semantic-cache-deploy composite-test
```

### Debugging and Management

```bash
# Secret management
semantic-cache-deploy debug secrets create
semantic-cache-deploy debug secrets view
semantic-cache-deploy debug secrets update

# Deployment analysis
semantic-cache-deploy debug analyze full
semantic-cache-deploy debug analyze quick

# API testing
semantic-cache-deploy debug test quick
semantic-cache-deploy debug test detailed
```

## Architecture

### Package Structure

```
deploy/local/
├── cmd/                    # Cobra command implementations
│   ├── cluster.go         # Cluster management commands
│   ├── dev.go            # Development commands
│   ├── workflow.go       # Workflow orchestration
│   ├── composite_test.go # Composite backend testing
│   └── debug.go          # Debugging tools
├── pkg/                   # Shared packages
│   ├── k3d/              # k3d cluster management
│   ├── kubernetes/       # Kubernetes client operations
│   ├── docker/          # Docker build and run
│   └── utils/           # Common utilities
└── main.go              # CLI entry point
```

### Key Components

1. **k3d Integration**: Direct integration with k3d v5 SDK for cluster management
2. **Kubernetes Client**: Native k8s client-go for resource management
3. **Docker Builder**: Docker image building and k3d import functionality
4. **Port Forwarding**: Automatic port forwarding for testing cluster services
5. **Configuration Management**: YAML-based config generation for testing

## Prerequisites

- Go 1.21 or later
- Docker
- k3d v5.x
- kubectl
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `DATABASE_URL`: PostgreSQL connection string (optional, defaults to cluster service)
- `DEBUG`: Enable debug logging

## Integration with Main Project

The tool is integrated with the main semantic cache project using Go workspaces:

```bash
# From project root
go work use ./deploy/local
```

This allows the tool to share code and dependencies with the main project while maintaining its own module.

## Comparison with Shell Scripts

| Feature | Shell Scripts | Go Implementation |
|---------|--------------|-------------------|
| Error Handling | Basic with `set -e` | Comprehensive error types and context |
| Dependency Management | Manual PATH checks | Go modules with versioning |
| Testing | Bash-based | Go test framework |
| Portability | Shell-dependent | Cross-platform binary |
| IDE Support | Limited | Full Go tooling support |
| Type Safety | None | Compile-time type checking |
| Concurrency | Background processes | Goroutines with proper synchronization |

## Development

### Building

```bash
# Build using Task
task build

# Build for all platforms
task build-all

# Build directly with go
go build -o bin/semantic-cache-deploy .
```

### Testing

See [TEST_GUIDE.md](TEST_GUIDE.md) for comprehensive testing documentation.

```bash
# Run tests (filters macOS warnings)
task test

# Run tests quietly
task test-quiet

# Generate coverage report
task test-coverage

# Use the test script
./scripts/test.sh --coverage
```

### Code Quality

```bash
# Format code
task fmt

# Check formatting
task fmt-check

# Run linter
task lint

# Run go vet
task vet

# Run all checks
task check
```

### Development with Task

This project uses [Task](https://taskfile.dev/) for build automation. See [TASKFILE_USAGE.md](TASKFILE_USAGE.md) for details.

```bash
# List all available tasks
task --list-all

# Run CI pipeline
task ci
```

## Future Enhancements

- [ ] Add support for multiple cluster configurations
- [ ] Implement streaming logs with follow mode
- [ ] Add Helm chart deployment option
- [ ] Support for remote cluster deployment
- [ ] Integration with CI/CD pipelines
- [ ] Prometheus metrics export
- [ ] Interactive mode with prompts
- [ ] Configuration file support