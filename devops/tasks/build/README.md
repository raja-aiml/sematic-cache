# Build Tasks

This directory contains build-related task definitions organized by technology.

## Available Task Files

### go.yaml
Go language build tasks including:
- **go:build** - Build Go binaries
- **go:test** - Run tests with various options
- **go:fmt** - Format code
- **go:lint** - Run linters
- **go:mod:** - Module management

### docker.yaml
Docker container build tasks including:
- **docker:build** - Build Docker images
- **docker:push** - Push to registry
- **docker:scan** - Security scanning
- **docker:run** - Container management

## Usage

Include the task files you need in your Taskfile.yaml:

```yaml
includes:
  go: ./devops/tasks/build/go.yaml
  docker: ./devops/tasks/build/docker.yaml

tasks:
  build:
    desc: Build application
    cmds:
      - task: go:build
      - task: docker:build
```

## Variables

### Go Tasks
- `BINARY_NAME` - Output binary name (default: directory name)
- `BINARY_DIR` - Output directory (default: bin)
- `PACKAGES` - Packages to test (default: ./...)
- `CGO_ENABLED` - CGO setting (default: 0)

### Docker Tasks  
- `IMAGE_NAME` - Docker image name (default: directory name)
- `IMAGE_TAG` - Docker image tag (default: latest)
- `REGISTRY` - Docker registry URL (optional)
- `DOCKERFILE` - Dockerfile path (default: Dockerfile)