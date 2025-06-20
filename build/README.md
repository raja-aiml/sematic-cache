# Build System Documentation

This directory contains the modular build system for the Semantic Cache project using Task (task.dev).

## Overview

The build system provides:
- **Common reusable tasks** that can be included in any Taskfile
- **Standardized build, test, and release workflows**
- **Cross-platform compatibility** using Task runner

## Structure

```
build/
├── Taskfile.common.yaml  # Common task definitions
├── Taskfile.example.yaml # Example of using common tasks
└── README.md            # This file
```

## Using Common Tasks

### In Your Project's Taskfile

Include the common tasks in your `Taskfile.yaml`:

```yaml
version: '3'

includes:
  common:
    taskfile: ./build/Taskfile.common.yaml
    dir: '{{.USER_WORKING_DIR}}'

tasks:
  build:
    desc: Build my application
    cmds:
      - task: common:build
        vars:
          BINARY_NAME: my-app
```

### Available Common Tasks

#### Build Tasks
- `common:build` - Build binary for current platform
- `common:build:all` - Build for all major platforms
- `common:build:multi` - Build for specific platforms

#### Test Tasks
- `common:test` - Run tests
- `common:test:short` - Run short tests
- `common:test:coverage` - Run tests with coverage
- `common:test:bench` - Run benchmarks

#### Code Quality Tasks
- `common:fmt` - Format code
- `common:fmt:check` - Check code formatting
- `common:vet` - Run go vet
- `common:lint` - Run linting

#### Dependency Tasks
- `common:deps` - Download dependencies
- `common:deps:update` - Update dependencies
- `common:deps:tidy` - Tidy dependencies

#### Docker Tasks
- `common:docker:build` - Build Docker image
- `common:docker:push` - Push Docker image

#### Utility Tasks
- `common:clean` - Clean build artifacts
- `common:info` - Show build information
- `common:check` - Run all checks
- `common:ci` - Run CI pipeline
- `common:release` - Create release artifacts

## Variables

Common variables that can be overridden:

```yaml
vars:
  BINARY_NAME: my-app          # Binary name (default: directory name)
  BINARY_DIR: ./bin            # Output directory for binaries
  COVERAGE_DIR: ./coverage     # Coverage reports directory
  MAIN: ./cmd/server          # Main package to build
  LDFLAGS: -X main.foo=bar    # Additional ldflags
  CGO_ENABLED: "1"            # CGO setting
  PACKAGES: ./...             # Packages to test/build
```

## Examples

### Basic Build

```bash
# Build current directory
task common:build

# Build with custom name
task common:build BINARY_NAME=myapp

# Build specific main package
task common:build MAIN=./cmd/server
```

### Multi-Platform Build

```bash
# Build for all platforms
task common:build:all

# Build for specific platforms
task common:build:multi PLATFORMS="linux/amd64 darwin/arm64"
```

### Testing

```bash
# Run all tests
task common:test

# Run tests in specific directory
task common:test DIR=./pkg/cache

# Run tests with coverage
task common:test:coverage

# Run benchmarks
task common:test:bench
```

### Docker

```bash
# Build Docker image
task common:docker:build IMAGE_NAME=my-app

# Build and push
task common:docker:build IMAGE_NAME=my-app
task common:docker:push IMAGE_NAME=my-app IMAGE_TAG=v1.0.0
```

### CI/CD

```bash
# Run full CI pipeline
task common:ci

# Create release
task common:release
```

## Testing

All test functionality is now handled through the common tasks:

```bash
# Run all tests
task common:test

# Run tests with coverage
task common:test:coverage

# Run benchmarks
task common:test:bench

# Run short tests
task common:test:short
```

## Directory-Specific Builds

To build a specific directory:

```bash
# Using Task with directory override
task common:build DIR=./cmd/server

# Or change to directory first
cd ./cmd/server && task common:build
```

## Integration with Root Taskfile

The root `Taskfile.yaml` includes these common tasks:

```yaml
includes:
  common:
    taskfile: ./build/Taskfile.common.yaml
    dir: '{{.USER_WORKING_DIR}}'

tasks:
  build:
    cmds:
      - task: common:build
        vars:
          BINARY_NAME: semantic-cache
          MAIN: ./cmd/server
```

## Best Practices

1. **Use includes** instead of duplicating tasks
2. **Override variables** to customize behavior
3. **Use task composition** to build complex workflows
4. **Keep project-specific tasks** in your project's Taskfile
5. **Use common tasks** for standard operations

## Migration from Scripts

If you have existing build scripts, migrate them to Task:

```bash
# Old way
./scripts/build.sh -d ./cmd/server

# New way
task common:build DIR=./cmd/server

# Or with variables
task common:build BINARY_NAME=server MAIN=./cmd/server
```

## Troubleshooting

### Task not found
Ensure you've included the common taskfile correctly:
```yaml
includes:
  common:
    taskfile: ./build/Taskfile.common.yaml
```

### Variables not working
Check variable precedence:
1. Command line: `task build BINARY_NAME=foo`
2. Task-level vars
3. Taskfile-level vars
4. Included taskfile vars

### Directory issues
Use `{{.USER_WORKING_DIR}}` to respect the user's current directory:
```yaml
includes:
  common:
    taskfile: ./build/Taskfile.common.yaml
    dir: '{{.USER_WORKING_DIR}}'
```