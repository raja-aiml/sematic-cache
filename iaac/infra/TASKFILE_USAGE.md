# Taskfile Usage Guide

This project now uses [Task](https://taskfile.dev/) instead of Make. Task is a task runner that aims to be simpler and easier to use than Make.

## Installation

### macOS
```bash
brew install go-task/tap/go-task
```

### Linux
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

### Windows
```bash
scoop install task
```

## Usage

### View all available tasks
```bash
task --list-all
# or simply
task
```

### Common commands

| Make command | Task command | Description |
|--------------|--------------|-------------|
| `make build` | `task build` | Build the binary |
| `make test` | `task test` | Run tests |
| `make lint` | `task lint` | Run linter |
| `make fmt` | `task fmt` | Format code |
| `make clean` | `task clean` | Clean build artifacts |
| `make check` | `task check` | Run all checks |
| `make ci` | `task ci` | Run CI pipeline |
| `make run` | `task run` | Build and run |
| `make dev-up` | `task dev-up` | Start development cluster |
| `make dev-down` | `task dev-down` | Stop development cluster |

### Key differences from Make

1. **Dependencies**: Task uses `deps:` instead of Make's prerequisite syntax
2. **Variables**: Task uses `{{.VAR_NAME}}` instead of `$(VAR_NAME)`
3. **Help**: Task has built-in help with `task --list-all`
4. **Parallel execution**: Task can run tasks in parallel by default when using deps
5. **YAML syntax**: More readable and maintainable than Makefile syntax

### Examples

```bash
# Build the project
task build

# Run all tests with coverage
task test-coverage

# Run CI pipeline (deps + check + build)
task ci

# Build for all platforms
task build-all

# Start development environment
task dev-up

# Clean everything
task clean
```

## Benefits over Make

1. **Cross-platform**: Works consistently across macOS, Linux, and Windows
2. **No tabs**: Uses YAML syntax instead of Make's tab-sensitive format
3. **Better error messages**: More descriptive error handling
4. **Modern features**: Built-in support for running tasks in parallel, watching files, etc.
5. **Simpler syntax**: More intuitive for developers not familiar with Make