# Testing Guide

This guide explains how to run tests for the Semantic Cache Deploy Local project.

## Running Tests

### Using Task (Recommended)

```bash
# Run all tests with verbose output (filters macOS warnings)
task test

# Run tests with minimal output
task test-quiet

# Run tests and generate coverage report
task test-coverage
```

### Using the Test Script

A test script is provided that automatically filters out macOS linker warnings:

```bash
# Run tests with verbose output
./scripts/test.sh

# Run tests quietly
./scripts/test.sh --quiet

# Run tests with coverage report
./scripts/test.sh --coverage

# Run tests quietly with coverage
./scripts/test.sh --quiet --coverage
```

### Direct Go Test Command

If you prefer to use `go test` directly:

```bash
# Basic test run
go test ./...

# With race detection and coverage
go test -v -race -coverprofile=coverage.out ./...

# Filter macOS warnings manually
go test -v -race -coverprofile=coverage.out ./... 2>&1 | grep -v "ld: warning"
```

## macOS Linker Warnings

On macOS, you may see warnings like:
- `ld: warning: -bind_at_load is deprecated on macOS`
- `ld: warning: '...' has malformed LC_DYSYMTAB`

These are harmless warnings from the macOS linker and don't affect test functionality. The Task commands and test script automatically filter these out.

## Coverage Reports

After running tests with coverage, you can:

1. View HTML coverage report:
   ```bash
   open coverage.html
   ```

2. View coverage summary in terminal:
   ```bash
   go tool cover -func=coverage.out
   ```

3. View coverage for specific packages:
   ```bash
   go tool cover -func=coverage.out | grep -E "pkg/docker|pkg/kubernetes"
   ```

## Test Organization

Tests are organized by package:
- Unit tests: `*_test.go` files alongside source code
- No integration tests that require external services
- Mock objects used for external dependencies

## Writing Tests

When writing new tests:
1. Use table-driven tests where appropriate
2. Mock external dependencies (Docker, Kubernetes, etc.)
3. Ensure tests are deterministic and don't depend on environment
4. Clean up any resources created during tests

## Troubleshooting

### Tests Failing Due to Missing Config

Some tests expect certain environment configurations. If tests fail:

1. Check if HOME environment variable affects tests
2. Ensure no real k8s config is interfering
3. Check Docker availability for Docker-related tests

### Timeout Issues

If tests timeout, you can increase the timeout:
```bash
go test -timeout 30m ./...
```