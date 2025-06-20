# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the semantic cache repository. It emphasizes architectural principles, code quality standards, and development best practices.

## Quick Start Summary

### Essential Commands
```bash
gofmt -w .              # Format all code (MANDATORY before commit)
go test ./...           # Run all tests (MANDATORY before commit)
go vet ./...            # Static analysis (MANDATORY before commit)
go run cmd/server/main.go -config config.yml  # Run server
```

### Git Commit Shortcut
When you need to quickly stage all changes and commit with an auto-generated message, use this instruction:
```
"git add, generate git commit message and commit"
```
This will:
1. Check git status to understand changes
2. Stage all modified and new files (limited to current directory when in deploy/local)
3. Generate a descriptive commit message following conventional commits format
4. Create the commit with co-authorship attribution

Note: When working in deploy/local, only changes within that directory will be staged and committed.

### Key Principles to Follow
1. **KISS**: Keep implementations simple and readable
2. **DRY**: Don't repeat code - extract common functionality
3. **SOLID**: Follow all five SOLID principles
4. **Testing**: Minimum 80% coverage with table-driven tests
5. **Error Handling**: Never ignore errors, always wrap with context
6. **Interfaces**: Depend on abstractions, not concrete types
7. **Context**: Always pass context.Context as first parameter
8. **Formatting**: Code MUST pass gofmt before committing

### Project Structure
- `core/`: Core functionality (cache, agents, orchestrator)
- `storage/`: Storage backend implementations
- `server/`: HTTP API server
- `openai/`: OpenAI integration
- `config/`: Configuration management
- `cmd/`: Application entry points

## Architectural Principles

### KISS (Keep It Simple, Stupid)
- Prefer simple, readable solutions over clever ones
- Each function should do one thing well
- Avoid premature optimization
- Clear naming over comments
- Example: Use standard library when possible instead of external dependencies

### DRY (Don't Repeat Yourself)
- Extract common functionality into reusable functions
- Use interfaces to share behavior across types
- Centralize configuration and constants
- Create shared utilities for common operations
- Example: Single embedding generation function used by all storage backends

### SOLID Principles

**Single Responsibility Principle (SRP)**
- Each struct/type should have one reason to change
- Separate concerns into different packages
- Example: Storage backends handle persistence, not similarity calculations

**Open/Closed Principle (OCP)**
- Open for extension, closed for modification
- Use interfaces and composition over inheritance
- Example: Storage interface allows adding new backends without modifying core

**Liskov Substitution Principle (LSP)**
- Subtypes must be substitutable for their base types
- All storage backends must fully implement the Storage interface
- Example: Switching between Redis and PostgreSQL should require no code changes

**Interface Segregation Principle (ISP)**
- Clients should not depend on interfaces they don't use
- Keep interfaces small and focused
- Example: Separate interfaces for basic cache operations vs. advanced queries

**Dependency Inversion Principle (DIP)**
- Depend on abstractions, not concretions
- High-level modules should not depend on low-level modules
- Example: Cache depends on Storage interface, not specific implementations

## Build Output Guidelines
- Always create build output to `bin` directory that is excluded in `.gitignore`
- Use consistent output directory for compiled binaries
- Ensure build artifacts are not committed to version control

## Code Style Requirements

### Mandatory Go Formatting
**All Go source files MUST conform to `gofmt` output.** Before committing any code:

```bash
# Format all Go code (MANDATORY)
gofmt -w .

# Run go vet (MANDATORY)
go vet ./...

# Run all tests (MANDATORY)
go test ./...
```

[Rest of the content remains the same as in the original file]