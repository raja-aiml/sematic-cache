# DevOps Tools

This directory contains development and operations tools for the semantic-cache project. These tools are kept separate from the main application to maintain a clean separation of concerns.

## Location

The tools are located under `devops/tools/` to keep all DevOps-related items together:
- Configuration files (Taskfile.*.yaml) 
- Scripts
- Development tools

## Structure

```
devops/
├── tools/              # This directory
│   ├── cmd/
│   │   └── devops/     # DevOps CLI tool
│   ├── internal/
│   │   ├── devops/     # DevOps command implementations
│   │   │   └── cmd/    # Cobra commands
│   │   └── taskdoc/    # Taskfile documentation generator
│   ├── go.mod          # Separate module for tools
│   └── README.md
├── Taskfile.*.yaml     # Common task definitions
└── scripts/            # Shell scripts
```

## Why Separate Module?

1. **Clean Separation**: Development tools don't mix with production code
2. **Independent Dependencies**: Tools can use different versions of dependencies
3. **Faster Builds**: Main application builds don't include tool dependencies
4. **Clear Boundaries**: Easy to identify what's a tool vs application code

## Available Tools

### DevOps CLI

A comprehensive CLI tool for development operations:

```bash
# Build the tool
task build:devops

# Show help
./bin/devops --help

# Generate Taskfile documentation
./bin/devops taskdoc

# Validate Taskfiles
./bin/devops validate

# Show version
./bin/devops version
```

#### Subcommands

- **taskdoc**: Generate documentation for all Taskfiles
  - Supports markdown and JSON output
  - Creates dependency graphs
  - Shows task hierarchies

- **validate**: Validate Taskfile syntax
  - Checks YAML syntax
  - Verifies required fields
  - Reports errors with file locations

- **version**: Display version information
  - Git commit hash
  - Build timestamp
  - Go version

## Development

### Adding New Tools

1. Create a new command in `cmd/`:
   ```go
   // cmd/newtool/main.go
   package main
   
   func main() {
       // Tool implementation
   }
   ```

2. Add build task to root Taskfile.yaml:
   ```yaml
   build:newtool:
     desc: Build the new tool
     cmds:
       - cd tools && go build -o ../bin/newtool ./cmd/newtool
   ```

### Adding DevOps Subcommands

1. Create command file in `internal/devops/cmd/`:
   ```go
   // internal/devops/cmd/newcmd.go
   package cmd
   
   var newCmd = &cobra.Command{
       Use:   "new",
       Short: "New command description",
       RunE:  runNew,
   }
   ```

2. Register in `internal/devops/cmd/root.go`:
   ```go
   func init() {
       rootCmd.AddCommand(newCmd)
   }
   ```

## Testing

Run tests for all tools:
```bash
cd tools && go test ./...
```

Run tests with coverage:
```bash
cd tools && go test -cover ./...
```

## Dependencies

Tools use a separate `go.mod` file to avoid polluting the main application's dependencies. Current tool dependencies:

- **cobra**: CLI framework
- **yaml.v3**: YAML parsing for taskdoc
- **testify**: Testing assertions

## Future Tools

Potential tools to add:

- **migrate**: Database migration tool
- **seed**: Database seeding tool
- **bench**: Performance benchmarking tool
- **lint**: Custom linting rules
- **generate**: Code generation utilities
- **release**: Release automation