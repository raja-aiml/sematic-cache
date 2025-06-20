# Taskfile Structure Documentation

This document outlines the reorganized Taskfile structure for the Semantic Cache project.

## Directory Structure

```
semantic-cache/
├── Taskfile.yaml                    # Root orchestration
├── devops/
│   ├── Taskfile.build.common.yaml  # Common build tasks
│   ├── Taskfile.deploy.common.yaml # Common deploy tasks
│   ├── Taskfile.example.yaml       # Usage examples
│   ├── README.md                   # DevOps system docs
│   └── STRUCTURE.md                # This file
└── iaac/
    ├── infra/
    │   └── Taskfile.yaml           # Local deployment tool
    └── blueprint/archive/
        └── Taskfile.yaml           # Deploy orchestration
```

## Task Organization

### 1. DevOps Build Tasks (`devops/Taskfile.build.common.yaml`)
**Purpose**: Common Go build, test, and code quality tasks

**Contains**:
- Reusable build tasks
- Tasks for: build, test, format, lint, dependencies, Docker, CI/CD

**Used by**: Root Taskfile, iaac/infra Taskfile

### 2. DevOps Deploy Tasks (`devops/Taskfile.deploy.common.yaml`)
**Purpose**: Kubernetes deployment and cluster management tasks

**Contains**:
- Reusable deployment tasks  
- Tasks for: cluster management, monitoring, testing, debugging

**Used by**: Root Taskfile, iaac/blueprint/archive Taskfile

### 3. Root Taskfile (`./Taskfile.yaml`)
**Purpose**: Main orchestration layer

**Includes**:
- `build:*` tasks from `devops/Taskfile.build.common.yaml`
- `deploy:*` tasks from `devops/Taskfile.deploy.common.yaml`

**Provides**: High-level commands for the entire project

### 4. IaaC Infra Taskfile (`iaac/infra/Taskfile.yaml`)
**Purpose**: Local deployment tool specific tasks

**Includes**:
- `build:*` tasks from `../../devops/Taskfile.build.common.yaml`

**Provides**: Go application build/test tasks for the deployment tool

## Task Namespacing

### Build Tasks (prefix: `build:`)
```bash
task build:build          # Build binary
task build:build:all      # Multi-platform build
task build:test           # Run tests
task build:test:coverage  # Coverage tests
task build:fmt            # Format code
task build:lint           # Lint code
task build:clean          # Clean artifacts
task build:docker:build   # Build Docker image
task build:ci             # CI pipeline
task build:release        # Create release
```

### Deploy Tasks (prefix: `deploy:`)
```bash
task deploy:full          # Complete workflow
task deploy:quick         # Quick workflow
task deploy:setup         # Setup cluster
task deploy:deploy        # Deploy apps
task deploy:test          # Run tests
task deploy:status        # Check status
task deploy:logs          # View logs
task deploy:health        # Health check
task deploy:cleanup       # Cleanup resources
task deploy:debug         # Debug issues
```

## Usage Examples

### From Project Root
```bash
# Build the main application
task build

# Run all tests
task test

# Complete deployment workflow
task full

# Quick development cycle
task quick

# Check deployment status
task status
```

### From IaaC Blueprint Directory
```bash
cd iaac/blueprint/archive
task full          # Full deployment workflow
task quick         # Quick deployment
task status        # Check status
task cleanup       # Clean up resources
```

### From IaaC Infra Directory
```bash
cd iaac/infra
task build         # Build deployment tool
task test          # Test deployment tool
task ci            # Run CI pipeline
```

## Key Benefits

1. **Modular Design**: Common tasks are reusable across projects
2. **Clear Separation**: Build vs Deploy responsibilities
3. **Consistent Interface**: Same commands work everywhere
4. **Easy Maintenance**: Update common tasks in one place
5. **Namespace Clarity**: `build:*` and `deploy:*` prefixes prevent conflicts

## Migration Notes

- All `common:*` references changed to `build:*` for build tasks
- Deploy tasks use `deploy:*` prefix for deployment operations
- Local deployment tool uses `build:*` for Go build operations
- Root Taskfile orchestrates both build and deploy namespaces

## Adding New Tasks

### For Build Tasks
Add to `devops/Taskfile.build.common.yaml` and reference as `build:taskname`

### For Deploy Tasks  
Add to `devops/Taskfile.deploy.common.yaml` and reference as `deploy:taskname`

### For Project-Specific Tasks
Add to the appropriate local Taskfile (root, iaac/blueprint/archive, or iaac/infra)

This structure provides a clean, maintainable, and scalable task organization system.