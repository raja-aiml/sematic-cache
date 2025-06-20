# DevOps Directory - Reusable Task Definitions

This directory contains modular DevOps task definitions that can be included in any Taskfile across the project.

## Structure

```
devops/
├── Taskfile.build.common.yaml   # Go build, test, and CI tasks
├── Taskfile.deploy.common.yaml  # Kubernetes deployment and cluster tasks
├── Taskfile.example.yaml        # Usage examples
├── README.md                    # This file
├── STRUCTURE.md                 # Architecture documentation
└── ARCHITECTURE.md              # Architecture guide
```

## Task Categories

### Build Tasks (`Taskfile.build.common.yaml`)

**Purpose**: Common Go build, test, and code quality operations

**Include as**: `build:`
```yaml
includes:
  build:
    taskfile: ./devops/Taskfile.build.common.yaml
    dir: '{{.USER_WORKING_DIR}}'
```

**Available Tasks**:
- `build:build` - Build binary for current platform
- `build:build:all` - Build for all major platforms  
- `build:build:multi` - Build for specific platforms
- `build:test` - Run tests
- `build:test:coverage` - Run tests with coverage
- `build:test:bench` - Run benchmarks
- `build:fmt` - Format code
- `build:lint` - Run linting
- `build:clean` - Clean build artifacts
- `build:deps` - Manage dependencies
- `build:docker:build` - Build Docker images
- `build:ci` - Run CI pipeline
- `build:release` - Create release artifacts

### Deploy Tasks (`Taskfile.deploy.common.yaml`)

**Purpose**: Kubernetes deployment and cluster management operations

**Include as**: `deploy:`
```yaml
includes:
  deploy:
    taskfile: ./devops/Taskfile.deploy.common.yaml
    dir: '{{.USER_WORKING_DIR}}'
```

**Available Tasks**:
- `deploy:full` - Complete production workflow
- `deploy:quick` - Quick development workflow
- `deploy:setup` - Setup cluster and infrastructure
- `deploy:deploy` - Deploy application
- `deploy:test:*` - Various test suites
- `deploy:status` - Check deployment status
- `deploy:logs` - View application logs
- `deploy:health` - Health checks
- `deploy:scale:*` - Scaling operations
- `deploy:debug:*` - Debugging tools
- `deploy:cleanup` - Resource cleanup

## Usage Examples

### In Your Project Taskfile

```yaml
version: '3'

includes:
  # Include build tasks
  build:
    taskfile: ./devops/Taskfile.build.common.yaml
    dir: '{{.USER_WORKING_DIR}}'
  
  # Include deploy tasks  
  deploy:
    taskfile: ./devops/Taskfile.deploy.common.yaml
    dir: '{{.USER_WORKING_DIR}}'

vars:
  BINARY_NAME: my-app
  CLUSTER_NAME: my-cluster

tasks:
  # Use build tasks
  build:
    desc: Build my application
    cmds:
      - task: build:build
        vars:
          BINARY_NAME: '{{.BINARY_NAME}}'

  # Use deploy tasks
  deploy:
    desc: Deploy my application
    cmds:
      - task: deploy:deploy
        vars:
          CLUSTER_NAME: '{{.CLUSTER_NAME}}'
```

### Command Line Usage

```bash
# Build operations
task build:build          # Build binary
task build:test           # Run tests
task build:fmt            # Format code
task build:ci             # Run CI pipeline

# Deploy operations  
task deploy:setup         # Setup cluster
task deploy:deploy        # Deploy application
task deploy:status        # Check status
task deploy:logs          # View logs
task deploy:cleanup       # Clean up
```

## Configuration Variables

### Build Variables
```yaml
vars:
  BINARY_NAME: my-app              # Binary name
  BINARY_DIR: ./bin               # Output directory
  MAIN: ./cmd/server              # Main package
  PACKAGES: ./...                 # Packages to test
  CGO_ENABLED: "0"               # CGO setting
  LDFLAGS: -X main.foo=bar       # Additional build flags
```

### Deploy Variables
```yaml
vars:
  CLUSTER_NAME: my-cluster        # Cluster name
  API_URL: http://localhost:8080  # API endpoint
  APP_NAMESPACE: app              # App namespace
  INFRA_NAMESPACE: infra          # Infrastructure namespace
  WORKFLOW_SCRIPT: ./deploy.sh   # Deployment script
```

## Key Benefits

1. **Separation of Concerns**: Build vs Deploy responsibilities clearly separated
2. **Reusability**: Same tasks work across all projects
3. **Consistency**: Standardized commands and behavior
4. **Modularity**: Include only what you need
5. **Maintainability**: Update tasks in one place
6. **Flexibility**: Override variables for customization

## Migration from Old Structure

### Before (Old approach)
```yaml
includes:
  common: ./build/Taskfile.common.yaml

tasks:
  build:
    cmds:
      - task: common:build
```

### After (Current approach)
```yaml
includes:
  build: ./devops/Taskfile.build.common.yaml
  deploy: ./devops/Taskfile.deploy.common.yaml

tasks:
  build:
    cmds:
      - task: build:build
  
  deploy:
    cmds:
      - task: deploy:deploy
```

## Best Practices

1. **Use Specific Includes**: Include `build:` and `deploy:` separately
2. **Override Variables**: Customize behavior through variables
3. **Compose Tasks**: Combine simple tasks into complex workflows
4. **Keep Wrappers Simple**: Local Taskfiles should be thin wrappers
5. **Document Variables**: Clearly document required variables

## Adding New Tasks

### For Build Tasks
1. Add to `Taskfile.build.common.yaml`
2. Use `build:` prefix
3. Focus on Go build/test/quality operations

### For Deploy Tasks
1. Add to `Taskfile.deploy.common.yaml`  
2. Use `deploy:` prefix
3. Focus on Kubernetes/Docker operations

### For Project-Specific Tasks
1. Add to local project Taskfile
2. Use tasks without prefix
3. Delegate to common tasks when possible

This modular approach provides clean separation, better organization, and maximum reusability across all projects.