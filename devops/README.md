# DevOps Modular Task System

A modular, maintainable task automation system for Go projects using [Task](https://taskfile.dev).

## 🚀 Quick Start

### 1. Include modules in your Taskfile.yaml

```yaml
version: '3'

includes:
  go: ./devops/tasks/build/go.yaml
  docker: ./devops/tasks/build/docker.yaml
  k8s: ./devops/tasks/deploy/k8s.yaml
```

### 2. Use namespaced tasks

```bash
# Build
task go:build
task docker:build

# Deploy
task k8s:deploy

# Quality
task sec:scan
```

## 📁 Structure

```
devops/
├── tasks/                    # Modular task files
│   ├── build/               # Build domain
│   │   ├── go.yaml         # Go compilation, testing
│   │   └── docker.yaml     # Container builds
│   ├── deploy/              # Deployment domain
│   │   ├── k3d.yaml        # Local k8s clusters
│   │   ├── k8s.yaml        # Kubernetes operations
│   │   └── helm.yaml       # Helm charts
│   └── quality/             # Code quality
│       └── security.yaml    # Security scanning
├── scripts/                  # Shell scripts
│   ├── lib/                # Shared libraries
│   └── install-tools.sh    # Tool installation
├── templates/               # Project templates
│   └── taskfile/           # Taskfile templates
├── docs/                    # Documentation
│   └── migration.md        # Migration guide
├── tools/                   # Development tools (separate Go module)
│   ├── cmd/devops/         # DevOps CLI tool
│   └── internal/           # Tool implementations
├── Taskfile.yaml                      # DevOps meta tasks
└── Taskfile.example.yaml             # Example template
```

## 📦 Available Modules

### Build Domain

**go.yaml** - Go build and test tasks
- `go:build` - Build binary
- `go:test` - Run tests
- `go:lint` - Run linters
- `go:fmt` - Format code

**docker.yaml** - Container management
- `docker:build` - Build image
- `docker:push` - Push to registry
- `docker:run` - Run container

### Deploy Domain

**k3d.yaml** - Local Kubernetes
- `k3d:create` - Create cluster
- `k3d:delete` - Delete cluster
- `k3d:kubeconfig` - Get kubeconfig

**k8s.yaml** - Kubernetes operations
- `k8s:deploy` - Deploy application
- `k8s:scale` - Scale deployment
- `k8s:logs` - View logs
- `k8s:exec` - Execute commands

**helm.yaml** - Helm charts
- `helm:install` - Install chart
- `helm:upgrade` - Upgrade release
- `helm:rollback` - Rollback release

### Quality Domain

**security.yaml** - Security scanning
- `sec:scan` - Run all scans
- `sec:scan:go` - Scan Go code
- `sec:scan:deps` - Scan dependencies
- `sec:scan:docker` - Scan containers

## 🔧 Configuration

### Variables

Each module accepts variables:

```yaml
vars:
  # Go module
  BINARY_NAME: myapp
  
  # Docker module
  IMAGE_NAME: myapp
  REGISTRY: docker.io/myorg
  
  # K8s module
  NAMESPACE: production
  APP_NAME: myapp
```

### Composition

Combine modules for complete workflows:

```yaml
tasks:
  deploy:
    desc: Build and deploy
    cmds:
      - task: go:build
      - task: docker:build
      - task: k8s:deploy
```

## 📚 Templates

Use templates to bootstrap new projects:

```bash
# Create from template
task devops:template:create TYPE=microservice OUTPUT=Taskfile.yaml
```

Available templates:
- `basic` - Simple Go project
- `microservice` - Full microservice with K8s

## 🔄 Migration from Old Structure

The old monolithic structure has been reorganized into modular, focused task files.

### Old Structure (Deprecated)
```
devops/
├── Taskfile.build.common.yaml   # 500+ lines
└── Taskfile.deploy.common.yaml  # 400+ lines
```

### New Modular Structure
```
devops/
├── tasks/
│   ├── build/
│   │   ├── go.yaml      # ~200 lines
│   │   └── docker.yaml  # ~150 lines
│   └── deploy/
│       ├── k3d.yaml     # ~100 lines
│       ├── k8s.yaml     # ~200 lines
│       └── helm.yaml    # ~150 lines
```

### Migration Steps

```bash
# Check migration status
task devops:migrate:check

# Show migration guide
task devops:migrate:guide
```

### Migration Complete ✅

The migration to modular task structure is complete. Use the new structure:

```yaml
# Modern modular approach
includes:
  go: ./devops/tasks/build/go.yaml
  docker: ./devops/tasks/build/docker.yaml
  k8s: ./devops/tasks/deploy/k8s.yaml
  helm: ./devops/tasks/deploy/helm.yaml
```

See [migration guide](docs/migration.md) for detailed instructions.

## 🛠️ DevOps Management

```bash
# Show structure info
task devops:info

# List all modules
task devops:list:modules

# List all tasks
task devops:list:tasks

# Validate taskfiles
task devops:validate

# Install tools
task devops:install:tools
```

## 📖 Documentation

- [Migration Guide](docs/migration.md) - Migrate from old structure
- [Build Tasks](tasks/build/README.md) - Build domain documentation
- [Deploy Tasks](tasks/deploy/README.md) - Deploy domain documentation
- [Quality Tasks](tasks/quality/README.md) - Quality domain documentation

## 🔧 DevOps Tools

The `tools/` directory contains the DevOps CLI tool:

```bash
# Build the devops tool
task go:build MAIN=./devops/tools/cmd/devops BINARY_NAME=devops

# Generate Taskfile documentation
./bin/devops taskdoc

# Validate Taskfiles
./bin/devops validate
```

## 🤝 Contributing

1. Keep modules focused and single-purpose
2. Use consistent naming conventions
3. Document all tasks with descriptions
4. Provide sensible defaults for variables
5. Test modules independently

## 📝 License

Same as parent project.