# Blueprint Decoupling Guide

## Overview

This guide explains how to decouple the iaac/infra code from the blueprint directory structure using a configuration-driven approach.

## Problem

The current implementation has tight coupling between:
- Hardcoded scenario names (minimal, development, service-mesh, etc.)
- Fixed directory structure (blueprint/scenarios/, blueprint/infra/, etc.)
- Hardcoded module and overlay names
- Assumptions about paths and namespaces

This makes it difficult to:
- Use different directory structures
- Add new scenarios without code changes
- Customize deployments for different environments
- Share blueprints across projects

## Solution: Configuration-Driven Blueprints

### 1. Blueprint Configuration File

Create a `.blueprint.yaml` file in your blueprint directory:

```yaml
version: "1.0"

metadata:
  name: "my-app-blueprint"
  description: "Custom blueprint for my application"
  version: "1.0.0"

paths:
  base: "."
  scenarios: "deployments"      # Can use any directory name
  infrastructure: "platform"    # Not locked to "infra"
  application: "services"       # Not locked to "app"
  modules: "components"         # Custom module directory
  overlays: "environments"      # Custom overlay directory

scenarios:
  basic:  # Custom scenario name
    name: "Basic Setup"
    description: "Minimal required components"
    components: ["database", "cache"]
    namespaces: ["backend", "frontend"]
    validation:
      required_components: ["postgresql", "redis"]
      tests: ["health-check"]

  production:  # Another custom scenario
    name: "Production"
    description: "Full production setup"
    components: ["database", "cache", "monitoring", "security"]
    namespaces: ["backend", "frontend", "observability"]

modules:
  database:
    name: "Database Stack"
    path: "db"  # Can override default path
    
  monitoring:
    name: "Observability"
    dependencies: ["metrics", "logs", "traces"]

overlays:
  staging:
    name: "Staging Environment"
    parameters:
      replicas: "2"
      
  production:
    name: "Production Environment"
    parameters:
      replicas: "3"
      enable_ha: "true"
```

### 2. Using Blueprint Manager

The blueprint manager automatically discovers and loads the configuration:

```go
import "github.com/raja-aiml/sematic-cache/deploy/local/pkg/blueprint"

// Auto-discover blueprint config
manager, err := blueprint.GetGlobalManager()
if err != nil {
    // Handle missing config - fall back to defaults
}

// Get scenario path (respects custom paths)
scenarioPath, err := manager.GetScenarioPath("basic")

// List available scenarios dynamically
scenarios := manager.ListAvailableScenarios()
for name, description := range scenarios {
    fmt.Printf("%s: %s\n", name, description)
}

// Get validation requirements
components, tests, err := manager.GetScenarioValidation("production")
```

### 3. Command Line Usage

The refactored commands work with any blueprint structure:

```bash
# Use default blueprint config discovery
iaac cluster up --scenario basic

# Specify custom blueprint config
iaac cluster up --blueprint-config ./custom-blueprint.yaml --scenario production

# List available scenarios from config
iaac cluster list-scenarios

# Validate against blueprint config
iaac validate blueprint
```

### 4. Migration Steps

To migrate existing blueprints:

1. **Create Blueprint Config**
   ```bash
   # In your blueprint directory
   cat > .blueprint.yaml << EOF
   version: "1.0"
   metadata:
     name: "semantic-cache-blueprint"
   # ... (see example above)
   EOF
   ```

2. **Update Command Usage**
   ```bash
   # Old (hardcoded)
   iaac cluster up --scenario minimal
   
   # New (config-driven)
   iaac cluster up  # Discovers .blueprint.yaml
   ```

3. **Custom Scenarios**
   ```yaml
   # Add your own scenarios without changing code
   scenarios:
     my-custom-setup:
       name: "My Custom Setup"
       components: ["my-db", "my-cache", "my-service"]
   ```

4. **Different Directory Structure**
   ```yaml
   paths:
     scenarios: "environments"
     infrastructure: "ops"
     application: "workloads"
   ```

### 5. Benefits

1. **Flexibility**
   - Use any directory structure
   - Define custom scenarios
   - Add new components without code changes

2. **Portability**
   - Share blueprints across projects
   - Version control blueprint definitions
   - Environment-specific configurations

3. **Maintainability**
   - Clear separation of configuration and code
   - Self-documenting blueprints
   - Easier to understand and modify

4. **Extensibility**
   - Add custom validation rules
   - Define new module types
   - Support multiple blueprint versions

### 6. Backward Compatibility

The system maintains backward compatibility:
- If no `.blueprint.yaml` is found, it falls back to hardcoded defaults
- Existing commands continue to work
- Migration can be gradual

### 7. Example: Custom Blueprint

Here's a complete example for a microservices blueprint:

```yaml
version: "1.0"

metadata:
  name: "microservices-platform"
  description: "Platform for microservices deployment"
  author: "Platform Team"

paths:
  base: "."
  scenarios: "environments"
  infrastructure: "platform"
  application: "services"
  modules: "addons"

scenarios:
  dev:
    name: "Development"
    description: "Local development with hot reload"
    components: ["core", "dev-tools"]
    namespaces: ["platform", "services", "tools"]
    
  staging:
    name: "Staging"
    description: "Staging environment matching production"
    components: ["core", "monitoring", "security"]
    namespaces: ["platform", "services", "monitoring"]
    
  production:
    name: "Production"
    description: "Production-grade deployment"
    components: ["core", "monitoring", "security", "backup"]
    namespaces: ["platform", "services", "monitoring", "security"]

modules:
  core:
    name: "Core Platform"
    description: "Essential platform components"
    dependencies: ["networking", "storage", "databases"]
    
  monitoring:
    name: "Observability Stack"
    description: "Metrics, logs, and traces"
    path: "observability"
    
  security:
    name: "Security Components"
    description: "Security and compliance tools"
    dependencies: ["policies", "scanning", "secrets"]

overlays:
  aws:
    name: "AWS Deployment"
    description: "Optimized for AWS EKS"
    parameters:
      storage_class: "gp3"
      load_balancer: "nlb"
      
  gcp:
    name: "GCP Deployment"
    description: "Optimized for GKE"
    parameters:
      storage_class: "pd-ssd"
      load_balancer: "gce"

validation:
  required_directories:
    - "environments"
    - "platform"
    - "services"
  required_files:
    - "kustomization.yaml"
    - "environments/dev/kustomization.yaml"
```

### 8. Testing Your Blueprint

Validate your blueprint configuration:

```bash
# Validate structure
iaac validate blueprint --blueprint-config .blueprint.yaml

# Test scenario deployment
iaac cluster up --scenario dev --dry-run

# List what would be deployed
iaac manifest generate --scenario staging
```

## Conclusion

The configuration-driven approach decouples the code from the blueprint structure, making it:
- More flexible for different use cases
- Easier to maintain and extend
- Portable across projects
- Self-documenting

This approach allows teams to define their own conventions while leveraging the power of the iaac tooling.