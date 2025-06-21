# Semantic Cache Infrastructure - Go Implementation

A production-ready Go-based CLI tool for managing semantic cache deployments in local Kubernetes environments using k3d. This tool provides a complete infrastructure-as-code solution with Docker SDK integration, Kubernetes client-go, and comprehensive testing capabilities.

## Overview

This tool replaces traditional shell scripts with a robust Go implementation that provides:
- Type-safe configuration management
- Comprehensive error handling
- SDK-based integrations (Docker SDK, Kubernetes client-go)
- Production-ready deployment workflows
- Built-in testing and debugging capabilities

## Features

- **k3d Cluster Management**: Create and manage local Kubernetes clusters
- **Docker SDK Integration**: Build, tag, and manage container images
- **Kubernetes Native**: Direct client-go integration for resource management
- **Three-Tier Cache Testing**: Support for memory + Redis + PostgreSQL backends
- **Production Workflows**: End-to-end deployment automation
- **Comprehensive Debugging**: Built-in troubleshooting and analysis tools

## Prerequisites

- Go 1.23.8 or later
- Docker Desktop or Docker Engine
- k3d v5.x (`brew install k3d` on macOS)
- kubectl (`brew install kubectl` on macOS)
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Installation

```bash
# From the iaac/infra directory
go build -o bin/iaac .

# Or install globally
go install .

# Using Task (recommended)
task build
```

## Quick Start

```bash
# Complete deployment workflow
iaac workflow full

# Step-by-step approach
iaac cluster up        # Create cluster
iaac dev build         # Build image
iaac dev deploy        # Deploy application
iaac dev test          # Test endpoints
```

## Command Reference

### Cluster Management (`cluster`)

Manage k3d clusters with pre-configured infrastructure components.

```bash
# Create cluster with infrastructure
iaac cluster up

# Destroy cluster
iaac cluster down

# Show cluster status
iaac cluster ps

# View pod logs
iaac cluster logs -n app -l app=semantic-cache

# Verify deployment health
iaac cluster test
```

### Development Commands (`dev`)

Build, deploy, and manage the semantic cache application.

```bash
# Build Docker image
iaac dev build

# Deploy with secrets
iaac dev deploy

# Test endpoints
iaac dev test

# View logs
iaac dev logs -f --tail=100

# Check status
iaac dev status

# Remove deployment
iaac dev remove
```

### Workflow Orchestration (`workflow`)

Production-ready deployment workflows with automated testing.

```bash
# Full deployment (setup → build → deploy → test)
iaac workflow full

# Individual phases
iaac workflow setup
iaac workflow build
iaac workflow deploy
iaac workflow test

# Cleanup and reset
iaac workflow cleanup
iaac workflow reset
```

### Composite Backend Testing (`composite-test`)

Test the three-tier cache architecture with cluster services.

```bash
# Run composite backend test
iaac composite-test
```

This command:
- Sets up port forwarding for PostgreSQL and Redis
- Initializes the database with pgvector
- Creates a composite configuration
- Runs the server with all three cache tiers
- Tests the endpoints
- Cleans up automatically

### Debug Commands (`debug`)

Comprehensive debugging and troubleshooting tools.

```bash
# Secret management
iaac debug secrets create
iaac debug secrets view
iaac debug secrets update

# Deployment analysis
iaac debug analyze full
iaac debug analyze quick

# API testing
iaac debug test quick
iaac debug test detailed
```

## Architecture

```
iaac/infra/
├── cmd/                    # Command implementations
│   ├── cluster.go         # Cluster management
│   ├── composite.go       # Composite backend testing
│   ├── debug.go           # Debugging utilities
│   ├── dev.go            # Development workflow
│   └── workflow.go       # Production workflows
├── pkg/                   # Core packages
│   ├── k3d/              # k3d cluster operations
│   ├── kubernetes/       # Kubernetes client operations
│   ├── docker/          # Docker SDK integration
│   ├── secrets/         # Secret management
│   ├── testing/         # HTTP endpoint testing
│   ├── database/        # PostgreSQL utilities
│   ├── constants/       # Configuration constants
│   └── utils/           # Common utilities
├── config/               # Configuration management
├── internal/             # Internal packages
└── docs/guide/          # Documentation

```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `DATABASE_URL`: PostgreSQL connection string (auto-configured for cluster)
- `DEBUG`: Enable debug logging
- `SC_DEPLOY_DEBUG`: Enable deployment debug mode

### Configuration File

The tool supports a YAML configuration file (`deploy-config.yaml`) for advanced settings:

```yaml
cluster:
  name: semantic-cache
  api_port: "6550"
  http_port: "8080:80"
  timeout: 5m

build:
  image_name: semantic-cache:local
  dockerfile: Dockerfile
  platform: linux/amd64

deploy:
  namespace: app
  timeout: 5m
  wait_for_ready: true
```

## Key Technologies

- **Go 1.23.8**: Modern Go with enhanced performance
- **Docker SDK**: Native Docker API integration
- **Kubernetes client-go**: Official Kubernetes Go client
- **k3d**: Lightweight Kubernetes for local development
- **Cobra**: CLI framework for command structure
- **OpenTelemetry**: Observability and tracing support

## Documentation

- [Development Guide](DEVELOPMENT_GUIDE.md) - Development workflow, testing, and debugging
- [Production Guide](PRODUCTION_GUIDE.md) - Production deployment and best practices
- [Environment Variables](ENV_VARS.md) - Complete list of configuration options

## Common Workflows

### 1. Fresh Development Setup

```bash
# Start fresh
iaac workflow reset
iaac workflow full
```

### 2. Iterative Development

```bash
# Make code changes, then:
iaac dev build
iaac dev deploy
iaac dev test
iaac dev logs -f
```

### 3. Debugging Issues

```bash
# Quick diagnosis
iaac debug analyze quick

# Full analysis
iaac debug analyze full

# Check secrets
iaac debug secrets view
```

### 4. Testing Composite Backend

```bash
# Test all three cache tiers
iaac composite-test
```

## Troubleshooting

### Common Issues

1. **Cluster creation fails**
   ```bash
   # Check Docker is running
   docker ps
   
   # Check k3d installation
   k3d version
   ```

2. **Build fails**
   ```bash
   # Check Docker daemon
   docker info
   
   # Clean build
   task clean build
   ```

3. **Deployment fails**
   ```bash
   # Check cluster status
   iaac cluster ps
   
   # Analyze deployment
   iaac debug analyze full
   ```

4. **Missing secrets**
   ```bash
   # Create from environment
   iaac debug secrets create
   ```

## Contributing

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for detailed development instructions.

## License

This project is part of the semantic cache system. See the main project for license details.

# Infrastructure as Code Tool (iaac)

A comprehensive CLI tool for managing application deployments in local Kubernetes environments with natural language support.

## Features

- **K3D Cluster Management**: Create, manage, and destroy local Kubernetes clusters
- **Application Deployment**: Deploy and manage applications with Docker and Kubernetes
- **Natural Language Interface**: Use plain English to execute infrastructure commands
- **Infrastructure Validation**: Validate configurations and deployments
- **Workflow Automation**: Automate complex deployment workflows

## NLP Agent

The iaac tool includes a powerful NLP agent that allows you to use natural language for infrastructure management.

## Features

- **Natural Language Processing**: Convert plain English queries to CLI commands
- **Interactive Mode**: Chat-like interface for continuous interaction
- **Safety Mechanisms**: Built-in protection against dangerous operations
- **Command Validation**: Ensures only valid commands are executed
- **Audit Logging**: Complete trail of all executed commands
- **Confidence Scoring**: Shows interpretation confidence levels

## Installation

1. Ensure you have an OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Build the iaac binary:
   ```bash
   go build -o iaac ./cmd/
   ```

## Usage

### Single Query Mode

Execute a single natural language query:

```bash
iaac agent "create a new cluster with 3 nodes"
```

### Interactive Mode

Start an interactive session:

```bash
iaac agent --interactive
```

### Configuration File

Use a custom configuration file:

```bash
iaac agent --config agent.yaml "show all clusters"
```

## Configuration

Create an `agent.yaml` file based on the provided `agent.yaml.example`:

```yaml
# OpenAI Configuration
openai_key: ${OPENAI_API_KEY}
openai_model: gpt-4-turbo-preview
openai_max_tokens: 1000

# Safety Configuration
enable_dangerous_commands: false
require_confirmation: true

# Command restrictions
command_whitelist:
  - cluster
  - blueprint
  - deploy

command_blacklist:
  - "cluster delete"
  - "cluster reset"

# Execution settings
command_timeout: 30s
audit_log_path: ./logs/audit.log
```

## Natural Language Examples

### Cluster Management

```
"Create a new k3d cluster called development"
→ iaac cluster create --name=development

"Show me all running clusters"
→ iaac cluster list --status=running

"Delete the test cluster"
→ iaac cluster delete --name=test

"Create a 3 node cluster with k3s version 1.28"
→ iaac cluster create --nodes=3 --k3s-version=1.28
```

### Deployment Operations

```
"Deploy nginx to the production cluster"
→ iaac deploy apply --name=nginx --cluster=production

"Apply the manifest from config/app.yaml"
→ iaac deploy apply --file=config/app.yaml

"Show deployment status for my application"
→ iaac deploy status --name=my-application
```

### Blueprint Management

```
"Validate the blueprint configuration"
→ iaac blueprint validate --file=blueprint.yaml

"List all available blueprints"
→ iaac blueprint list

"Show blueprint details for production"
→ iaac blueprint show --name=production
```

## Interactive Mode Commands

When in interactive mode, you can use these special commands:

- `help` or `?` - Show help information
- `commands` - List all available commands
- `examples` - Show example natural language queries
- `clear` - Clear the screen
- `exit` or `quit` - Exit interactive mode

## Safety Features

### Command Validation

The agent validates all commands before execution:
- Checks against whitelist/blacklist
- Validates required parameters
- Ensures command exists in registry

### Dangerous Command Protection

Commands marked as dangerous (delete, destroy, etc.) require:
- Explicit enabling in configuration
- User confirmation before execution
- Clear warning messages

### Audit Trail

All executed commands are logged with:
- Timestamp
- User
- Original query
- Executed command
- Success/failure status
- Execution duration

## Command Registry

Generate command documentation for the agent:

```bash
# Generate JSON registry
iaac docs --output commands.json

# Generate Markdown documentation
iaac docs --format markdown --output commands.md
```

The registry is used by the agent to understand available commands and their options.

## Best Practices

1. **Be Specific**: Include names, numbers, and specific details in your queries
   - Good: "Create a cluster named dev with 3 nodes"
   - Less specific: "Create a cluster"

2. **Use Natural Language**: Write queries as you would ask a colleague
   - "Can you show me all the running clusters?"
   - "I need to deploy nginx to production"

3. **Review Before Execution**: Always review the interpreted command before confirming

4. **Start with Safe Commands**: Begin with read-only operations like "list" or "show"

5. **Use Interactive Mode**: For complex tasks, interactive mode provides better feedback

## Troubleshooting

### Common Issues

1. **"OpenAI API key not set"**
   - Set the `OPENAI_API_KEY` environment variable
   - Or add it to your configuration file

2. **"Command not found in registry"**
   - Regenerate the command registry: `iaac docs`
   - Ensure the command exists in the CLI

3. **"Dangerous commands are disabled"**
   - Enable in configuration: `enable_dangerous_commands: true`
   - Or use the flag: `--enable-dangerous`

4. **Low confidence interpretations**
   - Rephrase your query with more specific details
   - Use command names directly in your query

### Debug Mode

Enable debug logging for troubleshooting:

```bash
iaac agent --debug "your query"
```

## Architecture

The agent consists of several components:

1. **NLP Engine**: OpenAI integration for natural language understanding
2. **Command Registry**: Database of available commands and options
3. **Command Parser**: Converts NLP output to executable commands
4. **Safety Validator**: Ensures commands are safe to execute
5. **Executor**: Runs commands with timeout and error handling
6. **Audit Logger**: Records all operations

## Security Considerations

1. **API Key Security**: Store OpenAI API keys securely
2. **Command Restrictions**: Use whitelists for production environments
3. **Audit Logs**: Regularly review audit logs for suspicious activity
4. **Confirmation Prompts**: Always require confirmation for destructive operations

## Advanced Usage

### Custom Command Builders

Implement custom command builders for specialized use cases:

```go
type CustomBuilder struct{}

func (b *CustomBuilder) Build(cmd *InterpretedCommand) ([]string, error) {
    // Custom command building logic
}
```

### Extending the NLP Engine

Add custom interpretation logic:

```go
type CustomNLPEngine struct {
    *OpenAINLPEngine
}

func (e *CustomNLPEngine) Interpret(ctx context.Context, query string, registry *CommandRegistry) (*InterpretedCommand, error) {
    // Custom interpretation logic
}
```

## Examples Repository

Find more examples and use cases in the examples directory:

- `examples/cluster-management.md` - Cluster operation examples
- `examples/deployment-scenarios.md` - Deployment workflows
- `examples/blueprint-usage.md` - Blueprint management examples

## Contributing

To contribute to the NLP agent:

1. Add new command patterns to the registry
2. Improve natural language understanding
3. Add safety validations
4. Enhance error messages and suggestions

## Support

For issues or questions:
- Check the troubleshooting section
- Review the examples
- Submit issues to the repository