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