# Agent Integration Guide

## Quick Start

### 1. Set Up OpenAI API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. Generate Command Registry

```bash
# Generate the command registry from your CLI
iaac docs --output commands.json
```

### 3. Test the Agent

```bash
# Test with a simple query
iaac agent "list all clusters"

# Start interactive mode
iaac agent --interactive
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Infrastructure Management
on:
  issue_comment:
    types: [created]

jobs:
  process-command:
    if: startsWith(github.event.comment.body, '/iaac')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      
      - name: Build iaac
        run: go build -o iaac ./cmd/
      
      - name: Process Command
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          QUERY="${{ github.event.comment.body }}"
          QUERY=${QUERY#/iaac }
          ./iaac agent "$QUERY"
```

### GitLab CI

```yaml
stages:
  - deploy

deploy-infrastructure:
  stage: deploy
  script:
    - export OPENAI_API_KEY=$OPENAI_API_KEY
    - go build -o iaac ./cmd/
    - ./iaac agent "$CI_COMMIT_MESSAGE"
  only:
    variables:
      - $CI_COMMIT_MESSAGE =~ /^DEPLOY:/
```

## Slack Integration

### Slack Bot Setup

```go
package main

import (
    "github.com/slack-go/slack"
    "github.com/raja-aiml/sematic-cache/deploy/local/pkg/agent"
)

func handleSlackMessage(event *slack.MessageEvent) {
    if !strings.HasPrefix(event.Text, "iaac:") {
        return
    }
    
    query := strings.TrimPrefix(event.Text, "iaac:")
    
    // Create agent
    config := &agent.Config{
        OpenAIKey: os.Getenv("OPENAI_API_KEY"),
    }
    
    cliAgent, err := agent.NewCLIAgent(config)
    if err != nil {
        respondToSlack(event, "Error: " + err.Error())
        return
    }
    
    // Process query
    result, err := cliAgent.ProcessQuery(context.Background(), query)
    if err != nil {
        respondToSlack(event, "Error: " + err.Error())
        return
    }
    
    // Send result back to Slack
    respondToSlack(event, formatResult(result))
}
```

## REST API Integration

### API Server

```go
package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "github.com/raja-aiml/sematic-cache/deploy/local/pkg/agent"
)

func main() {
    router := gin.Default()
    
    // Create agent
    config := loadConfig()
    cliAgent, _ := agent.NewCLIAgent(config)
    
    // Query endpoint
    router.POST("/query", func(c *gin.Context) {
        var req struct {
            Query string `json:"query"`
        }
        
        if err := c.ShouldBindJSON(&req); err != nil {
            c.JSON(400, gin.H{"error": err.Error()})
            return
        }
        
        result, err := cliAgent.ProcessQuery(c.Request.Context(), req.Query)
        if err != nil {
            c.JSON(500, gin.H{"error": err.Error()})
            return
        }
        
        c.JSON(200, result)
    })
    
    router.Run(":8080")
}
```

### API Client Example

```bash
# Query the agent via API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "create a new cluster with 3 nodes"}'
```

## Programmatic Usage

### Basic Integration

```go
package main

import (
    "context"
    "fmt"
    "github.com/raja-aiml/sematic-cache/deploy/local/pkg/agent"
)

func main() {
    // Configure agent
    config := &agent.Config{
        OpenAIKey: "your-api-key",
        EnableDangerousCommands: false,
        RequireConfirmation: true,
    }
    
    // Create agent
    cliAgent, err := agent.NewCLIAgent(config)
    if err != nil {
        panic(err)
    }
    
    // Process query
    query := "show all running clusters"
    result, err := cliAgent.ProcessQuery(context.Background(), query)
    if err != nil {
        panic(err)
    }
    
    if result.Success {
        fmt.Printf("Command: %s\n", result.Command)
        
        // Execute if needed
        execResult, err := cliAgent.ExecuteQuery(context.Background(), query)
        if err != nil {
            panic(err)
        }
        
        fmt.Printf("Output: %s\n", execResult.Output)
    }
}
```

### Custom Command Registry

```go
// Load custom registry
registry, err := agent.LoadRegistryFromFile("custom-commands.json")
if err != nil {
    panic(err)
}

// Create agent with custom registry
cliAgent := &agent.CLIAgent{
    config:   config,
    registry: registry,
    nlp:      nlpEngine,
    executor: executor,
}
```

## Security Best Practices

### 1. API Key Management

```go
// Use environment variables
config.OpenAIKey = os.Getenv("OPENAI_API_KEY")

// Or use a secrets manager
secret, err := secretsManager.GetSecret("openai-api-key")
config.OpenAIKey = secret
```

### 2. Command Restrictions

```yaml
# Restrict commands for production
command_whitelist:
  - "cluster list"
  - "deploy status"
  - "blueprint validate"

command_blacklist:
  - "cluster delete"
  - "database drop"
```

### 3. Audit Logging

```go
// Enable comprehensive audit logging
config.AuditLogPath = "/var/log/iaac/audit.log"

// Parse audit logs
type AuditQuery struct {
    User      string
    Query     string
    Command   string
    Success   bool
    Timestamp time.Time
}

func analyzeAuditLog(path string) ([]AuditQuery, error) {
    // Implementation
}
```

## Monitoring Integration

### Prometheus Metrics

```go
var (
    queryCounter = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "iaac_agent_queries_total",
            Help: "Total number of agent queries",
        },
        []string{"status"},
    )
    
    executionDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "iaac_agent_execution_duration_seconds",
            Help: "Duration of command execution",
        },
        []string{"command"},
    )
)
```

### Logging

```go
// Structured logging
logger.Info("Query processed",
    zap.String("query", query),
    zap.String("command", result.Command),
    zap.Float64("confidence", result.Confidence),
    zap.Bool("success", result.Success),
)
```

## Testing Integration

### Unit Tests

```go
func TestAgentIntegration(t *testing.T) {
    // Mock OpenAI responses
    mockNLP := &MockNLPEngine{
        responses: map[string]*agent.InterpretedCommand{
            "list clusters": {
                Command: "cluster",
                Subcommand: "list",
                Confidence: 0.95,
            },
        },
    }
    
    // Test query processing
    agent := &agent.CLIAgent{
        nlp: mockNLP,
        // ... other fields
    }
    
    result, err := agent.ProcessQuery(ctx, "list clusters")
    assert.NoError(t, err)
    assert.True(t, result.Success)
}
```

### Integration Tests

```go
func TestEndToEnd(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    // Requires real OpenAI API key
    config := &agent.Config{
        OpenAIKey: os.Getenv("OPENAI_API_KEY_TEST"),
    }
    
    agent, err := agent.NewCLIAgent(config)
    require.NoError(t, err)
    
    // Test real query
    result, err := agent.ProcessQuery(context.Background(), 
        "create a test cluster")
    
    assert.NoError(t, err)
    assert.Contains(t, result.Command.Command, "cluster")
}
```

## Performance Optimization

### Caching

```go
// Cache OpenAI responses
type CachedNLPEngine struct {
    *agent.OpenAINLPEngine
    cache map[string]*agent.InterpretedCommand
    mu    sync.RWMutex
}

func (e *CachedNLPEngine) Interpret(ctx context.Context, query string, registry *agent.CommandRegistry) (*agent.InterpretedCommand, error) {
    // Check cache first
    e.mu.RLock()
    if cached, ok := e.cache[query]; ok {
        e.mu.RUnlock()
        return cached, nil
    }
    e.mu.RUnlock()
    
    // Call OpenAI
    result, err := e.OpenAINLPEngine.Interpret(ctx, query, registry)
    if err != nil {
        return nil, err
    }
    
    // Cache result
    e.mu.Lock()
    e.cache[query] = result
    e.mu.Unlock()
    
    return result, nil
}
```

### Batch Processing

```go
// Process multiple queries efficiently
func BatchProcessQueries(agent *agent.CLIAgent, queries []string) ([]*agent.CommandResult, error) {
    results := make([]*agent.CommandResult, len(queries))
    errChan := make(chan error, len(queries))
    
    var wg sync.WaitGroup
    for i, query := range queries {
        wg.Add(1)
        go func(idx int, q string) {
            defer wg.Done()
            
            result, err := agent.ProcessQuery(context.Background(), q)
            if err != nil {
                errChan <- err
                return
            }
            results[idx] = result
        }(i, query)
    }
    
    wg.Wait()
    close(errChan)
    
    // Check for errors
    for err := range errChan {
        if err != nil {
            return nil, err
        }
    }
    
    return results, nil
}
```

# Agent Query Examples

## Cluster Management

### Creating Clusters

```
Query: "Create a new development cluster"
Command: iaac cluster create --name=development

Query: "I need a k3d cluster with 5 worker nodes for testing"
Command: iaac cluster create --nodes=5 --name=testing

Query: "Set up a highly available cluster called production with 3 control plane nodes"
Command: iaac cluster create --name=production --control-plane-nodes=3 --ha=true

Query: "Create a minimal cluster for CI/CD"
Command: iaac cluster create --name=ci-cd --nodes=1
```

### Listing and Viewing Clusters

```
Query: "Show me all my clusters"
Command: iaac cluster list

Query: "What clusters are currently running?"
Command: iaac cluster list --status=running

Query: "List clusters in JSON format"
Command: iaac cluster list --output=json

Query: "Give me details about the production cluster"
Command: iaac cluster get --name=production
```

### Managing Clusters

```
Query: "Stop the development cluster"
Command: iaac cluster stop --name=development

Query: "Start up the testing cluster"
Command: iaac cluster start --name=testing

Query: "Delete the old staging cluster"
Command: iaac cluster delete --name=staging

Query: "Remove all stopped clusters"
Command: iaac cluster prune --status=stopped
```

## Deployment Operations

### Deploying Applications

```
Query: "Deploy nginx to the production cluster"
Command: iaac deploy apply --name=nginx --cluster=production

Query: "Install redis in the development environment"
Command: iaac deploy apply --name=redis --cluster=development

Query: "Deploy my application from the manifest file app.yaml"
Command: iaac deploy apply --file=app.yaml

Query: "Deploy the frontend service to namespace web-apps"
Command: iaac deploy apply --name=frontend --namespace=web-apps
```

### Managing Deployments

```
Query: "Show the status of all deployments"
Command: iaac deploy list

Query: "Check if nginx is running properly"
Command: iaac deploy status --name=nginx

Query: "Scale the backend service to 5 replicas"
Command: iaac deploy scale --name=backend --replicas=5

Query: "Roll back the frontend deployment"
Command: iaac deploy rollback --name=frontend
```

## Blueprint Management

### Working with Blueprints

```
Query: "Show me all available infrastructure blueprints"
Command: iaac blueprint list

Query: "Validate my blueprint configuration file"
Command: iaac blueprint validate --file=blueprint.yaml

Query: "Check if the production blueprint is valid"
Command: iaac blueprint validate --name=production

Query: "Create a new blueprint from the template"
Command: iaac blueprint create --template=basic --name=my-blueprint
```

### Applying Blueprints

```
Query: "Apply the staging environment blueprint"
Command: iaac blueprint apply --name=staging

Query: "Deploy infrastructure using the prod-west blueprint"
Command: iaac blueprint apply --name=prod-west

Query: "Show me what would happen if I apply this blueprint"
Command: iaac blueprint apply --name=development --dry-run
```

## Complex Scenarios

### Multi-Step Operations

```
Query: "Create a new cluster and deploy nginx to it"
Commands:
1. iaac cluster create --name=new-cluster
2. iaac deploy apply --name=nginx --cluster=new-cluster

Query: "Set up a complete development environment with monitoring"
Commands:
1. iaac cluster create --name=dev-env --nodes=3
2. iaac deploy apply --name=prometheus --cluster=dev-env
3. iaac deploy apply --name=grafana --cluster=dev-env
```

### Conditional Operations

```
Query: "Update nginx if it's already deployed, otherwise install it"
Analysis: Check deployment status first
Command: iaac deploy status --name=nginx || iaac deploy apply --name=nginx

Query: "Delete the test cluster if it exists"
Analysis: Safe deletion with existence check
Command: iaac cluster get --name=test && iaac cluster delete --name=test
```

## Database Operations

### Managing Databases

```
Query: "Create a PostgreSQL database for my application"
Command: iaac database create --type=postgresql --name=app-db

Query: "Backup the production database"
Command: iaac database backup --name=production-db

Query: "Restore the database from yesterday's backup"
Command: iaac database restore --name=production-db --backup=yesterday
```

## Security Operations

### Certificate Management

```
Query: "Generate SSL certificates for my domain"
Command: iaac cert generate --domain=example.com

Query: "Renew expiring certificates"
Command: iaac cert renew --all

Query: "List all certificates"
Command: iaac cert list
```

### Secret Management

```
Query: "Create a secret for database passwords"
Command: iaac secret create --name=db-passwords

Query: "Update the API key secret"
Command: iaac secret update --name=api-keys

Query: "Show all secrets in the production namespace"
Command: iaac secret list --namespace=production
```

## Monitoring and Logging

### Viewing Logs

```
Query: "Show me logs from the web application"
Command: iaac logs --app=web-app

Query: "Get the last 100 lines of error logs"
Command: iaac logs --severity=error --tail=100

Query: "Stream live logs from the API service"
Command: iaac logs --app=api-service --follow
```

### Monitoring Resources

```
Query: "Check resource usage across all clusters"
Command: iaac monitor resources --all

Query: "Show CPU and memory usage for the production cluster"
Command: iaac monitor resources --cluster=production

Query: "Alert me if any service is down"
Command: iaac monitor health --alert
```

## Tips for Natural Language Queries

1. **Be Specific**: Include names, quantities, and specific requirements
2. **Use Action Words**: "create", "deploy", "show", "delete", "update"
3. **Include Context**: Mention cluster names, namespaces, or environments
4. **Ask Questions**: "What clusters are running?" works as well as commands
5. **Use Common Terms**: The agent understands common DevOps terminology

## Interactive Mode Examples

```
iaac> What can I do with clusters?
iaac> Create a new testing environment
iaac> Show me everything deployed to production
iaac> Help me debug the failing service
iaac> How do I scale my application?
```