# DevOps Go Packages

This directory contains Go packages that replace shell scripts with SDK-based implementations, following the architectural principle of preferring SDKs over CLI commands.

## 📦 Packages

### logger
Provides colored logging functionality with different log levels.
- Replaces shell echo/printf with structured logging
- Supports color output with automatic TTY detection
- Thread-safe logging operations

### osutil
Operating system and platform detection utilities.
- Replaces shell `uname` commands
- Cross-platform OS/architecture detection
- Command existence checking

### http
HTTP client with retry logic and health checking.
- Replaces shell `curl` commands
- Built-in retry mechanism
- Service health checking

### docker
Docker operations using the official Docker SDK.
- Replaces `docker` CLI commands
- Container management
- Image operations
- Log streaming

### kubernetes
Kubernetes operations using client-go SDK.
- Replaces `kubectl` CLI commands
- Deployment management
- Namespace operations
- Pod operations

## 🔧 Usage

### Logger Example
```go
import "github.com/raja-aiml/sematic-cache/devops/pkg/devops/logger"

log := logger.New()
log.Info("Starting operation...")
log.Success("Operation completed")
log.Error("Operation failed: %v", err)
```

### Docker Example
```go
import "github.com/raja-aiml/sematic-cache/devops/pkg/devops/docker"

client, err := docker.NewClient()
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// Check if Docker is running
if client.IsRunning(ctx) {
    containers, _ := client.ListContainers(ctx, true)
    for _, c := range containers {
        log.Info("Container: %s (%s)", c.Name, c.Status)
    }
}
```

### Kubernetes Example
```go
import "github.com/raja-aiml/sematic-cache/devops/pkg/devops/kubernetes"

k8s, err := kubernetes.NewClient()
if err != nil {
    log.Fatal(err)
}

// Wait for deployment
err = k8s.WaitForDeployment(ctx, "my-app", "default", 5*time.Minute)

// Scale deployment
err = k8s.ScaleDeployment(ctx, "my-app", "default", 3)
```

## 🔄 Migration from Shell Scripts

### Before (Shell)
```bash
# Check Docker
docker info >/dev/null 2>&1 || die "Docker not running"

# Wait for container
while ! docker inspect --format='{{.State.Health.Status}}' my-container | grep -q "healthy"; do
    sleep 1
done

# Check Kubernetes deployment
kubectl rollout status deployment/my-app -n default
```

### After (Go)
```go
// Check Docker
if !dockerClient.IsRunning(ctx) {
    log.Fatal("Docker not running")
}

// Wait for container
err := dockerClient.WaitForContainer(ctx, "my-container", 60*time.Second)

// Check Kubernetes deployment
err := k8sClient.WaitForDeployment(ctx, "my-app", "default", 5*time.Minute)
```

## 🛠️ Tools Installer

The `cmd/tools-installer` replaces `install-tools.sh` with a Go implementation:

```bash
# Build the installer
go build -o bin/tools-installer ./cmd/tools-installer

# Run the installer
./bin/tools-installer

# Skip confirmation
./bin/tools-installer -skip-confirmation

# Enable debug logging
./bin/tools-installer -debug
```

## 📚 Benefits

1. **Type Safety**: Compile-time checking vs runtime errors
2. **Better Error Handling**: Structured errors with context
3. **Cross-Platform**: Works on Windows without WSL
4. **Testing**: Easy to write unit tests
5. **Performance**: Concurrent operations, connection reuse
6. **No Dependencies**: No need for bash, curl, etc.

## 🧪 Testing

Run all tests:
```bash
cd devops/pkg/devops
go test ./...
```

Run with coverage:
```bash
go test -cover ./...
```

## 📝 Adding New Packages

1. Create new package directory
2. Implement functionality using official SDKs
3. Add comprehensive tests
4. Update documentation
5. Migrate dependent scripts

## 🔗 Dependencies

All dependencies use official SDKs:
- Docker: `github.com/docker/docker`
- Kubernetes: `k8s.io/client-go`
- Colors: `github.com/fatih/color`
- Testing: `github.com/stretchr/testify`