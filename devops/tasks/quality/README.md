# Quality Tasks

This directory contains code quality and security task definitions.

## Available Task Files

### security.yaml
Security scanning and vulnerability detection:
- **sec:scan** - Run all security scans
- **sec:scan:go** - Scan Go code with gosec
- **sec:scan:deps** - Scan dependencies with nancy
- **sec:scan:docker** - Scan containers with trivy/grype
- **sec:scan:k8s** - Scan K8s manifests with kubesec
- **sec:scan:secrets** - Scan for exposed secrets
- **sec:audit** - Interactive security audit
- **sec:report** - Generate security report

## Usage

Include the task files you need:

```yaml
includes:
  sec: ./devops/tasks/quality/security.yaml

tasks:
  security:
    desc: Run security checks
    cmds:
      - task: sec:scan
      - task: sec:report
```

## Security Tools

### Required Tools
- **gosec** - Go security checker
- **nancy** - Dependency vulnerability scanner
- **trivy/grype** - Container scanner
- **kubesec** - Kubernetes security scanner
- **gitleaks** - Secret scanner

### Installation

```bash
# Install Go tools
task sec:install:tools

# Install container scanners
brew install trivy
# or
brew install grype

# Install kubesec
brew install kubesec
```

## Best Practices

1. **Run security scans in CI** - Catch issues early
2. **Regular dependency updates** - Keep dependencies current
3. **Container scanning** - Scan all images before deployment
4. **Secret scanning** - Prevent credential leaks
5. **Generate reports** - Track security posture over time