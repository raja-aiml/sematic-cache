package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
	"github.com/spf13/cobra"
)

// precheckCmd represents the precheck command
var precheckCmd = &cobra.Command{
	Use:   "precheck",
	Short: "Check system prerequisites and dependencies",
	Long: `Precheck verifies that all required tools and dependencies are installed
and properly configured for the development environment.

This includes checking for:
- Go version
- Docker
- Kubernetes tools (kubectl, k3d)
- Task runner
- Other development tools`,
	Example: `  # Run all prechecks
  devops precheck

  # Run prechecks with verbose output
  devops precheck -v`,
	RunE: runPrecheck,
}

var (
	precheckVerbose bool
	precheckQuiet   bool
)

func init() {
	precheckCmd.Flags().BoolVarP(&precheckVerbose, "verbose", "v", false, "Show verbose output")
	precheckCmd.Flags().BoolVarP(&precheckQuiet, "quiet", "q", false, "Only show errors")
}

type tool struct {
	name        string
	command     string
	versionCmd  string
	minVersion  string
	required    bool
	installHint string
}

func runPrecheck(cmd *cobra.Command, args []string) error {
	log := logger.New()

	if !precheckQuiet {
		log.Info("Running system prechecks...")
		log.Info("========================")
		fmt.Println()
	}

	// Define tools to check
	tools := []tool{
		{
			name:        "Go",
			command:     "go",
			versionCmd:  "version",
			minVersion:  "1.21",
			required:    true,
			installHint: "Visit https://golang.org/dl/",
		},
		{
			name:        "Docker",
			command:     "docker",
			versionCmd:  "version --format '{{.Server.Version}}'",
			required:    true,
			installHint: "Visit https://docs.docker.com/get-docker/",
		},
		{
			name:        "kubectl",
			command:     "kubectl",
			versionCmd:  "version --client",
			required:    false,
			installHint: "Run: devops install kubectl",
		},
		{
			name:        "k3d",
			command:     "k3d",
			versionCmd:  "version",
			required:    false,
			installHint: "Run: devops install k3d",
		},
		{
			name:        "Task",
			command:     "task",
			versionCmd:  "--version",
			required:    true,
			installHint: "Run: devops install task",
		},
		{
			name:        "Helm",
			command:     "helm",
			versionCmd:  "version --short",
			required:    false,
			installHint: "Run: devops install helm",
		},
		{
			name:        "golangci-lint",
			command:     "golangci-lint",
			versionCmd:  "version",
			required:    false,
			installHint: "Run: devops install golangci-lint",
		},
	}

	// Check system info
	if !precheckQuiet {
		log.Info("System Information:")
		fmt.Printf("  OS: %s\n", runtime.GOOS)
		fmt.Printf("  Architecture: %s\n", runtime.GOARCH)
		fmt.Printf("  CPUs: %d\n", runtime.NumCPU())
		fmt.Println()
	}

	// Check each tool
	failedChecks := 0
	missingRequired := []string{}

	for _, t := range tools {
		if err := checkTool(t, log); err != nil {
			if t.required {
				missingRequired = append(missingRequired, t.name)
			}
			failedChecks++
		}
	}

	// Check Go workspace
	if !precheckQuiet {
		fmt.Println()
		log.Info("Go Workspace:")
	}
	checkGoWorkspace(log)

	// Check Docker daemon
	if !precheckQuiet {
		fmt.Println()
		log.Info("Docker Status:")
	}
	checkDockerDaemon(log)

	// Summary
	if !precheckQuiet {
		fmt.Println()
		if failedChecks == 0 {
			log.Success("All prechecks passed! Your environment is ready.")
		} else {
			log.Warn("Some prechecks failed. %d tools are missing or outdated.", failedChecks)
			if len(missingRequired) > 0 {
				log.Error("Missing required tools: %v", missingRequired)
				log.Info("Install missing tools with: devops install")
			}
		}
	}

	if len(missingRequired) > 0 {
		return fmt.Errorf("missing required tools: %v", missingRequired)
	}

	return nil
}

func checkTool(t tool, log *logger.Logger) error {
	// Check if tool exists
	path, err := exec.LookPath(t.command)
	if err != nil {
		if t.required {
			log.Error("❌ %s: NOT FOUND (required)", t.name)
		} else {
			log.Warn("⚠️  %s: NOT FOUND (optional)", t.name)
		}
		if precheckVerbose && t.installHint != "" {
			fmt.Printf("     Install: %s\n", t.installHint)
		}
		return err
	}

	// Get version
	var versionOutput string
	if t.versionCmd != "" {
		cmd := exec.Command("sh", "-c", fmt.Sprintf("%s %s", t.command, t.versionCmd))
		output, err := cmd.CombinedOutput()
		if err != nil {
			log.Warn("⚠️  %s: Found at %s but couldn't get version", t.name, path)
			if precheckVerbose {
				fmt.Printf("     Error: %v\n", err)
				fmt.Printf("     Output: %s\n", string(output))
			}
			return nil
		}
		versionOutput = string(output)
	}

	if precheckVerbose {
		log.Success("✅ %s: %s", t.name, versionOutput)
		fmt.Printf("     Path: %s\n", path)
	} else if !precheckQuiet {
		log.Success("✅ %s", t.name)
	}

	return nil
}

func checkGoWorkspace(log *logger.Logger) {
	// Check for go.mod
	if _, err := os.Stat("go.mod"); err == nil {
		log.Success("✅ go.mod found")
	} else {
		log.Warn("⚠️  go.mod not found in current directory")
	}

	// Check for go.work
	if _, err := os.Stat("go.work"); err == nil {
		log.Success("✅ go.work found (workspace mode)")
	}

	// Check GOPATH
	gopath := os.Getenv("GOPATH")
	if gopath != "" && precheckVerbose {
		fmt.Printf("  GOPATH: %s\n", gopath)
	}

	// Check module mode
	if precheckVerbose {
		cmd := exec.Command("go", "env", "GO111MODULE")
		if output, err := cmd.Output(); err == nil {
			fmt.Printf("  GO111MODULE: %s", output)
		}
	}
}

func checkDockerDaemon(log *logger.Logger) {
	cmd := exec.Command("docker", "info")
	if err := cmd.Run(); err != nil {
		log.Error("❌ Docker daemon is not running")
		if precheckVerbose {
			fmt.Println("     Start Docker Desktop or run: sudo systemctl start docker")
		}
		return
	}

	log.Success("✅ Docker daemon is running")

	// Check Docker Compose
	if _, err := exec.LookPath("docker-compose"); err == nil {
		log.Success("✅ Docker Compose is available")
	} else {
		// Check for docker compose (v2)
		cmd := exec.Command("docker", "compose", "version")
		if err := cmd.Run(); err == nil {
			log.Success("✅ Docker Compose v2 is available")
		}
	}
}
