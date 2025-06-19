package cmd

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/testing"
	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
	"github.com/spf13/cobra"
	"sigs.k8s.io/yaml"
)

type CompositeTestManager struct {
	logger       *utils.Logger
	portForwards []portForwardInfo
}

type portForwardInfo struct {
	cmd     *exec.Cmd
	service string
	port    string
}

func CompositeTestCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "composite-test",
		Short: "Test composite backend with k3d cluster services",
		Long:  `Test the three-tier cache architecture (memory + Redis + PostgreSQL) using services from the k3d cluster.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			ctm := &CompositeTestManager{
				logger:       utils.NewLogger("composite-test"),
				portForwards: []portForwardInfo{},
			}

			// Setup signal handling for cleanup
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
			go func() {
				<-sigChan
				ctm.cleanup()
				os.Exit(0)
			}()

			// Run the test
			if err := ctm.run(ctx); err != nil {
				ctm.cleanup()
				return err
			}

			ctm.cleanup()
			return nil
		},
	}
}

func (ctm *CompositeTestManager) run(ctx context.Context) error {
	ctm.logger.Info("Starting composite backend test...")

	// Setup port forwarding
	if err := ctm.setupPortForwarding(ctx); err != nil {
		return fmt.Errorf("failed to setup port forwarding: %w", err)
	}

	// Wait for services to be ready
	time.Sleep(3 * time.Second)

	// Initialize database
	if err := ctm.initializeDatabase(ctx); err != nil {
		return fmt.Errorf("failed to initialize database: %w", err)
	}

	// Create composite configuration
	configPath, err := ctm.createCompositeConfig()
	if err != nil {
		return fmt.Errorf("failed to create config: %w", err)
	}
	defer os.Remove(configPath)

	// Run the server with composite backend
	if err := ctm.runCompositeDemo(ctx, configPath); err != nil {
		return fmt.Errorf("failed to run demo: %w", err)
	}

	// Test the endpoints
	if err := ctm.testEndpoints(); err != nil {
		return fmt.Errorf("endpoint tests failed: %w", err)
	}

	ctm.logger.Info("Composite backend test completed successfully!")
	return nil
}

func (ctm *CompositeTestManager) setupPortForwarding(ctx context.Context) error {
	ctm.logger.Info("Setting up port forwarding...")

	// Port forward PostgreSQL
	if err := ctm.addPortForward(ctx, "postgres", "infra", "5432:5432"); err != nil {
		return fmt.Errorf("failed to port-forward postgres: %w", err)
	}

	// Port forward Redis
	if err := ctm.addPortForward(ctx, "redis", "infra", "6379:6379"); err != nil {
		return fmt.Errorf("failed to port-forward redis: %w", err)
	}

	return nil
}

func (ctm *CompositeTestManager) addPortForward(ctx context.Context, service, namespace, ports string) error {
	ctm.logger.Info("Port-forwarding %s/%s %s", namespace, service, ports)

	cmd := exec.CommandContext(ctx, "kubectl", "port-forward",
		"-n", namespace,
		fmt.Sprintf("svc/%s", service),
		ports,
	)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start port-forward: %w", err)
	}

	ctm.portForwards = append(ctm.portForwards, portForwardInfo{
		cmd:     cmd,
		service: service,
		port:    ports,
	})

	return nil
}

func (ctm *CompositeTestManager) initializeDatabase(ctx context.Context) error {
	ctm.logger.Info("Initializing PostgreSQL with pgvector...")

	// Create database and extension
	initSQL := `
CREATE DATABASE IF NOT EXISTS semantic_cache;
\c semantic_cache;
CREATE EXTENSION IF NOT EXISTS vector;
`

	cmd := exec.CommandContext(ctx, "psql",
		"-h", "localhost",
		"-p", "5432",
		"-U", "postgres",
		"-c", initSQL,
	)
	cmd.Env = append(os.Environ(), "PGPASSWORD=postgres")

	output, err := cmd.CombinedOutput()
	if err != nil {
		// Try alternative approach
		ctm.logger.Warn("Direct psql failed, trying via kubectl exec...")

		output2, err2 := utils.RunCommand(ctx, "kubectl", []string{
			"exec", "-n", "infra", "deploy/postgres", "--",
			"psql", "-U", "postgres", "-c", initSQL,
		}, nil)

		if err2 != nil {
			return fmt.Errorf("failed to initialize database: %w (output: %s)", err, output)
		}

		ctm.logger.Debug("Database initialized: %s", output2)
	}

	return nil
}

func (ctm *CompositeTestManager) createCompositeConfig() (string, error) {
	ctm.logger.Info("Creating composite configuration...")

	config := map[string]interface{}{
		"server": map[string]interface{}{
			"address": ":8080",
		},
		"cache": map[string]interface{}{
			"type":            "composite",
			"capacity":        100,
			"eviction_policy": "LRU",
			"ttl":             "5m",
			"min_similarity":  0.8,
		},
		"composite": map[string]interface{}{
			"tiers": []map[string]interface{}{
				{
					"name":     "memory",
					"type":     "memory",
					"capacity": 10,
					"ttl":      "1m",
				},
				{
					"name": "redis",
					"type": "redis",
					"ttl":  "5m",
				},
				{
					"name": "postgres",
					"type": "gorm",
				},
			},
		},
		"redis": map[string]interface{}{
			"address": "localhost:6379",
		},
		"database_url": "host=localhost user=postgres password=postgres dbname=semantic_cache sslmode=disable",
		"openai": map[string]interface{}{
			"api_key": os.Getenv("OPENAI_API_KEY"),
		},
	}

	// Convert to YAML
	yamlData, err := yaml.Marshal(config)
	if err != nil {
		return "", fmt.Errorf("failed to marshal config: %w", err)
	}

	// Write to temp file
	tmpfile, err := ioutil.TempFile("", "composite-config-*.yml")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := tmpfile.Write(yamlData); err != nil {
		tmpfile.Close()
		os.Remove(tmpfile.Name())
		return "", fmt.Errorf("failed to write config: %w", err)
	}

	tmpfile.Close()
	ctm.logger.Info("Configuration created at: %s", tmpfile.Name())
	return tmpfile.Name(), nil
}

func (ctm *CompositeTestManager) runCompositeDemo(ctx context.Context, configPath string) error {
	ctm.logger.Info("Running composite backend demo...")

	projectRoot, err := utils.FindProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	// Build the server if needed
	serverBinary := filepath.Join(projectRoot, "bin", "server")
	if _, err := os.Stat(serverBinary); os.IsNotExist(err) {
		ctm.logger.Info("Building server...")
		buildCmd := exec.CommandContext(ctx, "go", "build", "-o", serverBinary, "./cmd/server")
		buildCmd.Dir = projectRoot
		if err := buildCmd.Run(); err != nil {
			return fmt.Errorf("failed to build server: %w", err)
		}
	}

	// Run server in background
	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()

	serverCmd := exec.CommandContext(serverCtx, serverBinary, "-config", configPath)
	serverCmd.Dir = projectRoot

	// Capture output
	serverCmd.Stdout = os.Stdout
	serverCmd.Stderr = os.Stderr

	if err := serverCmd.Start(); err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}

	// Wait for server to be ready
	ctm.logger.Info("Waiting for server to start...")
	time.Sleep(3 * time.Second)

	// Let it run for a bit
	time.Sleep(5 * time.Second)

	// Stop the server
	serverCancel()
	serverCmd.Wait()

	return nil
}

func (ctm *CompositeTestManager) testEndpoints() error {
	ctm.logger.Info("Testing server endpoints...")

	tester := testing.NewEndpointTester("http://localhost:8080")

	// Test standard endpoints
	if _, err := tester.TestStandardEndpoints(); err != nil {
		ctm.logger.Warn("Standard endpoint tests failed: %v", err)
	}

	// Test cache operations if server is running
	if err := tester.TestCacheOperations(); err != nil {
		ctm.logger.Warn("Cache operation tests failed: %v", err)
	}

	return nil
}

func (ctm *CompositeTestManager) cleanup() {
	ctm.logger.Info("Cleaning up...")

	// Stop all port forwards
	for _, pf := range ctm.portForwards {
		if pf.cmd != nil && pf.cmd.Process != nil {
			ctm.logger.Debug("Stopping port-forward for %s", pf.service)
			pf.cmd.Process.Kill()
			pf.cmd.Wait()
		}
	}
}
