// Package tools provides tool registry implementation
package tools

import (
	"context"
	"fmt"
	"sync"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// Registry manages multiple tools
type Registry struct {
	tools  map[string]interfaces.ToolInstaller
	mu     sync.RWMutex
	logger interfaces.Logger
}

// NewRegistry creates a new tool registry
func NewRegistry(logger interfaces.Logger) *Registry {
	return &Registry{
		tools:  make(map[string]interfaces.ToolInstaller),
		logger: logger,
	}
}

// Register adds a tool to the registry
func (r *Registry) Register(tool interfaces.ToolInstaller) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if tool == nil {
		return fmt.Errorf("tool cannot be nil")
	}

	name := tool.Name()
	if name == "" {
		return fmt.Errorf("tool name cannot be empty")
	}

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool %s already registered", name)
	}

	r.tools[name] = tool
	r.logger.Debug("Registered tool: %s", name)
	return nil
}

// Get retrieves a tool by name
func (r *Registry) Get(name string) (interfaces.ToolInstaller, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, exists := r.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool %s not found", name)
	}

	return tool, nil
}

// List returns all registered tools
func (r *Registry) List() []interfaces.ToolInstaller {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]interfaces.ToolInstaller, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}

	return tools
}

// InstallAll installs all registered tools
func (r *Registry) InstallAll(ctx context.Context) error {
	tools := r.List()

	r.logger.Info("Installing %d tools...", len(tools))

	// Check what's already installed
	var toInstall []interfaces.ToolInstaller
	for _, tool := range tools {
		if tool.IsInstalled() {
			version, _ := tool.GetInstalledVersion()
			r.logger.Success("%s is already installed: %s", tool.Name(), version)
		} else {
			toInstall = append(toInstall, tool)
		}
	}

	if len(toInstall) == 0 {
		r.logger.Success("All tools are already installed!")
		return nil
	}

	// Install missing tools
	r.logger.Info("Installing %d missing tools...", len(toInstall))

	for _, tool := range toInstall {
		r.logger.Info("Installing %s (%s)...", tool.Name(), tool.Description())

		if err := tool.Install(ctx); err != nil {
			return fmt.Errorf("failed to install %s: %w", tool.Name(), err)
		}
	}

	r.logger.Success("All tools installed successfully!")
	return nil
}

// ValidateAll validates all registered tools
func (r *Registry) ValidateAll() error {
	tools := r.List()

	r.logger.Info("Validating %d tools...", len(tools))

	var errors []error
	for _, tool := range tools {
		if err := tool.Validate(); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("validation failed for %d tools", len(errors))
	}

	r.logger.Success("All tools validated successfully!")
	return nil
}

// InstallAllWithOptions installs all tools with options
func (r *Registry) InstallAllWithOptions(ctx context.Context, opts interfaces.InstallOptions) error {
	tools := r.List()

	r.logger.Info("Installing %d tools...", len(tools))

	if opts.Parallel && opts.MaxConcurrency > 1 {
		return r.installParallel(ctx, tools, opts)
	}

	return r.installSequential(ctx, tools, opts)
}

// installSequential installs tools one by one
func (r *Registry) installSequential(ctx context.Context, tools []interfaces.ToolInstaller, opts interfaces.InstallOptions) error {
	for _, tool := range tools {
		if !opts.Force && tool.IsInstalled() {
			version, _ := tool.GetInstalledVersion()
			r.logger.Success("%s is already installed: %s", tool.Name(), version)
			continue
		}

		r.logger.Info("Installing %s (%s)...", tool.Name(), tool.Description())

		if err := tool.Install(ctx); err != nil {
			return fmt.Errorf("failed to install %s: %w", tool.Name(), err)
		}

		if !opts.SkipValidation {
			if err := tool.Validate(); err != nil {
				return fmt.Errorf("validation failed for %s: %w", tool.Name(), err)
			}
		}
	}

	return nil
}

// installParallel installs tools in parallel
func (r *Registry) installParallel(ctx context.Context, tools []interfaces.ToolInstaller, opts interfaces.InstallOptions) error {
	// Use a semaphore to limit concurrency
	sem := make(chan struct{}, opts.MaxConcurrency)
	errChan := make(chan error, len(tools))
	var wg sync.WaitGroup

	for _, tool := range tools {
		if !opts.Force && tool.IsInstalled() {
			version, _ := tool.GetInstalledVersion()
			r.logger.Success("%s is already installed: %s", tool.Name(), version)
			continue
		}

		wg.Add(1)
		go func(t interfaces.ToolInstaller) {
			defer wg.Done()

			// Acquire semaphore
			sem <- struct{}{}
			defer func() { <-sem }()

			r.logger.Info("Installing %s (%s)...", t.Name(), t.Description())

			if err := t.Install(ctx); err != nil {
				errChan <- fmt.Errorf("failed to install %s: %w", t.Name(), err)
				return
			}

			if !opts.SkipValidation {
				if err := t.Validate(); err != nil {
					errChan <- fmt.Errorf("validation failed for %s: %w", t.Name(), err)
					return
				}
			}
		}(tool)
	}

	// Wait for all installations to complete
	go func() {
		wg.Wait()
		close(errChan)
	}()

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("installation failed for %d tools: %v", len(errors), errors[0])
	}

	r.logger.Success("All tools installed successfully!")
	return nil
}
