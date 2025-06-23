package taskdoc

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Taskfile represents a parsed Taskfile.yaml
type Taskfile struct {
	Path     string
	Version  string                 `yaml:"version"`
	Includes map[string]Include     `yaml:"includes"`
	Vars     map[string]interface{} `yaml:"vars"`
	Env      map[string]interface{} `yaml:"env"`
	Tasks    map[string]Task        `yaml:"tasks"`
	Silent   bool                   `yaml:"silent"`
	Comments []string               // Extracted from file
}

// Include represents an included taskfile
type Include struct {
	Taskfile string `yaml:"taskfile"`
	Dir      string `yaml:"dir"`
}

// Task represents a task definition
type Task struct {
	Desc          string                 `yaml:"desc"`
	Summary       string                 `yaml:"summary"`
	Aliases       []string               `yaml:"aliases"`
	Deps          []interface{}          `yaml:"deps"`
	Cmds          []interface{}          `yaml:"cmds"`
	Preconditions []interface{}          `yaml:"preconditions"`
	Dir           string                 `yaml:"dir"`
	Vars          map[string]interface{} `yaml:"vars"`
	Env           map[string]interface{} `yaml:"env"`
	Silent        bool                   `yaml:"silent"`
	Interactive   bool                   `yaml:"interactive"`
}

// Generator generates documentation for Taskfiles
type Generator struct {
	rootDir   string
	taskfiles map[string]*Taskfile
	verbose   bool
}

// Option configures the generator
type Option func(*Generator)

// WithRootDir sets the root directory
func WithRootDir(dir string) Option {
	return func(g *Generator) {
		g.rootDir = dir
	}
}

// WithVerbose enables verbose output
func WithVerbose(verbose bool) Option {
	return func(g *Generator) {
		g.verbose = verbose
	}
}

// NewGenerator creates a new documentation generator
func NewGenerator(opts ...Option) (*Generator, error) {
	g := &Generator{
		rootDir:   ".",
		taskfiles: make(map[string]*Taskfile),
	}

	for _, opt := range opts {
		opt(g)
	}

	// Find and parse all taskfiles
	if err := g.findAndParseTaskfiles(); err != nil {
		return nil, fmt.Errorf("failed to parse taskfiles: %w", err)
	}

	return g, nil
}

// findAndParseTaskfiles finds all Taskfile.yaml files and parses them
func (g *Generator) findAndParseTaskfiles() error {
	pattern := filepath.Join(g.rootDir, "**", "Taskfile*.y*ml")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}

	// Also check root directory
	rootPatterns := []string{
		filepath.Join(g.rootDir, "Taskfile.yaml"),
		filepath.Join(g.rootDir, "Taskfile.yml"),
	}
	for _, p := range rootPatterns {
		if _, err := os.Stat(p); err == nil {
			matches = append(matches, p)
		}
	}

	// Walk directory tree for nested taskfiles
	err = filepath.Walk(g.rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors
		}
		if strings.Contains(path, ".git") {
			return nil
		}
		if strings.HasPrefix(info.Name(), "Taskfile") &&
			(strings.HasSuffix(info.Name(), ".yaml") || strings.HasSuffix(info.Name(), ".yml")) {
			matches = append(matches, path)
		}
		return nil
	})
	if err != nil {
		return err
	}

	// Remove duplicates
	seen := make(map[string]bool)
	unique := []string{}
	for _, m := range matches {
		if !seen[m] {
			seen[m] = true
			unique = append(unique, m)
		}
	}

	// Parse each taskfile
	for _, path := range unique {
		if g.verbose {
			fmt.Printf("Parsing: %s\n", path)
		}
		taskfile, err := g.parseTaskfile(path)
		if err != nil {
			if g.verbose {
				fmt.Printf("Warning: failed to parse %s: %v\n", path, err)
			}
			continue
		}
		relPath, _ := filepath.Rel(g.rootDir, path)
		g.taskfiles[relPath] = taskfile
	}

	return nil
}

// parseTaskfile parses a single taskfile
func (g *Generator) parseTaskfile(path string) (*Taskfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Extract comments from the beginning of the file
	lines := strings.Split(string(data), "\n")
	var comments []string
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "#") && !strings.HasPrefix(strings.TrimSpace(line), "#!") {
			comment := strings.TrimPrefix(strings.TrimSpace(line), "#")
			comments = append(comments, strings.TrimSpace(comment))
		} else if strings.TrimSpace(line) != "" {
			break
		}
	}

	var taskfile Taskfile
	if err := yaml.Unmarshal(data, &taskfile); err != nil {
		return nil, err
	}

	taskfile.Path = path
	taskfile.Comments = comments

	return &taskfile, nil
}

// GenerateMarkdown generates markdown documentation
func (g *Generator) GenerateMarkdown() (string, error) {
	var sb strings.Builder

	// Header
	sb.WriteString("# Taskfile Structure and Dependencies\n\n")
	sb.WriteString(fmt.Sprintf("Generated on: %s\n\n", time.Now().Format("2006-01-02 15:04:05")))

	// Overview
	sb.WriteString("## Overview\n\n")
	sb.WriteString("This project uses [Task](https://taskfile.dev) for build automation and deployment orchestration.\n\n")

	// Statistics
	totalTasks := 0
	for _, tf := range g.taskfiles {
		totalTasks += len(tf.Tasks)
	}
	sb.WriteString(fmt.Sprintf("- **Total Taskfiles**: %d\n", len(g.taskfiles)))
	sb.WriteString(fmt.Sprintf("- **Total Tasks**: %d\n\n", totalTasks))

	// Hierarchy
	sb.WriteString("## Taskfile Hierarchy\n\n")
	sb.WriteString("```\n")
	sb.WriteString(g.generateHierarchy())
	sb.WriteString("```\n\n")

	// Taskfile details
	sb.WriteString("## Taskfile Details\n\n")

	// Sort taskfiles by path
	var paths []string
	for path := range g.taskfiles {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	for _, path := range paths {
		tf := g.taskfiles[path]
		sb.WriteString(fmt.Sprintf("### 📁 %s\n\n", path))

		// Purpose from comments
		if len(tf.Comments) > 0 {
			sb.WriteString("**Purpose:**\n")
			for _, comment := range tf.Comments[:min(5, len(tf.Comments))] {
				sb.WriteString(fmt.Sprintf("> %s\n", comment))
			}
			sb.WriteString("\n")
		}

		// Statistics
		sb.WriteString(fmt.Sprintf("**Number of tasks:** %d\n\n", len(tf.Tasks)))

		// Includes
		if len(tf.Includes) > 0 {
			sb.WriteString("**Includes:**\n")
			for name, inc := range tf.Includes {
				sb.WriteString(fmt.Sprintf("- `%s`: %s\n", name, inc.Taskfile))
			}
			sb.WriteString("\n")
		}

		// Task categories
		categories := g.getTaskCategories(tf)
		if len(categories) > 0 {
			sb.WriteString("**Main task categories:**\n")
			for _, cat := range categories[:min(10, len(categories))] {
				sb.WriteString(fmt.Sprintf("- `%s`\n", cat))
			}
			sb.WriteString("\n")
		}

		// Key tasks with descriptions
		keyTasks := g.getKeyTasks(tf)
		if len(keyTasks) > 0 {
			sb.WriteString("**Key tasks:**\n")
			for _, task := range keyTasks[:min(5, len(keyTasks))] {
				sb.WriteString(fmt.Sprintf("- `%s`: %s\n", task.Name, task.Desc))
			}
			sb.WriteString("\n")
		}

		sb.WriteString("---\n\n")
	}

	// Task flows
	sb.WriteString("## Task Flow Examples\n\n")
	sb.WriteString(g.generateTaskFlows())

	// Common patterns
	sb.WriteString("## Common Task Patterns\n\n")
	sb.WriteString(g.generatePatterns())

	// Quick reference
	sb.WriteString("## Quick Reference\n\n")
	sb.WriteString(g.generateQuickReference())

	return sb.String(), nil
}

// generateHierarchy creates a tree view of taskfile includes
func (g *Generator) generateHierarchy() string {
	var sb strings.Builder

	// Find root taskfile
	rootPath := "Taskfile.yaml"
	if _, exists := g.taskfiles[rootPath]; !exists {
		// Try to find the root taskfile
		for path := range g.taskfiles {
			if filepath.Base(path) == "Taskfile.yaml" && filepath.Dir(path) == "." {
				rootPath = path
				break
			}
		}
	}

	// Generate tree starting from root
	sb.WriteString(fmt.Sprintf("%s\n", rootPath))
	if root, exists := g.taskfiles[rootPath]; exists {
		g.writeIncludes(&sb, root, "├── ", "│   ")
	}

	// List other taskfiles not included from root
	sb.WriteString("\nStandalone Taskfiles:\n")
	for path := range g.taskfiles {
		if path != rootPath {
			sb.WriteString(fmt.Sprintf("- %s\n", path))
		}
	}

	return sb.String()
}

// writeIncludes recursively writes included taskfiles
func (g *Generator) writeIncludes(sb *strings.Builder, tf *Taskfile, prefix, indent string) {
	if len(tf.Includes) == 0 {
		return
	}

	i := 0
	for name, inc := range tf.Includes {
		i++
		isLast := i == len(tf.Includes)

		if isLast {
			sb.WriteString(fmt.Sprintf("%s└── %s: %s\n", indent, name, inc.Taskfile))
		} else {
			sb.WriteString(fmt.Sprintf("%s├── %s: %s\n", indent, name, inc.Taskfile))
		}
	}
}

// getTaskCategories extracts task category prefixes
func (g *Generator) getTaskCategories(tf *Taskfile) []string {
	categories := make(map[string]int)

	for name := range tf.Tasks {
		parts := strings.Split(name, ":")
		if len(parts) > 1 {
			categories[parts[0]]++
		} else {
			categories["(root)"]++
		}
	}

	// Sort by frequency
	var result []string
	for cat := range categories {
		result = append(result, cat)
	}
	sort.Slice(result, func(i, j int) bool {
		return categories[result[i]] > categories[result[j]]
	})

	return result
}

// TaskInfo holds task name and description
type TaskInfo struct {
	Name string
	Desc string
}

// getKeyTasks returns tasks with descriptions
func (g *Generator) getKeyTasks(tf *Taskfile) []TaskInfo {
	var tasks []TaskInfo

	for name, task := range tf.Tasks {
		if task.Desc != "" {
			tasks = append(tasks, TaskInfo{Name: name, Desc: task.Desc})
		}
	}

	// Sort by name
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].Name < tasks[j].Name
	})

	return tasks
}

// generateTaskFlows creates mermaid diagrams for common workflows
func (g *Generator) generateTaskFlows() string {
	flows := []struct {
		title string
		flow  string
	}{
		{
			title: "🚀 Full Production Workflow",
			flow: `graph LR
    A[task full] --> B[setup]
    B --> C[build]
    C --> D[deploy]
    D --> E[test]
    B --> F[Create k3d cluster]
    C --> G[Build Go binary]
    D --> H[Deploy to k8s]
    E --> I[Run tests]`,
		},
		{
			title: "🔧 Development Workflow",
			flow: `graph LR
    A[task dev] --> B[build]
    A --> C[deploy:dev]
    B --> D[Compile]
    C --> E[Quick deploy]
    C --> F[Quick test]`,
		},
		{
			title: "🧪 CI/CD Pipeline",
			flow: `graph TD
    A[task ci] --> B[fmt:check]
    A --> C[vet]
    A --> D[lint]
    A --> E[test]
    A --> F[build]
    E --> G[coverage]
    F --> H[docker:build]`,
		},
	}

	var sb strings.Builder
	for _, f := range flows {
		sb.WriteString(fmt.Sprintf("### %s\n\n", f.title))
		sb.WriteString("```mermaid\n")
		sb.WriteString(f.flow)
		sb.WriteString("\n```\n\n")
	}

	return sb.String()
}

// generatePatterns documents common task naming patterns
func (g *Generator) generatePatterns() string {
	return `| Pattern | Example | Purpose |
|---------|---------|---------|
| build:* | build:test, build:docker | Build-related tasks |
| deploy:* | deploy:setup, deploy:verify | Deployment tasks |
| test:* | test:unit, test:integration | Testing tasks |
| clean/cleanup | clean, cleanup | Cleanup tasks |
| *:info/*:status | cluster:info, status | Information tasks |
| port-forward:* | port-forward:api | Port forwarding |
| docs:* | docs:generate, docs:flow | Documentation |

`
}

// generateQuickReference creates a quick command reference
func (g *Generator) generateQuickReference() string {
	var sb strings.Builder

	sb.WriteString("### Most Used Commands\n\n")
	sb.WriteString("```bash\n")
	sb.WriteString("# Development\n")
	sb.WriteString("task dev          # Quick development cycle\n")
	sb.WriteString("task build        # Build the application\n")
	sb.WriteString("task test         # Run all tests\n")
	sb.WriteString("task fmt          # Format code\n")
	sb.WriteString("\n")
	sb.WriteString("# Deployment\n")
	sb.WriteString("task full         # Complete production workflow\n")
	sb.WriteString("task setup        # Create k3d cluster\n")
	sb.WriteString("task deploy       # Deploy application\n")
	sb.WriteString("task status       # Check deployment status\n")
	sb.WriteString("\n")
	sb.WriteString("# Maintenance\n")
	sb.WriteString("task clean        # Clean build artifacts\n")
	sb.WriteString("task cleanup      # Destroy cluster\n")
	sb.WriteString("task logs         # View application logs\n")
	sb.WriteString("task docs         # Show documentation\n")
	sb.WriteString("```\n\n")

	sb.WriteString("### Getting Help\n\n")
	sb.WriteString("```bash\n")
	sb.WriteString("task --list       # List all available tasks\n")
	sb.WriteString("task --list-all   # List all tasks including included ones\n")
	sb.WriteString("task -h          # Show task help\n")
	sb.WriteString("task <task> -h   # Show help for specific task\n")
	sb.WriteString("```\n")

	return sb.String()
}

// GenerateJSON generates JSON documentation
func (g *Generator) GenerateJSON() (string, error) {
	data := map[string]interface{}{
		"generated": time.Now().Format(time.RFC3339),
		"taskfiles": g.taskfiles,
		"statistics": map[string]interface{}{
			"total_taskfiles": len(g.taskfiles),
			"total_tasks":     g.countTotalTasks(),
		},
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", err
	}

	return string(jsonData), nil
}

// GenerateFlow generates just the task flow visualization
func (g *Generator) GenerateFlow() (string, error) {
	var sb strings.Builder

	sb.WriteString("🔀 Task Flow Overview\n")
	sb.WriteString("====================\n\n")

	// Hierarchy
	sb.WriteString(g.generateHierarchy())
	sb.WriteString("\n")

	// Key workflows
	sb.WriteString("High-level workflows:\n")
	sb.WriteString("├── full → setup → build → deploy → test\n")
	sb.WriteString("├── quick → build → deploy → quick-test\n")
	sb.WriteString("└── dev → build → deploy:dev\n\n")

	// Task categories summary
	sb.WriteString("Task categories:\n")
	categories := g.getAllCategories()
	for i, cat := range categories[:min(5, len(categories))] {
		isLast := i == min(4, len(categories)-1)
		if isLast {
			sb.WriteString(fmt.Sprintf("└── %s:* - %s\n", cat, g.getCategoryDescription(cat)))
		} else {
			sb.WriteString(fmt.Sprintf("├── %s:* - %s\n", cat, g.getCategoryDescription(cat)))
		}
	}

	return sb.String(), nil
}

// Helper functions

func (g *Generator) countTotalTasks() int {
	total := 0
	for _, tf := range g.taskfiles {
		total += len(tf.Tasks)
	}
	return total
}

func (g *Generator) getAllCategories() []string {
	categories := make(map[string]int)

	for _, tf := range g.taskfiles {
		for cat := range g.extractCategories(tf) {
			categories[cat]++
		}
	}

	var result []string
	for cat := range categories {
		result = append(result, cat)
	}
	sort.Slice(result, func(i, j int) bool {
		return categories[result[i]] > categories[result[j]]
	})

	return result
}

func (g *Generator) extractCategories(tf *Taskfile) map[string]bool {
	categories := make(map[string]bool)
	for name := range tf.Tasks {
		parts := strings.Split(name, ":")
		if len(parts) > 1 {
			categories[parts[0]] = true
		}
	}
	return categories
}

func (g *Generator) getCategoryDescription(cat string) string {
	descriptions := map[string]string{
		"build":        "Compilation & packaging",
		"deploy":       "Kubernetes operations",
		"test":         "Testing & validation",
		"docker":       "Container operations",
		"clean":        "Cleanup operations",
		"docs":         "Documentation",
		"port-forward": "Port forwarding",
		"debug":        "Debugging tools",
		"scale":        "Scaling operations",
	}

	if desc, exists := descriptions[cat]; exists {
		return desc
	}
	return "Various operations"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

