package taskdoc

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGenerator(t *testing.T) {
	tests := []struct {
		name    string
		opts    []Option
		wantErr bool
	}{
		{
			name: "default options",
			opts: nil,
		},
		{
			name: "with root dir",
			opts: []Option{WithRootDir(".")},
		},
		{
			name: "with verbose",
			opts: []Option{WithVerbose(true)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := NewGenerator(tt.opts...)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, g)
			assert.NotNil(t, g.taskfiles)
		})
	}
}

func TestParseTaskfile(t *testing.T) {
	// Create a temporary taskfile for testing
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "Taskfile.yaml")

	content := `# Test taskfile
# This is a comment
version: '3'

vars:
  TEST_VAR: test

includes:
  build:
    taskfile: ./build.yaml

tasks:
  test:
    desc: Run tests
    cmds:
      - go test ./...

  build:
    desc: Build project
    deps: [test]
    cmds:
      - go build ./...
`

	err := os.WriteFile(testFile, []byte(content), 0644)
	require.NoError(t, err)

	g := &Generator{
		taskfiles: make(map[string]*Taskfile),
	}

	taskfile, err := g.parseTaskfile(testFile)
	require.NoError(t, err)
	assert.NotNil(t, taskfile)

	// Check parsed content
	assert.Equal(t, "3", taskfile.Version)
	assert.Len(t, taskfile.Comments, 2)
	assert.Contains(t, taskfile.Comments[0], "Test taskfile")
	assert.Len(t, taskfile.Tasks, 2)
	assert.Equal(t, "Run tests", taskfile.Tasks["test"].Desc)
	assert.Equal(t, "Build project", taskfile.Tasks["build"].Desc)
	assert.Len(t, taskfile.Includes, 1)
}

func TestGetTaskCategories(t *testing.T) {
	g := &Generator{}

	taskfile := &Taskfile{
		Tasks: map[string]Task{
			"build":            {},
			"build:docker":     {},
			"build:binary":     {},
			"test":             {},
			"test:unit":        {},
			"test:integration": {},
			"deploy":           {},
			"clean":            {},
		},
	}

	categories := g.getTaskCategories(taskfile)

	assert.Contains(t, categories, "build")
	assert.Contains(t, categories, "test")
	assert.Contains(t, categories, "(root)")

	// Check that we have the right number of categories
	// We have 3 build tasks, 3 test tasks, and 3 root tasks (test, deploy, clean)
	assert.Len(t, categories, 3) // build, test, (root)
}

func TestGetKeyTasks(t *testing.T) {
	g := &Generator{}

	taskfile := &Taskfile{
		Tasks: map[string]Task{
			"build":  {Desc: "Build the project"},
			"test":   {Desc: "Run tests"},
			"clean":  {}, // No description
			"deploy": {Desc: "Deploy application"},
		},
	}

	keyTasks := g.getKeyTasks(taskfile)

	assert.Len(t, keyTasks, 3) // Only tasks with descriptions
	assert.Equal(t, "build", keyTasks[0].Name)
	assert.Equal(t, "Build the project", keyTasks[0].Desc)
}

func TestGenerateMarkdown(t *testing.T) {
	// Create a simple generator with test data
	g := &Generator{
		rootDir: ".",
		taskfiles: map[string]*Taskfile{
			"Taskfile.yaml": {
				Path:     "Taskfile.yaml",
				Version:  "3",
				Comments: []string{"Main taskfile", "For testing"},
				Tasks: map[string]Task{
					"build": {Desc: "Build project"},
					"test":  {Desc: "Run tests"},
				},
				Includes: map[string]Include{
					"common": {Taskfile: "./common.yaml"},
				},
			},
		},
	}

	markdown, err := g.GenerateMarkdown()
	require.NoError(t, err)
	assert.NotEmpty(t, markdown)

	// Check content
	assert.Contains(t, markdown, "# Taskfile Structure and Dependencies")
	assert.Contains(t, markdown, "## Overview")
	assert.Contains(t, markdown, "Main taskfile")
	assert.Contains(t, markdown, "Build project")
	assert.Contains(t, markdown, "## Task Flow Examples")
}

func TestGenerateJSON(t *testing.T) {
	g := &Generator{
		rootDir: ".",
		taskfiles: map[string]*Taskfile{
			"test.yaml": {
				Path:    "test.yaml",
				Version: "3",
				Tasks: map[string]Task{
					"test": {Desc: "Test task"},
				},
			},
		},
	}

	jsonStr, err := g.GenerateJSON()
	require.NoError(t, err)
	assert.NotEmpty(t, jsonStr)

	// Check JSON structure
	assert.Contains(t, jsonStr, `"generated"`)
	assert.Contains(t, jsonStr, `"taskfiles"`)
	assert.Contains(t, jsonStr, `"statistics"`)
	assert.Contains(t, jsonStr, `"total_taskfiles": 1`)
}

func TestGenerateFlow(t *testing.T) {
	g := &Generator{
		rootDir: ".",
		taskfiles: map[string]*Taskfile{
			"Taskfile.yaml": {
				Includes: map[string]Include{
					"build": {Taskfile: "./build.yaml"},
				},
				Tasks: map[string]Task{
					"build:app":   {},
					"deploy:prod": {},
					"test:unit":   {},
				},
			},
		},
	}

	flow, err := g.GenerateFlow()
	require.NoError(t, err)
	assert.NotEmpty(t, flow)

	// Check content
	assert.Contains(t, flow, "Task Flow Overview")
	assert.Contains(t, flow, "High-level workflows")
	assert.Contains(t, flow, "Task categories")
}

func TestGetCategoryDescription(t *testing.T) {
	g := &Generator{}

	tests := []struct {
		category string
		expected string
	}{
		{"build", "Compilation & packaging"},
		{"deploy", "Kubernetes operations"},
		{"test", "Testing & validation"},
		{"unknown", "Various operations"},
	}

	for _, tt := range tests {
		t.Run(tt.category, func(t *testing.T) {
			desc := g.getCategoryDescription(tt.category)
			assert.Equal(t, tt.expected, desc)
		})
	}
}

func TestGenerateHierarchy(t *testing.T) {
	g := &Generator{
		taskfiles: map[string]*Taskfile{
			"Taskfile.yaml": {
				Includes: map[string]Include{
					"build":  {Taskfile: "./devops/build.yaml"},
					"deploy": {Taskfile: "./devops/deploy.yaml"},
				},
			},
			"iaac/Taskfile.yaml": {
				Includes: map[string]Include{
					"common": {Taskfile: "../common.yaml"},
				},
			},
		},
	}

	hierarchy := g.generateHierarchy()
	assert.NotEmpty(t, hierarchy)

	// Check structure
	assert.Contains(t, hierarchy, "Taskfile.yaml")
	assert.Contains(t, hierarchy, "build:")
	assert.Contains(t, hierarchy, "deploy:")
	assert.Contains(t, hierarchy, "Standalone Taskfiles")
	assert.Contains(t, hierarchy, "iaac/Taskfile.yaml")
}

func TestMinFunction(t *testing.T) {
	assert.Equal(t, 1, min(1, 2))
	assert.Equal(t, 1, min(2, 1))
	assert.Equal(t, 5, min(5, 5))
}

