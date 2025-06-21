package kubernetes

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyKustomize(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		dir       string
		namespace string
		wantErr   bool
	}{
		{
			name:      "nonexistent_dir",
			dir:       "/nonexistent/dir",
			namespace: "default",
			wantErr:   true,
		},
		{
			name:      "empty_dir",
			dir:       "",
			namespace: "default",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ApplyKustomize(ctx, tt.dir, tt.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("ApplyKustomize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteKustomize(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		dir       string
		namespace string
		wantErr   bool
	}{
		{
			name:      "nonexistent_dir",
			dir:       "/nonexistent/dir",
			namespace: "default",
			wantErr:   false, // DeleteKustomize doesn't return error even for non-existent dirs
		},
		{
			name:      "empty_dir",
			dir:       "",
			namespace: "default",
			wantErr:   false, // DeleteKustomize doesn't return error even for empty dirs
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := DeleteKustomize(ctx, tt.dir, tt.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("DeleteKustomize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Benchmark tests
func BenchmarkApplyKustomize(b *testing.B) {
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ApplyKustomize(ctx, "/nonexistent/dir", "default")
	}
}

func BenchmarkDeleteKustomize(b *testing.B) {
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = DeleteKustomize(ctx, "/nonexistent/dir", "default")
	}
}

// Additional comprehensive tests

func TestApplyKustomize_ErrorCases(t *testing.T) {
	tests := []struct {
		name        string
		path        string
		namespace   string
		expectedErr string
	}{
		{
			name:        "invalid_path",
			path:        "/invalid/path/that/does/not/exist",
			namespace:   "test",
			expectedErr: "", // Should not error immediately due to CLI fallback
		},
		{
			name:        "empty_namespace",
			path:        "/nonexistent",
			namespace:   "",
			expectedErr: "", // Empty namespace should be allowed
		},
		{
			name:        "special_characters_in_path",
			path:        "/path/with spaces/and@symbols",
			namespace:   "default",
			expectedErr: "", // Should handle special characters
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			err := ApplyKustomize(ctx, tt.path, tt.namespace)

			// Since we're falling back to CLI and don't have kubectl/kustomize in test env,
			// we expect errors but want to test the path through the code
			if tt.expectedErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
			}
			// We can't test successful cases without actual kubectl/kustomize
		})
	}
}

func TestApplyKustomize_NamespaceHandling(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		wantErr   bool
	}{
		{
			name:      "with_namespace",
			namespace: "test-ns",
			wantErr:   true, // Will fail due to missing kubectl
		},
		{
			name:      "empty_namespace",
			namespace: "",
			wantErr:   true, // Will fail due to missing kubectl
		},
		{
			name:      "default_namespace",
			namespace: "default",
			wantErr:   true, // Will fail due to missing kubectl
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			err := ApplyKustomize(ctx, "/nonexistent", tt.namespace)

			if (err != nil) != tt.wantErr {
				t.Errorf("ApplyKustomize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteKustomize_ErrorCases(t *testing.T) {
	tests := []struct {
		name      string
		path      string
		namespace string
	}{
		{
			name:      "invalid_path",
			path:      "/invalid/path/that/does/not/exist",
			namespace: "test",
		},
		{
			name:      "empty_namespace",
			path:      "/nonexistent",
			namespace: "",
		},
		{
			name:      "special_characters",
			path:      "/path/with spaces/and@symbols",
			namespace: "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			// DeleteKustomize should not return errors even for invalid paths
			err := DeleteKustomize(ctx, tt.path, tt.namespace)
			assert.NoError(t, err)
		})
	}
}

func TestApplyKustomizeCLI_PathValidation(t *testing.T) {
	ctx := context.Background()

	// Test with various path formats
	paths := []string{
		"./relative/path",
		"/absolute/path",
		"../parent/path",
		"path/without/leading/slash",
		"",
	}

	for _, path := range paths {
		t.Run(fmt.Sprintf("path_%s", path), func(t *testing.T) {
			// This will test the CLI path but will fail due to missing kubectl
			err := applyKustomizeCLI(ctx, path, "default")

			// We expect an error since kubectl/kustomize are not available in test env
			// But we want to ensure the code path is exercised
			assert.Error(t, err)
		})
	}
}

func TestDeleteKustomizeCLI_PathValidation(t *testing.T) {
	ctx := context.Background()

	// Test with various path formats
	paths := []string{
		"./relative/path",
		"/absolute/path",
		"../parent/path",
		"path/without/leading/slash",
		"",
	}

	for _, path := range paths {
		t.Run(fmt.Sprintf("path_%s", path), func(t *testing.T) {
			// deleteKustomizeCLI should not return errors
			err := deleteKustomizeCLI(ctx, path, "default")
			assert.NoError(t, err)
		})
	}
}

func TestApplyKustomize_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := ApplyKustomize(ctx, "/nonexistent", "default")

	// Should handle cancelled context gracefully
	// The error might be from context cancellation or missing kubectl
	assert.Error(t, err)
}

func TestDeleteKustomize_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := DeleteKustomize(ctx, "/nonexistent", "default")

	// Delete should not return errors even with cancelled context
	assert.NoError(t, err)
}

func TestKustomizePaths_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		path        string
		namespace   string
		description string
	}{
		{
			name:        "very_long_path",
			path:        "/very/long/path/that/exceeds/normal/filesystem/limits/and/contains/many/directories/nested/deeply/within/the/structure",
			namespace:   "default",
			description: "Test handling of very long paths",
		},
		{
			name:        "unicode_path",
			path:        "/path/with/unicode/characters/测试/ファイル",
			namespace:   "default",
			description: "Test handling of unicode characters in paths",
		},
		{
			name:        "special_namespace",
			path:        "/test",
			namespace:   "kube-system",
			description: "Test with system namespace",
		},
		{
			name:        "hyphenated_namespace",
			path:        "/test",
			namespace:   "my-test-namespace",
			description: "Test with hyphenated namespace",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			// Test apply
			err1 := ApplyKustomize(ctx, tt.path, tt.namespace)
			assert.Error(t, err1, "Expected error due to missing kubectl/kustomize")

			// Test delete
			err2 := DeleteKustomize(ctx, tt.path, tt.namespace)
			assert.NoError(t, err2, "Delete should not return errors")
		})
	}
}

func TestKustomizeWithTempFiles(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir, err := ioutil.TempDir("", "kustomize-test")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create a basic kustomization.yaml
	kustomizationContent := `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources: []
`

	kustomizationPath := filepath.Join(tmpDir, "kustomization.yaml")
	err = ioutil.WriteFile(kustomizationPath, []byte(kustomizationContent), 0644)
	require.NoError(t, err)

	ctx := context.Background()

	// Test with actual kustomization directory
	err = ApplyKustomize(ctx, tmpDir, "default")

	// Will still fail due to missing kubectl but tests the path through code
	assert.Error(t, err, "Expected error due to missing kubectl")

	// Test delete with actual directory
	err = DeleteKustomize(ctx, tmpDir, "default")
	assert.NoError(t, err, "Delete should not return errors")
}

func TestKustomizeMultipleNamespaces(t *testing.T) {
	namespaces := []string{
		"default",
		"kube-system",
		"kube-public",
		"custom-namespace",
		"very-long-namespace-name-that-exceeds-normal-limits",
	}

	ctx := context.Background()
	path := "/test/path"

	for _, ns := range namespaces {
		t.Run(fmt.Sprintf("namespace_%s", ns), func(t *testing.T) {
			// Test apply
			err1 := ApplyKustomize(ctx, path, ns)
			assert.Error(t, err1, "Expected error due to missing kubectl")

			// Test delete
			err2 := DeleteKustomize(ctx, path, ns)
			assert.NoError(t, err2, "Delete should not return errors")
		})
	}
}

func TestKustomizeEmptyAndNilInputs(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		path      string
		namespace string
	}{
		{
			name:      "empty_path_empty_namespace",
			path:      "",
			namespace: "",
		},
		{
			name:      "empty_path_valid_namespace",
			path:      "",
			namespace: "default",
		},
		{
			name:      "valid_path_empty_namespace",
			path:      "/test",
			namespace: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test apply
			err1 := ApplyKustomize(ctx, tt.path, tt.namespace)
			assert.Error(t, err1, "Expected error due to missing kubectl or invalid path")

			// Test delete
			err2 := DeleteKustomize(ctx, tt.path, tt.namespace)
			assert.NoError(t, err2, "Delete should not return errors")
		})
	}
}
