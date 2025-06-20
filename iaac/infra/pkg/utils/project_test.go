package utils

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindProjectRoot(t *testing.T) {
	// Create a temporary directory structure
	tmpDir, err := os.MkdirTemp("", "test-project")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Logf("failed to remove temp directory: %v", err)
		}
	}()

	// Create nested directories
	subDir := filepath.Join(tmpDir, "sub", "dir", "deep")
	err = os.MkdirAll(subDir, 0755)
	if err != nil {
		t.Fatal(err)
	}

	// Create go.mod in root
	goModPath := filepath.Join(tmpDir, "go.mod")
	err = os.WriteFile(goModPath, []byte("module test\n"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		startDir    string
		wantRoot    string
		wantErr     bool
		setupFunc   func()
		cleanupFunc func()
	}{
		{
			name:     "from_root_directory",
			startDir: tmpDir,
			wantRoot: tmpDir,
			wantErr:  false,
		},
		{
			name:     "from_subdirectory",
			startDir: subDir,
			wantRoot: tmpDir,
			wantErr:  false,
		},
		{
			name:     "current_directory",
			startDir: "",
			wantRoot: "", // Will be determined at runtime
			wantErr:  false,
		},
		{
			name:     "no_go_mod",
			startDir: "",
			wantRoot: "",
			wantErr:  true,
			setupFunc: func() {
				// Create temp dir without go.mod
				tempNoMod, _ := os.MkdirTemp("", "no-go-mod")
				if err := os.Chdir(tempNoMod); err != nil {
					t.Logf("failed to change to temp directory: %v", err)
				}
			},
			cleanupFunc: func() {
				// Return to original directory
				if err := os.Chdir(tmpDir); err != nil {
					t.Logf("failed to return to original directory: %v", err)
				}
			},
		},
	}

	// Save current directory
	originalDir, _ := os.Getwd()
	defer func() {
		if err := os.Chdir(originalDir); err != nil {
			t.Logf("failed to return to original directory: %v", err)
		}
	}()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setupFunc != nil {
				tt.setupFunc()
			}
			if tt.cleanupFunc != nil {
				defer tt.cleanupFunc()
			}

			if tt.startDir != "" {
				err := os.Chdir(tt.startDir)
				if err != nil {
					t.Fatal(err)
				}
			}

			got, err := FindProjectRoot()
			if (err != nil) != tt.wantErr {
				t.Errorf("FindProjectRoot() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Resolve symlinks for comparison
				gotResolved, _ := filepath.EvalSymlinks(got)

				// For current directory test, just check that go.mod exists
				if tt.name == "current_directory" {
					goModPath := filepath.Join(got, "go.mod")
					if _, err := os.Stat(goModPath); os.IsNotExist(err) {
						t.Errorf("go.mod not found at %s", goModPath)
					}
				} else if tt.wantRoot != "" {
					wantResolved, _ := filepath.EvalSymlinks(tt.wantRoot)
					if gotResolved != wantResolved {
						t.Errorf("FindProjectRoot() = %v, want %v", got, tt.wantRoot)
					}
				}
			}
		})
	}
}

func TestFindProjectRoot_MaxDepth(t *testing.T) {
	// Create a very deep directory structure without go.mod
	tmpDir, err := os.MkdirTemp("", "test-deep")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Logf("failed to remove temp directory: %v", err)
		}
	}()

	// Create 12 levels deep (more than maxDepth)
	deepPath := tmpDir
	for i := 0; i < 12; i++ {
		deepPath = filepath.Join(deepPath, "level")
	}
	err = os.MkdirAll(deepPath, 0755)
	if err != nil {
		t.Fatal(err)
	}

	// Save current directory
	originalDir, _ := os.Getwd()
	defer func() {
		if err := os.Chdir(originalDir); err != nil {
			t.Logf("failed to return to original directory: %v", err)
		}
	}()

	// Change to deep directory
	err = os.Chdir(deepPath)
	if err != nil {
		t.Fatal(err)
	}

	// Should fail to find go.mod
	_, err = FindProjectRoot()
	if err == nil {
		t.Error("FindProjectRoot() expected error for deep directory without go.mod")
	}
}
