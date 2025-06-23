package osutil

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetOS(t *testing.T) {
	os := GetOS()
	
	// Verify that we get a valid OS
	switch runtime.GOOS {
	case "linux":
		assert.Equal(t, Linux, os)
	case "darwin":
		assert.Equal(t, Darwin, os)
	case "windows":
		assert.Equal(t, Windows, os)
	default:
		assert.Equal(t, Unknown, os)
	}
}

func TestGetArch(t *testing.T) {
	arch := GetArch()
	
	// Verify that we get a valid architecture
	switch runtime.GOARCH {
	case "amd64":
		assert.Equal(t, AMD64, arch)
	case "arm64":
		assert.Equal(t, ARM64, arch)
	case "386":
		assert.Equal(t, I386, arch)
	default:
		assert.Equal(t, UnknownArch, arch)
	}
}

func TestGetPlatform(t *testing.T) {
	platform := GetPlatform()
	
	assert.Equal(t, GetOS(), platform.OS)
	assert.Equal(t, GetArch(), platform.Arch)
}

func TestPlatformString(t *testing.T) {
	tests := []struct {
		name     string
		platform Platform
		expected string
	}{
		{
			name: "linux amd64",
			platform: Platform{
				OS:   Linux,
				Arch: AMD64,
			},
			expected: "linux/amd64",
		},
		{
			name: "darwin arm64",
			platform: Platform{
				OS:   Darwin,
				Arch: ARM64,
			},
			expected: "darwin/arm64",
		},
		{
			name: "windows amd64",
			platform: Platform{
				OS:   Windows,
				Arch: AMD64,
			},
			expected: "windows/amd64",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.platform.String())
		})
	}
}

func TestCommandExists(t *testing.T) {
	// Test with a command that should exist on all platforms
	exists := CommandExists("echo")
	assert.True(t, exists, "echo command should exist")

	// Test with a command that shouldn't exist
	exists = CommandExists("this-command-does-not-exist-12345")
	assert.False(t, exists, "non-existent command should not exist")
}

func TestVerifyCommands(t *testing.T) {
	tests := []struct {
		name        string
		commands    []string
		shouldError bool
	}{
		{
			name:        "all commands exist",
			commands:    []string{"echo"},
			shouldError: false,
		},
		{
			name:        "some commands missing",
			commands:    []string{"echo", "this-does-not-exist"},
			shouldError: true,
		},
		{
			name:        "empty list",
			commands:    []string{},
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			missing, err := VerifyCommands(tt.commands)
			if tt.shouldError {
				assert.Error(t, err)
				assert.NotEmpty(t, missing)
			} else {
				assert.NoError(t, err)
				assert.Empty(t, missing)
			}
		})
	}
}

func TestGetExecutableSuffix(t *testing.T) {
	suffix := GetExecutableSuffix()
	
	if GetOS() == Windows {
		assert.Equal(t, ".exe", suffix)
	} else {
		assert.Equal(t, "", suffix)
	}
}

func TestNormalizePath(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:  "forward slashes",
			input: "/path/to/file",
			expected: func() string {
				if GetOS() == Windows {
					return "\\path\\to\\file"
				}
				return "/path/to/file"
			}(),
		},
		{
			name:  "backslashes",
			input: "\\path\\to\\file",
			expected: func() string {
				if GetOS() == Windows {
					return "\\path\\to\\file"
				}
				return "/path/to/file"
			}(),
		},
		{
			name:  "mixed slashes",
			input: "/path\\to/file",
			expected: func() string {
				if GetOS() == Windows {
					return "\\path\\to\\file"
				}
				return "/path/to/file"
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizePath(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetUsername(t *testing.T) {
	username := GetUsername()
	assert.NotEqual(t, "unknown", username, "Should get a valid username")
	assert.NotEmpty(t, username, "Username should not be empty")
}

func TestGetHomeDir(t *testing.T) {
	homeDir := GetHomeDir()
	assert.NotEmpty(t, homeDir, "Home directory should not be empty")
}

func TestIsRoot(t *testing.T) {
	// Just verify the function runs without error
	// We can't guarantee the result as it depends on how tests are run
	_ = IsRoot()
}