// Package osutil provides OS and platform detection utilities
package osutil

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// OS represents the operating system type
type OS string

const (
	// Linux operating system
	Linux OS = "linux"
	// Darwin (macOS) operating system
	Darwin OS = "darwin"
	// Windows operating system
	Windows OS = "windows"
	// Unknown operating system
	Unknown OS = "unknown"
)

// Arch represents the CPU architecture
type Arch string

const (
	// AMD64 architecture (x86_64)
	AMD64 Arch = "amd64"
	// ARM64 architecture
	ARM64 Arch = "arm64"
	// I386 architecture
	I386 Arch = "386"
	// UnknownArch for unrecognized architectures
	UnknownArch Arch = "unknown"
)

// GetOS returns the current operating system
func GetOS() OS {
	switch runtime.GOOS {
	case "linux":
		return Linux
	case "darwin":
		return Darwin
	case "windows":
		return Windows
	default:
		return Unknown
	}
}

// GetArch returns the current CPU architecture
func GetArch() Arch {
	switch runtime.GOARCH {
	case "amd64":
		return AMD64
	case "arm64":
		return ARM64
	case "386":
		return I386
	default:
		return UnknownArch
	}
}

// Platform represents OS and architecture combination
type Platform struct {
	OS   OS
	Arch Arch
}

// GetPlatform returns the current platform information
func GetPlatform() Platform {
	return Platform{
		OS:   GetOS(),
		Arch: GetArch(),
	}
}

// String returns the platform as a string (e.g., "linux/amd64")
func (p Platform) String() string {
	return fmt.Sprintf("%s/%s", p.OS, p.Arch)
}

// CommandExists checks if a command is available in PATH
func CommandExists(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

// VerifyCommands checks if all required commands exist
func VerifyCommands(commands []string) ([]string, error) {
	var missing []string

	for _, cmd := range commands {
		if !CommandExists(cmd) {
			missing = append(missing, cmd)
		}
	}

	if len(missing) > 0 {
		return missing, fmt.Errorf("missing required commands: %s", strings.Join(missing, ", "))
	}

	return nil, nil
}

// IsRoot checks if the current process is running as root/admin
func IsRoot() bool {
	switch GetOS() {
	case Linux, Darwin:
		// Check if running as root on Unix-like systems
		cmd := exec.Command("id", "-u")
		output, err := cmd.Output()
		if err != nil {
			return false
		}
		return strings.TrimSpace(string(output)) == "0"
	case Windows:
		// On Windows, check if running as Administrator
		// This is a simplified check - for production use, you'd want to use Windows API
		cmd := exec.Command("net", "session")
		err := cmd.Run()
		return err == nil
	default:
		return false
	}
}

// GetUsername returns the current username
func GetUsername() string {
	switch GetOS() {
	case Linux, Darwin:
		cmd := exec.Command("whoami")
		output, err := cmd.Output()
		if err != nil {
			return "unknown"
		}
		return strings.TrimSpace(string(output))
	case Windows:
		cmd := exec.Command("echo", "%USERNAME%")
		output, err := cmd.Output()
		if err != nil {
			return "unknown"
		}
		return strings.TrimSpace(string(output))
	default:
		return "unknown"
	}
}

// GetHomeDir returns the user's home directory
func GetHomeDir() string {
	switch GetOS() {
	case Linux, Darwin:
		cmd := exec.Command("sh", "-c", "echo $HOME")
		output, err := cmd.Output()
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(output))
	case Windows:
		cmd := exec.Command("cmd", "/c", "echo %USERPROFILE%")
		output, err := cmd.Output()
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(output))
	default:
		return ""
	}
}

// GetExecutableSuffix returns the executable suffix for the current OS
func GetExecutableSuffix() string {
	if GetOS() == Windows {
		return ".exe"
	}
	return ""
}

// NormalizePath converts a path to the OS-specific format
func NormalizePath(path string) string {
	if GetOS() == Windows {
		return strings.ReplaceAll(path, "/", "\\")
	}
	return strings.ReplaceAll(path, "\\", "/")
}
