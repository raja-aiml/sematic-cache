package utils

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

type ExecOptions struct {
	Dir     string
	Env     []string
	Timeout time.Duration
	Silent  bool
}

func RunCommand(ctx context.Context, name string, args []string, opts *ExecOptions) (string, error) {
	if opts == nil {
		opts = &ExecOptions{}
	}

	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, name, args...)

	if opts.Dir != "" {
		cmd.Dir = opts.Dir
	}

	cmd.Env = append(os.Environ(), opts.Env...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if !opts.Silent {
		logger := NewLogger("exec")
		logger.Debug("Running: %s %s", name, strings.Join(args, " "))
	}

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("command failed: %w\nstderr: %s", err, stderr.String())
	}

	return stdout.String(), nil
}

func CommandExists(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func RunShellCommand(ctx context.Context, command string, opts *ExecOptions) (string, error) {
	return RunCommand(ctx, "sh", []string{"-c", command}, opts)
}
