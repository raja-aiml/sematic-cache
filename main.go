// Package main provides a convenience wrapper to run the server from the root directory.
// The actual server implementation is in cmd/server/main.go
package main

import (
	"log"
	"os"
	"os/exec"
)

func main() {
	// Build the server if needed
	log.Println("Starting semantic cache server...")
	
	// Run the server from cmd/server
	cmd := exec.Command("go", "run", "./cmd/server", "-config", "config.yml")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	
	// Pass through environment variables
	cmd.Env = os.Environ()
	
	// Run the server
	if err := cmd.Run(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
