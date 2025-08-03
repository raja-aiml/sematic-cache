package cmd

import (
	"net"
	"strings"
	"time"
)

// checkTCPConnection checks if a TCP connection can be established to the endpoint
func checkTCPConnection(endpoint string) bool {
	// Parse the endpoint to get host and port
	if strings.HasPrefix(endpoint, "http://") {
		endpoint = strings.TrimPrefix(endpoint, "http://")
	} else if strings.HasPrefix(endpoint, "https://") {
		endpoint = strings.TrimPrefix(endpoint, "https://")
	}
	
	// Try to establish TCP connection
	conn, err := net.DialTimeout("tcp", endpoint, 2*time.Second)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}