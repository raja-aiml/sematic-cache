package constants

import (
	"testing"
	"time"
)

func TestConstants(t *testing.T) {
	tests := []struct {
		name     string
		constant interface{}
		expected interface{}
	}{
		// Cluster defaults
		{
			name:     "default_cluster_name",
			constant: DefaultClusterName,
			expected: "semantic-cache",
		},
		{
			name:     "default_api_port",
			constant: DefaultAPIPort,
			expected: "6550",
		},
		// Container defaults
		{
			name:     "default_image_name",
			constant: DefaultImageName,
			expected: "semantic-cache:local",
		},
		// Namespace defaults
		{
			name:     "app_namespace",
			constant: AppNamespace,
			expected: "app",
		},
		{
			name:     "infra_namespace",
			constant: InfraNamespace,
			expected: "infra",
		},
		// Timeout defaults
		{
			name:     "default_timeout",
			constant: DefaultTimeout,
			expected: 5 * time.Minute,
		},
		{
			name:     "default_build_timeout",
			constant: DefaultBuildTimeout,
			expected: 10 * time.Minute,
		},
		{
			name:     "default_http_timeout",
			constant: DefaultHTTPTimeout,
			expected: 5 * time.Second,
		},
		{
			name:     "default_cluster_timeout",
			constant: DefaultClusterTimeout,
			expected: 300 * time.Second,
		},
		// Port mappings
		{
			name:     "http_port",
			constant: HTTPPort,
			expected: "8080:80",
		},
		{
			name:     "https_port",
			constant: HTTPSPort,
			expected: "8443:443",
		},
		// Secret names
		{
			name:     "app_secret_name",
			constant: AppSecretName,
			expected: "semantic-cache-secrets",
		},
		// Database URL
		{
			name:     "default_database_url",
			constant: DefaultDatabaseURL,
			expected: "postgres://postgres:postgres@postgres.infra.svc.cluster.local:5432/semantic_cache?sslmode=disable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.constant != tt.expected {
				t.Errorf("Constant %s = %v, want %v", tt.name, tt.constant, tt.expected)
			}
		})
	}
}

func TestTimeoutValues(t *testing.T) {
	// Verify timeout relationships
	if DefaultTimeout >= DefaultBuildTimeout {
		t.Error("DefaultTimeout should be less than DefaultBuildTimeout")
	}

	if DefaultHTTPTimeout >= DefaultTimeout {
		t.Error("DefaultHTTPTimeout should be less than DefaultTimeout")
	}

	if DefaultClusterTimeout < DefaultTimeout {
		t.Error("DefaultClusterTimeout should be greater than or equal to DefaultTimeout")
	}
}

func TestPortFormats(t *testing.T) {
	// Test that port mappings follow host:container format
	ports := []struct {
		name string
		port string
	}{
		{"HTTPPort", HTTPPort},
		{"HTTPSPort", HTTPSPort},
	}

	for _, p := range ports {
		t.Run(p.name, func(t *testing.T) {
			if len(p.port) == 0 {
				t.Errorf("%s is empty", p.name)
			}
			// Simple check for colon separator
			if !contains(p.port, ":") {
				t.Errorf("%s does not follow host:container format: %s", p.name, p.port)
			}
		})
	}
}

func TestNamespaceNaming(t *testing.T) {
	// Verify namespaces follow Kubernetes naming conventions
	namespaces := []struct {
		name      string
		namespace string
	}{
		{"AppNamespace", AppNamespace},
		{"InfraNamespace", InfraNamespace},
	}

	for _, ns := range namespaces {
		t.Run(ns.name, func(t *testing.T) {
			if len(ns.namespace) == 0 {
				t.Errorf("%s is empty", ns.name)
			}
			// Check lowercase (Kubernetes requirement)
			if ns.namespace != toLowerCase(ns.namespace) {
				t.Errorf("%s should be lowercase: %s", ns.name, ns.namespace)
			}
		})
	}
}

// Helper functions
func contains(s, substr string) bool {
	for i := 0; i < len(s); i++ {
		if i+len(substr) <= len(s) && s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func toLowerCase(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			result[i] = c + 32
		} else {
			result[i] = c
		}
	}
	return string(result)
}