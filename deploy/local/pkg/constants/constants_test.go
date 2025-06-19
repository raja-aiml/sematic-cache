package constants

import "testing"

func TestConstants(t *testing.T) {
	// Test that constants are defined
	if DefaultClusterName == "" {
		t.Error("DefaultClusterName should not be empty")
	}
	
	if DefaultAPIPort == "" {
		t.Error("DefaultAPIPort should not be empty")
	}
	
	if DefaultImageName == "" {
		t.Error("DefaultImageName should not be empty")
	}
	
	if AppNamespace == "" {
		t.Error("AppNamespace should not be empty")
	}
	
	if InfraNamespace == "" {
		t.Error("InfraNamespace should not be empty")
	}
	
	if DefaultTimeout == 0 {
		t.Error("DefaultTimeout should not be zero")
	}
	
	if HTTPPort == "" {
		t.Error("HTTPPort should not be empty")
	}
	
	if AppSecretName == "" {
		t.Error("AppSecretName should not be empty")
	}
}

func TestDefaultValues(t *testing.T) {
	// Test specific values
	if DefaultClusterName != "semantic-cache" {
		t.Errorf("DefaultClusterName = %v, want %v", DefaultClusterName, "semantic-cache")
	}
	
	if DefaultAPIPort != "6550" {
		t.Errorf("DefaultAPIPort = %v, want %v", DefaultAPIPort, "6550")
	}
	
	if AppNamespace != "app" {
		t.Errorf("AppNamespace = %v, want %v", AppNamespace, "app")
	}
	
	if InfraNamespace != "infra" {
		t.Errorf("InfraNamespace = %v, want %v", InfraNamespace, "infra")
	}
}