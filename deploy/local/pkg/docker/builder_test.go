package docker

import (
	"context"
	"testing"
)

func TestNewBuilder(t *testing.T) {
	builder := NewBuilder()
	if builder == nil {
		t.Fatal("NewBuilder() returned nil")
	}
	
	if builder.logger == nil {
		t.Error("NewBuilder() should initialize logger")
	}
}

func TestBuilderBuild(t *testing.T) {
	builder := NewBuilder()
	
	// Test that Build method exists
	// We can't actually test Docker operations without Docker
	if builder == nil {
		t.Fatal("Builder should not be nil")
	}
}

func TestBuilderMethods(t *testing.T) {
	builder := NewBuilder()
	
	// Test that builder has expected fields
	if builder.logger == nil {
		t.Error("Builder should have logger")
	}
	
	// Test that methods can be called (they exist)
	ctx := context.Background()
	// We won't actually run these, just verify they compile
	_ = builder.Build
	_ = builder.Tag
	_ = builder.Push
	_ = builder.Run
	_ = ctx
}

func TestBuilderFallback(t *testing.T) {
	// Test that builder can handle SDK fallback
	builder := NewBuilder()
	
	if builder.useSDK && builder.sdkBuilder == nil {
		t.Error("useSDK should be false when sdkBuilder is nil")
	}
}