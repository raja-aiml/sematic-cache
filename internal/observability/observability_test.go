package observability

import (
	"context"
	"testing"
	"time"
)

func TestInit(t *testing.T) {
	// Test with OTel Collector endpoint
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	shutdown, err := Init(ctx, "test-service", "localhost:4317")
	if err != nil {
		// If connection fails, skip the test (OTel Collector might not be running)
		t.Skipf("Skipping test - OTel Collector not available: %v", err)
	}

	// Verify shutdown works
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown failed: %v", err)
	}
}
