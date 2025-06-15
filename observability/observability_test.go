package observability

import (
	"context"
	"testing"
)

func TestInit(t *testing.T) {
	shutdown, err := Init(context.Background(), "test-service", "http://localhost:14268/api/traces")
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown failed: %v", err)
	}
}
