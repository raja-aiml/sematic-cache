package kubernetes

import (
	"context"
	"testing"
)

func TestApplyKustomize(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		dir       string
		namespace string
		wantErr   bool
	}{
		{
			name:      "nonexistent_dir",
			dir:       "/nonexistent/dir",
			namespace: "default",
			wantErr:   true,
		},
		{
			name:      "empty_dir",
			dir:       "",
			namespace: "default",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ApplyKustomize(ctx, tt.dir, tt.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("ApplyKustomize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteKustomize(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		dir       string
		namespace string
		wantErr   bool
	}{
		{
			name:      "nonexistent_dir",
			dir:       "/nonexistent/dir",
			namespace: "default",
			wantErr:   false, // DeleteKustomize doesn't return error even for non-existent dirs
		},
		{
			name:      "empty_dir",
			dir:       "",
			namespace: "default",
			wantErr:   false, // DeleteKustomize doesn't return error even for empty dirs
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := DeleteKustomize(ctx, tt.dir, tt.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("DeleteKustomize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Benchmark tests
func BenchmarkApplyKustomize(b *testing.B) {
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ApplyKustomize(ctx, "/nonexistent/dir", "default")
	}
}

func BenchmarkDeleteKustomize(b *testing.B) {
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = DeleteKustomize(ctx, "/nonexistent/dir", "default")
	}
}
