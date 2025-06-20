package kubernetes

import (
	"context"
	"testing"
)

func TestNewLogRetriever(t *testing.T) {
	// Mock client
	client := &Client{}

	lr := NewLogRetriever(client)

	if lr == nil {
		t.Fatal("NewLogRetriever returned nil")
	}

	if lr.client != client {
		t.Error("NewLogRetriever() client mismatch")
	}

	if lr.logger == nil {
		t.Error("NewLogRetriever() logger is nil")
	}
}

func TestLogRetriever_GetLogs(t *testing.T) {
	// This will fail without a real k8s client
	lr := NewLogRetriever(nil)
	ctx := context.Background()

	tests := []struct {
		name    string
		opts    LogOptions
		wantErr bool
	}{
		{
			name: "basic_options",
			opts: LogOptions{
				Namespace:     "default",
				LabelSelector: "app=test",
				TailLines:     100,
				ShowPodName:   true,
			},
			wantErr: true, // Will fail without client
		},
		{
			name: "empty_namespace",
			opts: LogOptions{
				Namespace:     "",
				LabelSelector: "app=test",
			},
			wantErr: true,
		},
		{
			name: "no_selector",
			opts: LogOptions{
				Namespace: "default",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := lr.GetLogs(ctx, tt.opts)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetLogs() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLogOptions(t *testing.T) {
	// Test LogOptions struct
	opts := LogOptions{
		Namespace:     "test-ns",
		LabelSelector: "app=test",
		TailLines:     50,
		ShowPodName:   true,
		Follow:        false,
		PodName:       "test-pod",
	}

	if opts.Namespace != "test-ns" {
		t.Errorf("LogOptions.Namespace = %v", opts.Namespace)
	}

	if opts.LabelSelector != "app=test" {
		t.Errorf("LogOptions.LabelSelector = %v", opts.LabelSelector)
	}

	if opts.TailLines != 50 {
		t.Errorf("LogOptions.TailLines = %v", opts.TailLines)
	}

	if !opts.ShowPodName {
		t.Error("LogOptions.ShowPodName should be true")
	}

	if opts.Follow {
		t.Error("LogOptions.Follow should be false")
	}

	if opts.PodName != "test-pod" {
		t.Errorf("LogOptions.PodName = %v", opts.PodName)
	}
}

func TestLogRetriever_StreamLogs(t *testing.T) {
	lr := NewLogRetriever(nil)
	ctx := context.Background()

	tests := []struct {
		name    string
		opts    LogOptions
		wantErr bool
	}{
		{
			name: "follow_false",
			opts: LogOptions{
				Namespace:     "default",
				LabelSelector: "app=test",
				Follow:        false,
			},
			wantErr: true, // Will fail without client
		},
		{
			name: "follow_true",
			opts: LogOptions{
				Namespace:     "default",
				LabelSelector: "app=test",
				Follow:        true,
			},
			wantErr: false, // Just shows warning
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := lr.StreamLogs(ctx, tt.opts)
			if (err != nil) != tt.wantErr {
				t.Errorf("StreamLogs() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLogRetriever_GetPodStatus(t *testing.T) {
	lr := NewLogRetriever(nil)
	ctx := context.Background()

	// This will fail without a real k8s client
	_, err := lr.GetPodStatus(ctx, "default", "app=test")
	if err == nil {
		t.Error("GetPodStatus() expected error without client")
	}
}

func TestPodStatus(t *testing.T) {
	// Test PodStatus struct
	status := PodStatus{
		Name:      "test-pod",
		Namespace: "default",
		Phase:     "Running",
		Ready:     true,
		Restarts:  2,
	}

	if status.Name != "test-pod" {
		t.Errorf("PodStatus.Name = %v", status.Name)
	}

	if status.Namespace != "default" {
		t.Errorf("PodStatus.Namespace = %v", status.Namespace)
	}

	if status.Phase != "Running" {
		t.Errorf("PodStatus.Phase = %v", status.Phase)
	}

	if !status.Ready {
		t.Error("PodStatus.Ready should be true")
	}

	if status.Restarts != 2 {
		t.Errorf("PodStatus.Restarts = %v", status.Restarts)
	}
}

// Benchmark tests
func BenchmarkNewLogRetriever(b *testing.B) {
	client := &Client{}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = NewLogRetriever(client)
	}
}
