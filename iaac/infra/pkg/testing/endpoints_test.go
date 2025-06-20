package testing

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewEndpointTester(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
	}{
		{
			name:    "http_url",
			baseURL: "http://localhost:8080",
		},
		{
			name:    "https_url",
			baseURL: "https://example.com",
		},
		{
			name:    "empty_url",
			baseURL: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			et := NewEndpointTester(tt.baseURL)

			if et == nil {
				t.Fatal("NewEndpointTester returned nil")
			}

			if et.baseURL != tt.baseURL {
				t.Errorf("NewEndpointTester() baseURL = %v, want %v", et.baseURL, tt.baseURL)
			}

			if et.client == nil {
				t.Error("NewEndpointTester() client is nil")
			}

			if et.logger == nil {
				t.Error("NewEndpointTester() logger is nil")
			}
		})
	}
}

func TestEndpointTester_TestEndpoint(t *testing.T) {
	// Create test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		case "/error":
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"internal error"}`))
		case "/json":
			var body map[string]interface{}
			_ = json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(body); err != nil {
				t.Logf("failed to encode response: %v", err)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	et := NewEndpointTester(ts.URL)

	tests := []struct {
		name        string
		method      string
		path        string
		body        interface{}
		wantStatus  int
		wantSuccess bool
	}{
		{
			name:        "health_endpoint",
			method:      "GET",
			path:        "/health",
			body:        nil,
			wantStatus:  http.StatusOK,
			wantSuccess: true,
		},
		{
			name:        "error_endpoint",
			method:      "GET",
			path:        "/error",
			body:        nil,
			wantStatus:  http.StatusInternalServerError,
			wantSuccess: false,
		},
		{
			name:        "not_found",
			method:      "GET",
			path:        "/notfound",
			body:        nil,
			wantStatus:  http.StatusNotFound,
			wantSuccess: false,
		},
		{
			name:        "post_with_json",
			method:      "POST",
			path:        "/json",
			body:        map[string]string{"key": "value"},
			wantStatus:  http.StatusOK,
			wantSuccess: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := et.TestEndpoint(tt.method, tt.path, tt.body)

			if result.StatusCode != tt.wantStatus {
				t.Errorf("TestEndpoint() StatusCode = %v, want %v", result.StatusCode, tt.wantStatus)
			}

			if result.Success != tt.wantSuccess {
				t.Errorf("TestEndpoint() Success = %v, want %v", result.Success, tt.wantSuccess)
			}

			if result.Method != tt.method {
				t.Errorf("TestEndpoint() Method = %v, want %v", result.Method, tt.method)
			}

			if result.Endpoint != ts.URL+tt.path {
				t.Errorf("TestEndpoint() Endpoint = %v, want %v", result.Endpoint, ts.URL+tt.path)
			}
		})
	}
}

func TestEndpointTester_TestStandardEndpoints(t *testing.T) {
	// Create test server with standard endpoints
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"healthy"}`))
		case "/metrics":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("# HELP app_info\n# TYPE app_info gauge\napp_info 1\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	et := NewEndpointTester(ts.URL)

	results, err := et.TestStandardEndpoints()
	if err != nil {
		t.Fatalf("TestStandardEndpoints() error = %v", err)
	}

	if len(results) != 2 {
		t.Errorf("TestStandardEndpoints() returned %d results, want 2", len(results))
	}

	// Check that both standard endpoints were tested
	foundHealth := false
	foundMetrics := false

	for _, result := range results {
		if result.Method == "GET" && result.Endpoint == ts.URL+"/health" {
			foundHealth = true
			if !result.Success {
				t.Error("Health endpoint should be successful")
			}
		}
		if result.Method == "GET" && result.Endpoint == ts.URL+"/metrics" {
			foundMetrics = true
			if !result.Success {
				t.Error("Metrics endpoint should be successful")
			}
		}
	}

	if !foundHealth {
		t.Error("Health endpoint not tested")
	}
	if !foundMetrics {
		t.Error("Metrics endpoint not tested")
	}
}

func TestEndpointTester_TestCacheOperations(t *testing.T) {
	// Create test server with cache endpoints
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/set":
			if r.Method == "POST" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"success":true}`))
			} else {
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		case "/get":
			if r.Method == "GET" && r.URL.Query().Get("key") == "test-key" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"value":"test-value"}`))
			} else {
				w.WriteHeader(http.StatusNotFound)
			}
		case "/query":
			if r.Method == "POST" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"results":[]}`))
			} else {
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	et := NewEndpointTester(ts.URL)

	err := et.TestCacheOperations()
	if err != nil {
		t.Errorf("TestCacheOperations() error = %v", err)
	}
}

func TestEndpointTester_TestEndpoint_InvalidURL(t *testing.T) {
	et := NewEndpointTester("http://[::1]:namedport") // Invalid URL

	result := et.TestEndpoint("GET", "/test", nil)

	if result.Error == nil {
		t.Error("TestEndpoint() expected error for invalid URL")
	}

	if result.Success {
		t.Error("TestEndpoint() Success should be false for invalid URL")
	}
}

func TestEndpointTester_TestEndpoint_MarshalError(t *testing.T) {
	et := NewEndpointTester("http://localhost")

	// Create a type that can't be marshaled to JSON
	type invalidType struct {
		Channel chan int
	}

	result := et.TestEndpoint("POST", "/test", invalidType{Channel: make(chan int)})

	if result.Error == nil {
		t.Error("TestEndpoint() expected error for unmarshalable body")
	}

	if result.Success {
		t.Error("TestEndpoint() Success should be false for marshal error")
	}
}

func TestTestResult(t *testing.T) {
	// Test TestResult struct
	result := TestResult{
		Endpoint:   "http://localhost/health",
		Method:     "GET",
		StatusCode: 200,
		Success:    true,
		Error:      nil,
		Body:       `{"status":"ok"}`,
	}

	// Verify fields
	if result.Endpoint != "http://localhost/health" {
		t.Errorf("TestResult.Endpoint = %v, want %v", result.Endpoint, "http://localhost/health")
	}

	if result.Method != "GET" {
		t.Errorf("TestResult.Method = %v, want %v", result.Method, "GET")
	}

	if result.StatusCode != 200 {
		t.Errorf("TestResult.StatusCode = %v, want %v", result.StatusCode, 200)
	}

	if !result.Success {
		t.Error("TestResult.Success should be true")
	}

	if result.Error != nil {
		t.Errorf("TestResult.Error = %v, want nil", result.Error)
	}

	if result.Body != `{"status":"ok"}` {
		t.Errorf("TestResult.Body = %v, want %v", result.Body, `{"status":"ok"}`)
	}
}

// Benchmark tests
func BenchmarkNewEndpointTester(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewEndpointTester("http://localhost:8080")
	}
}

func BenchmarkTestEndpoint(b *testing.B) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer ts.Close()

	et := NewEndpointTester(ts.URL)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = et.TestEndpoint("GET", "/health", nil)
	}
}
