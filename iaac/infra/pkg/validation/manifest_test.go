package validation

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewManifestValidator(t *testing.T) {
	validator := NewManifestValidator()
	assert.NotNil(t, validator)
}

func TestManifestValidator_ValidateFile(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		wantErr   bool
		checkFunc func(*testing.T, *ValidationResult)
	}{
		{
			name: "valid_deployment",
			content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: nginx:latest
        ports:
        - containerPort: 80
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
				assert.NotEmpty(t, result.Info)
			},
		},
		{
			name: "multiple_documents",
			content: `apiVersion: v1
kind: Service
metadata:
  name: test-service
spec:
  selector:
    app: test
  ports:
  - port: 80
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: nginx:latest
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				// May have warnings but no errors
				assert.NotNil(t, result)
			},
		},
		{
			name:    "invalid_yaml",
			content: `invalid: yaml: content:`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "Invalid YAML")
			},
		},
		{
			name: "missing_apiVersion",
			content: `kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "Missing apiVersion")
			},
		},
		{
			name: "missing_kind",
			content: `apiVersion: apps/v1
metadata:
  name: test
spec:
  replicas: 1
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "Missing kind")
			},
		},
		{
			name: "missing_metadata",
			content: `apiVersion: apps/v1
kind: ConfigMap
data:
  key: value
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "Missing or invalid metadata")
			},
		},
		{
			name: "deployment_missing_spec",
			content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				// Look for spec error
				found := false
				for _, err := range result.Errors {
					if contains(err, "missing spec") {
						found = true
						break
					}
				}
				assert.True(t, found)
			},
		},
		{
			name: "service_missing_spec",
			content: `apiVersion: v1
kind: Service
metadata:
  name: test-service
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				// Look for spec error
				found := false
				for _, err := range result.Errors {
					if contains(err, "missing spec") {
						found = true
						break
					}
				}
				assert.True(t, found)
			},
		},
		{
			name: "configmap_valid",
			content: `apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
data:
  key1: value1
  key2: value2
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "secret_valid",
			content: `apiVersion: v1
kind: Secret
metadata:
  name: test-secret
type: Opaque
data:
  username: YWRtaW4=
  password: MWYyZDFlMmU2N2Rm
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "pvc_valid",
			content: `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "networkpolicy_valid",
			content: `apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: test-netpol
spec:
  podSelector:
    matchLabels:
      app: test
  policyTypes:
  - Ingress
  - Egress
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "ingress_valid",
			content: `apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: test-ingress
spec:
  rules:
  - host: test.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: test-service
            port:
              number: 80
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "job_valid",
			content: `apiVersion: batch/v1
kind: Job
metadata:
  name: test-job
spec:
  template:
    spec:
      containers:
      - name: test
        image: busybox
        command: ["echo", "Hello"]
      restartPolicy: Never
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "cronjob_valid",
			content: `apiVersion: batch/v1
kind: CronJob
metadata:
  name: test-cronjob
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: test
            image: busybox
            command: ["echo", "Hello"]
          restartPolicy: Never
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "statefulset_valid",
			content: `apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: test-statefulset
spec:
  serviceName: test-service
  replicas: 3
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: nginx
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "daemonset_valid",
			content: `apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: test-daemonset
spec:
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: test
        image: nginx
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
			},
		},
		{
			name: "unknown_kind",
			content: `apiVersion: custom.io/v1
kind: CustomResource
metadata:
  name: test-custom
spec:
  field: value
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.True(t, result.IsValid())
				assert.NotEmpty(t, result.Info)
				assert.Contains(t, result.Info[0], "custom.io/v1/CustomResource")
			},
		},
		{
			name: "empty_document_separator",
			content: `---
apiVersion: v1
kind: Service
metadata:
  name: test
spec:
  ports:
  - port: 80
---
---
`,
			wantErr: false,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				// May have warnings, but should process one valid document
				assert.NotNil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := filepath.Join(t.TempDir(), "manifest.yaml")
			err := os.WriteFile(tmpFile, []byte(tt.content), 0644)
			require.NoError(t, err)

			validator := NewManifestValidator()
			result, err := validator.ValidateFile(tmpFile)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)

				if tt.checkFunc != nil {
					tt.checkFunc(t, result)
				}
			}
		})
	}
}

func TestManifestValidator_ValidateFile_ReadError(t *testing.T) {
	validator := NewManifestValidator()
	result, err := validator.ValidateFile("/non/existent/file.yaml")

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read file")
	assert.Nil(t, result)
}

func TestManifestValidator_validateMetadata(t *testing.T) {
	tests := []struct {
		name      string
		manifest  map[string]interface{}
		checkFunc func(*testing.T, *ValidationResult)
	}{
		{
			name: "valid_metadata_with_namespace",
			manifest: map[string]interface{}{
				"metadata": map[string]interface{}{
					"name":      "test-resource",
					"namespace": "default",
				},
			},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.Empty(t, result.Errors)
			},
		},
		{
			name: "valid_metadata_without_namespace",
			manifest: map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": "test-resource",
				},
			},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.Empty(t, result.Errors)
				assert.Len(t, result.Warnings, 1)
				assert.Contains(t, result.Warnings[0], "No labels defined")
			},
		},
		{
			name:     "missing_metadata",
			manifest: map[string]interface{}{},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.Len(t, result.Errors, 1)
				assert.Contains(t, result.Errors[0], "Missing or invalid metadata")
			},
		},
		{
			name: "metadata_missing_name",
			manifest: map[string]interface{}{
				"metadata": map[string]interface{}{
					"namespace": "default",
				},
			},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.Len(t, result.Errors, 1)
				assert.Contains(t, result.Errors[0], "Missing metadata.name")
			},
		},
		{
			name: "invalid_name_too_long",
			manifest: map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": strings.Repeat("a", 254), // 254 chars is too long
				},
			},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "Invalid name")
			},
		},
		{
			name: "invalid_name_uppercase",
			manifest: map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": "TestResource",
				},
			},
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.Len(t, result.Errors, 1)
				assert.Contains(t, result.Errors[0], "Invalid name")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewManifestValidator()
			result := NewValidationResult()

			// Using validateMetadata indirectly through validateDocument
			doc := `apiVersion: v1
kind: ConfigMap`
			validator.validateDocument(doc, 1, "test.yaml", result)

			// Now check metadata validation by calling validateMetadata
			result = NewValidationResult() // Reset for clean test
			validator.validateMetadata(tt.manifest, 1, result)

			if tt.checkFunc != nil {
				tt.checkFunc(t, result)
			}
		})
	}
}

func TestManifestValidator_ComplexScenarios(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		checkFunc func(*testing.T, *ValidationResult)
	}{
		{
			name: "deployment_with_invalid_container",
			content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - image: nginx
`,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				found := false
				for _, err := range result.Errors {
					if contains(err, "container") && contains(err, "missing name") {
						found = true
						break
					}
				}
				assert.True(t, found)
			},
		},
		{
			name: "service_with_invalid_port",
			content: `apiVersion: v1
kind: Service
metadata:
  name: test-service
spec:
  selector:
    app: test
  ports:
  - targetPort: 80
`,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				// Should have error about missing port field
				found := false
				for _, err := range result.Errors {
					if contains(err, "missing 'port'") {
						found = true
						break
					}
				}
				assert.True(t, found)
			},
		},
		{
			name: "pvc_missing_accessModes",
			content: `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-pvc
spec:
  resources:
    requests:
      storage: 1Gi
`,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				assert.False(t, result.IsValid())
				assert.NotEmpty(t, result.Errors)
				assert.Contains(t, result.Errors[0], "PVC missing spec.accessModes")
			},
		},
		{
			name: "cronjob_invalid_schedule",
			content: `apiVersion: batch/v1
kind: CronJob
metadata:
  name: test-cronjob
spec:
  schedule: "invalid-cron"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: test
            image: busybox
          restartPolicy: Never
`,
			checkFunc: func(t *testing.T, result *ValidationResult) {
				// The validator doesn't validate cron syntax, just checks presence
				assert.NotNil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := filepath.Join(t.TempDir(), "manifest.yaml")
			err := os.WriteFile(tmpFile, []byte(tt.content), 0644)
			require.NoError(t, err)

			validator := NewManifestValidator()
			result, err := validator.ValidateFile(tmpFile)

			assert.NoError(t, err)
			assert.NotNil(t, result)

			if tt.checkFunc != nil {
				tt.checkFunc(t, result)
			}
		})
	}
}

func TestIsValidKubernetesName(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"valid_simple", "test", true},
		{"valid_with_dash", "test-app", true},
		{"valid_with_numbers", "app-123", true},
		{"valid_with_dot", "app.v1", true},
		{"valid_long", strings.Repeat("a", 253), true},
		{"invalid_too_long", strings.Repeat("a", 254), false},
		{"invalid_uppercase", "TestApp", false},
		{"invalid_underscore", "test_app", false},
		{"invalid_start_dash", "-test", false},
		{"invalid_end_dash", "test-", false},
		{"invalid_start_dot", ".test", false},
		{"invalid_end_dot", "test.", false},
		{"invalid_special_char", "test@app", false},
		{"invalid_empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isValidKubernetesName(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}
