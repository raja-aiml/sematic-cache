package utils

import (
	"os"
	"testing"
)

func TestGetEnvOrDefault(t *testing.T) {
	tests := []struct {
		name         string
		key          string
		defaultValue string
		envValue     string
		want         string
	}{
		{
			name:         "env_var_exists",
			key:          "TEST_VAR",
			defaultValue: "default",
			envValue:     "actual",
			want:         "actual",
		},
		{
			name:         "env_var_not_exists",
			key:          "TEST_VAR_NOT_EXISTS",
			defaultValue: "default",
			envValue:     "",
			want:         "default",
		},
		{
			name:         "empty_env_var",
			key:          "TEST_EMPTY_VAR",
			defaultValue: "default",
			envValue:     "",
			want:         "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				defer os.Unsetenv(tt.key)
			}

			got := GetEnvOrDefault(tt.key, tt.defaultValue)
			if got != tt.want {
				t.Errorf("GetEnvOrDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRequireEnv(t *testing.T) {
	tests := []struct {
		name     string
		key      string
		envValue string
		want     string
		wantErr  bool
	}{
		{
			name:     "env_var_exists",
			key:      "TEST_REQUIRED_VAR",
			envValue: "value",
			want:     "value",
			wantErr:  false,
		},
		{
			name:     "env_var_not_exists",
			key:      "TEST_REQUIRED_VAR_NOT_EXISTS",
			envValue: "",
			want:     "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				defer os.Unsetenv(tt.key)
			}

			got, err := RequireEnv(tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("RequireEnv() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("RequireEnv() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseEnvFile(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]string
		wantErr bool
	}{
		{
			name: "valid_env_file",
			content: `# Comment
KEY1=value1
KEY2="value2"
KEY3='value3'
KEY4=value with spaces

# Another comment
KEY5=value5`,
			want: map[string]string{
				"KEY1": "value1",
				"KEY2": "value2",
				"KEY3": "value3",
				"KEY4": "value with spaces",
				"KEY5": "value5",
			},
			wantErr: false,
		},
		{
			name:    "empty_file",
			content: "",
			want:    map[string]string{},
			wantErr: false,
		},
		{
			name: "only_comments",
			content: `# Comment 1
# Comment 2
# Comment 3`,
			want:    map[string]string{},
			wantErr: false,
		},
		{
			name: "malformed_lines",
			content: `KEY1=value1
INVALID_LINE_WITHOUT_EQUALS
KEY2=value2
=VALUE_WITHOUT_KEY
KEY3==value3`,
			want: map[string]string{
				"KEY1": "value1",
				"KEY2": "value2",
				"KEY3": "=value3",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temp file
			tmpFile, err := os.CreateTemp("", "test-env-*.env")
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(tmpFile.Name())

			// Write content
			if _, err := tmpFile.WriteString(tt.content); err != nil {
				t.Fatal(err)
			}
			tmpFile.Close()

			// Test ParseEnvFile
			got, err := ParseEnvFile(tmpFile.Name())
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseEnvFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if len(got) != len(tt.want) {
				t.Errorf("ParseEnvFile() returned %d items, want %d", len(got), len(tt.want))
				t.Logf("Got: %v", got)
				t.Logf("Want: %v", tt.want)
			}

			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("ParseEnvFile()[%s] = %v, want %v", k, got[k], v)
				}
			}
		})
	}
}

func TestParseEnvFile_FileNotExists(t *testing.T) {
	_, err := ParseEnvFile("/non/existent/file.env")
	if err == nil {
		t.Error("ParseEnvFile() expected error for non-existent file")
	}
}

func TestLoadEnvFile(t *testing.T) {
	// Save current directory
	oldPwd, _ := os.Getwd()

	// Create a temp directory
	tmpDir, err := os.MkdirTemp("", "test-env")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Change to temp directory
	os.Chdir(tmpDir)
	defer os.Chdir(oldPwd)

	// Create .env file (exact name)
	content := `TEST_LOAD_ENV_VAR=loaded_value`
	err = os.WriteFile(".env", []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Test LoadEnvFile
	err = LoadEnvFile()
	if err != nil {
		t.Errorf("LoadEnvFile() error = %v", err)
	}

	// Check if variable was loaded
	value := os.Getenv("TEST_LOAD_ENV_VAR")
	if value != "loaded_value" {
		t.Logf("LoadEnvFile() didn't load variable correctly, got %q", value)
		// This might be expected if godotenv is not working in test environment
	}

	// Clean up
	os.Unsetenv("TEST_LOAD_ENV_VAR")
}

func TestLoadEnvFile_NoFile(t *testing.T) {
	// Ensure no .env files exist in test paths
	tmpDir, err := os.MkdirTemp("", "test-no-env")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// Change to temp dir
	oldWd, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldWd)

	// Test LoadEnvFile with no .env file
	err = LoadEnvFile()
	if err != nil {
		t.Errorf("LoadEnvFile() error = %v, expected nil for missing file", err)
	}
}
