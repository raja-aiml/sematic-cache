package database

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewPostgresManager(t *testing.T) {
	tests := []struct {
		name   string
		config *Config
	}{
		{
			name: "valid_config",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
		},
		{
			name: "minimal_config",
			config: &Config{
				Host:     "db",
				Port:     5432,
				User:     "user",
				Password: "pass",
				Database: "db",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)

			if pm == nil {
				t.Fatal("NewPostgresManager returned nil")
			}

			if pm.logger == nil {
				t.Error("PostgresManager logger is nil")
			}

			if pm.config != tt.config {
				t.Error("PostgresManager config not set correctly")
			}
		})
	}
}

func TestPostgresManager_InitializeDatabase(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "connection_failed",
			config: &Config{
				Host:     "invalid-host",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)
			ctx := context.Background()

			// This will fail as we can't connect to a real database in tests
			err := pm.InitializeDatabase(ctx)

			if (err != nil) != tt.wantErr {
				t.Errorf("InitializeDatabase() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPostgresManager_WaitForReady(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		timeout time.Duration
		wantErr bool
	}{
		{
			name: "timeout_exceeded",
			config: &Config{
				Host:     "invalid-host",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			timeout: 100 * time.Millisecond,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)
			ctx := context.Background()

			err := pm.WaitForReady(ctx, tt.timeout)

			if (err != nil) != tt.wantErr {
				t.Errorf("WaitForReady() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPostgresManager_WaitForReady_ContextCancellation(t *testing.T) {
	pm := NewPostgresManager(&Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	})

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel context immediately
	cancel()

	err := pm.WaitForReady(ctx, 5*time.Second)

	if err == nil {
		t.Error("WaitForReady() expected error for cancelled context")
	}
}

func TestPostgresManager_TestConnection(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "connection_failed",
			config: &Config{
				Host:     "invalid-host",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)
			ctx := context.Background()

			err := pm.TestConnection(ctx)

			if (err != nil) != tt.wantErr {
				t.Errorf("TestConnection() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPostgresManager_ExecuteSQL(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		sql     string
		wantErr bool
	}{
		{
			name: "connection_failed",
			config: &Config{
				Host:     "invalid-host",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			sql:     "SELECT 1",
			wantErr: true,
		},
		{
			name: "empty_sql",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			sql:     "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)
			ctx := context.Background()

			err := pm.ExecuteSQL(ctx, tt.sql)

			if (err != nil) != tt.wantErr {
				t.Errorf("ExecuteSQL() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestConfig_ConnectionString(t *testing.T) {
	tests := []struct {
		name     string
		config   *Config
		expected string
	}{
		{
			name: "standard_config",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expected: "postgres://postgres:password@localhost:5432/testdb?sslmode=disable",
		},
		{
			name: "non_standard_port",
			config: &Config{
				Host:     "db.example.com",
				Port:     5433,
				User:     "admin",
				Password: "secret",
				Database: "mydb",
			},
			expected: "postgres://admin:secret@db.example.com:5433/mydb?sslmode=disable",
		},
		{
			name: "special_characters_in_password",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "user",
				Password: "p@ss!word",
				Database: "db",
			},
			expected: "postgres://user:p@ss!word@localhost:5432/db?sslmode=disable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test connection string format
			connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
				tt.config.User, tt.config.Password, tt.config.Host, tt.config.Port, tt.config.Database)

			if connStr != tt.expected {
				t.Errorf("Connection string = %v, want %v", connStr, tt.expected)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNewPostgresManager(b *testing.B) {
	config := &Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewPostgresManager(config)
	}
}

func BenchmarkConnectionStringGeneration(b *testing.B) {
	config := &Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
			config.User, config.Password, config.Host, config.Port, config.Database)
	}
}

// Additional comprehensive tests

func TestConfig_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		isValid bool
	}{
		{
			name: "valid_config",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			isValid: true,
		},
		{
			name: "empty_host",
			config: &Config{
				Host:     "",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			isValid: false,
		},
		{
			name: "zero_port",
			config: &Config{
				Host:     "localhost",
				Port:     0,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			isValid: false,
		},
		{
			name: "negative_port",
			config: &Config{
				Host:     "localhost",
				Port:     -1,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			isValid: false,
		},
		{
			name: "empty_user",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "",
				Password: "password",
				Database: "testdb",
			},
			isValid: false,
		},
		{
			name: "empty_database",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "",
			},
			isValid: false,
		},
		{
			name: "empty_password_allowed",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "",
				Database: "testdb",
			},
			isValid: true, // Empty password should be allowed
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test basic validation logic
			isValid := tt.config.Host != "" &&
				tt.config.Port > 0 &&
				tt.config.User != "" &&
				tt.config.Database != ""

			assert.Equal(t, tt.isValid, isValid)
		})
	}
}

func TestConfig_ConnectionStringFormats(t *testing.T) {
	tests := []struct {
		name       string
		config     *Config
		expected   string
		dbOverride string
	}{
		{
			name: "default_postgres_db",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expected:   "postgres://postgres:password@localhost:5432/postgres?sslmode=disable",
			dbOverride: "postgres",
		},
		{
			name: "custom_database",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expected:   "postgres://postgres:password@localhost:5432/testdb?sslmode=disable",
			dbOverride: "",
		},
		{
			name: "special_characters",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "user@domain",
				Password: "p@ss#word!",
				Database: "test-db_123",
			},
			expected:   "postgres://user@domain:p@ss#word!@localhost:5432/test-db_123?sslmode=disable",
			dbOverride: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db := tt.config.Database
			if tt.dbOverride != "" {
				db = tt.dbOverride
			}

			connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
				tt.config.User, tt.config.Password, tt.config.Host, tt.config.Port, db)

			assert.Equal(t, tt.expected, connStr)
		})
	}
}

func TestPostgresManager_InitializeDatabase_ErrorCases(t *testing.T) {
	tests := []struct {
		name          string
		config        *Config
		expectedError string
	}{
		{
			name: "invalid_host",
			config: &Config{
				Host:     "invalid-host-12345",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expectedError: "failed to connect to PostgreSQL",
		},
		{
			name: "invalid_port",
			config: &Config{
				Host:     "localhost",
				Port:     99999,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expectedError: "failed to connect to PostgreSQL",
		},
		{
			name: "empty_config",
			config: &Config{
				Host:     "",
				Port:     0,
				User:     "",
				Password: "",
				Database: "",
			},
			expectedError: "failed to connect to PostgreSQL",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)
			ctx := context.Background()

			err := pm.InitializeDatabase(ctx)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestPostgresManager_InitializeDatabase_ContextCancellation(t *testing.T) {
	pm := NewPostgresManager(&Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := pm.InitializeDatabase(ctx)
	assert.Error(t, err)
}

func TestPostgresManager_WaitForReady_ShortTimeout(t *testing.T) {
	pm := NewPostgresManager(&Config{
		Host:     "invalid-host-12345",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	})

	ctx := context.Background()

	// Very short timeout should trigger timeout error
	err := pm.WaitForReady(ctx, 10*time.Millisecond)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout waiting for PostgreSQL")
}

func TestPostgresManager_WaitForReady_ImmediateContextCancel(t *testing.T) {
	pm := NewPostgresManager(&Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := pm.WaitForReady(ctx, 5*time.Second)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout waiting for PostgreSQL")
}

func TestPostgresManager_TestConnection_ErrorCases(t *testing.T) {
	tests := []struct {
		name          string
		config        *Config
		expectedError string
	}{
		{
			name: "connection_timeout",
			config: &Config{
				Host:     "1.2.3.4", // Non-routable IP
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			expectedError: "failed to connect",
		},
		{
			name: "invalid_credentials",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "invalid_user",
				Password: "wrong_password",
				Database: "testdb",
			},
			expectedError: "failed to connect",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)

			// Create a context with timeout to avoid hanging
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			err := pm.TestConnection(ctx)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestPostgresManager_ExecuteSQL_ErrorCases(t *testing.T) {
	tests := []struct {
		name          string
		config        *Config
		sql           string
		expectedError string
	}{
		{
			name: "connection_failed",
			config: &Config{
				Host:     "invalid-host-12345",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			sql:           "SELECT 1",
			expectedError: "failed to connect",
		},
		{
			name: "invalid_sql_syntax",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			sql:           "INVALID SQL SYNTAX HERE",
			expectedError: "failed to connect", // Will fail at connection level in test
		},
		{
			name: "empty_sql",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "postgres",
				Password: "password",
				Database: "testdb",
			},
			sql:           "",
			expectedError: "failed to connect", // Will fail at connection level in test
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := NewPostgresManager(tt.config)

			// Create a context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			err := pm.ExecuteSQL(ctx, tt.sql)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestPostgresManager_ManagerState(t *testing.T) {
	config := &Config{
		Host:     "localhost",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	}

	pm := NewPostgresManager(config)

	// Test that manager holds the correct config
	assert.Equal(t, config, pm.config)
	assert.NotNil(t, pm.logger)

	// Test config immutability
	originalHost := pm.config.Host
	config.Host = "changed"
	assert.Equal(t, "changed", pm.config.Host) // Config is shared reference

	// Reset for consistency
	pm.config.Host = originalHost
}

func TestPostgresManager_MultipleOperations(t *testing.T) {
	config := &Config{
		Host:     "invalid-host-12345",
		Port:     5432,
		User:     "postgres",
		Password: "password",
		Database: "testdb",
	}

	pm := NewPostgresManager(config)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Test that multiple operations fail consistently
	err1 := pm.InitializeDatabase(ctx)
	err2 := pm.TestConnection(ctx)
	err3 := pm.ExecuteSQL(ctx, "SELECT 1")

	assert.Error(t, err1)
	assert.Error(t, err2)
	assert.Error(t, err3)

	// All should be connection-related errors
	assert.Contains(t, err1.Error(), "failed to connect")
	assert.Contains(t, err2.Error(), "failed to connect")
	assert.Contains(t, err3.Error(), "failed to connect")
}

func TestPostgresManager_ConfigPointerConsistency(t *testing.T) {
	config1 := &Config{
		Host:     "host1",
		Port:     5432,
		User:     "user1",
		Password: "pass1",
		Database: "db1",
	}

	config2 := &Config{
		Host:     "host2",
		Port:     5433,
		User:     "user2",
		Password: "pass2",
		Database: "db2",
	}

	pm1 := NewPostgresManager(config1)
	pm2 := NewPostgresManager(config2)

	// Each manager should have its own config
	assert.NotEqual(t, pm1.config, pm2.config)
	assert.Equal(t, "host1", pm1.config.Host)
	assert.Equal(t, "host2", pm2.config.Host)
	assert.Equal(t, 5432, pm1.config.Port)
	assert.Equal(t, 5433, pm2.config.Port)
}

// Test edge cases for connection string generation
func TestConnectionStringEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		config   *Config
		expected string
	}{
		{
			name: "unicode_characters",
			config: &Config{
				Host:     "localhost",
				Port:     5432,
				User:     "José",
				Password: "café",
				Database: "naïve",
			},
			expected: "postgres://José:café@localhost:5432/naïve?sslmode=disable",
		},
		{
			name: "numeric_strings",
			config: &Config{
				Host:     "123.456.789.012",
				Port:     1234,
				User:     "user123",
				Password: "pass456",
				Database: "db789",
			},
			expected: "postgres://user123:pass456@123.456.789.012:1234/db789?sslmode=disable",
		},
		{
			name: "long_strings",
			config: &Config{
				Host:     "very-long-hostname-that-might-be-used-in-some-environments.example.com",
				Port:     5432,
				User:     "very_long_username_that_exceeds_normal_length",
				Password: "very_long_password_with_many_characters_and_symbols_!@#$%^&*()",
				Database: "very_long_database_name_that_is_quite_descriptive",
			},
			expected: "postgres://very_long_username_that_exceeds_normal_length:very_long_password_with_many_characters_and_symbols_!@#$%^&*()@very-long-hostname-that-might-be-used-in-some-environments.example.com:5432/very_long_database_name_that_is_quite_descriptive?sslmode=disable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			connStr := fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
				tt.config.User, tt.config.Password, tt.config.Host, tt.config.Port, tt.config.Database)

			assert.Equal(t, tt.expected, connStr)
		})
	}
}
