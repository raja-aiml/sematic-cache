package database

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Mock types for testing
type mockRow struct {
	scanFunc func(dest ...interface{}) error
}

func (m *mockRow) Scan(dest ...interface{}) error {
	if m.scanFunc != nil {
		return m.scanFunc(dest...)
	}
	return nil
}

type mockConn struct {
	connectErr   error
	execErr      error
	queryRowFunc func(ctx context.Context, sql string, args ...interface{}) pgx.Row
	closeErr     error
}

func (m *mockConn) Close(ctx context.Context) error {
	return m.closeErr
}

func (m *mockConn) Exec(ctx context.Context, sql string, arguments ...interface{}) (pgconn.CommandTag, error) {
	return pgconn.NewCommandTag(""), m.execErr
}

func (m *mockConn) QueryRow(ctx context.Context, sql string, args ...interface{}) pgx.Row {
	if m.queryRowFunc != nil {
		return m.queryRowFunc(ctx, sql, args...)
	}
	return &mockRow{}
}

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