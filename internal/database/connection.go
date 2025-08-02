package database

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"time"

	_ "github.com/lib/pq" // PostgreSQL driver
)

// WaitForDatabase implements retry logic for database connection
func WaitForDatabase(dsn string) error {
	const (
		maxRetries    = 30
		retryInterval = 2 * time.Second
		pingTimeout   = 5 * time.Second
	)

	for i := 0; i < maxRetries; i++ {
		if err := tryConnect(dsn, pingTimeout); err == nil {
			slog.Info("Database connection established")
			return nil
		} else {
			slog.Warn("Database connection attempt failed",
				"attempt", i+1,
				"max_retries", maxRetries,
				"error", err,
			)
			if i < maxRetries-1 {
				time.Sleep(retryInterval)
			}
		}
	}

	return fmt.Errorf("failed to connect to database after %d attempts", maxRetries)
}

func tryConnect(dsn string, timeout time.Duration) error {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return err
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return db.PingContext(ctx)
}
