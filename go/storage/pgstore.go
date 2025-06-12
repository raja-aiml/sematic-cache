package storage

import (
	"database/sql"
)

// scanner defines the minimal interface for sql.Row
// so it can be mocked in tests.
type scanner interface {
	Scan(dest ...interface{}) error
}

// function variables for testability
var (
	execFunc = func(db *sql.DB, query string, args ...interface{}) (sql.Result, error) {
		return db.Exec(query, args...)
	}
	queryRowFunc = func(db *sql.DB, query string, args ...interface{}) scanner {
		return db.QueryRow(query, args...)
	}
)

// PGStore stores prompts and answers in PostgreSQL.
type PGStore struct {
	db *sql.DB
}

// NewPGStore opens a PostgreSQL connection.
func NewPGStore(conn string) (*PGStore, error) {
	db, err := sql.Open("postgres", conn)
	if err != nil {
		return nil, err
	}
	return &PGStore{db: db}, nil
}

// Init creates the table if it doesn't exist.
func (s *PGStore) Init() error {
	_, err := execFunc(s.db, `CREATE TABLE IF NOT EXISTS cache (
        prompt TEXT PRIMARY KEY,
        embedding VECTOR(1536),
        answer TEXT
    )`)
	return err
}

// Set inserts or updates a cached answer.
func (s *PGStore) Set(prompt string, embedding []float32, answer string) error {
	_, err := execFunc(s.db, `INSERT INTO cache(prompt, embedding, answer)
        VALUES ($1, $2, $3)
        ON CONFLICT (prompt) DO UPDATE SET embedding = EXCLUDED.embedding, answer = EXCLUDED.answer`, prompt, embedding, answer)
	return err
}

// Get retrieves the answer for a prompt.
func (s *PGStore) Get(prompt string) (string, bool, error) {
	row := queryRowFunc(s.db, `SELECT answer FROM cache WHERE prompt = $1`, prompt)
	var ans string
	if err := row.Scan(&ans); err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}
	return ans, true, nil
}

// Close releases any database resources.
func (s *PGStore) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}
