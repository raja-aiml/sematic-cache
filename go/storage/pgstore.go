package storage

import (
	"database/sql"
	"time"
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
	prepareFunc = func(db *sql.DB, query string) (stmt, error) {
		st, err := db.Prepare(query)
		if err != nil {
			return nil, err
		}
		return stmtWrapper{st}, nil
	}
)

type stmtWrapper struct{ *sql.Stmt }

func (s stmtWrapper) Exec(args ...interface{}) (sql.Result, error) { return s.Stmt.Exec(args...) }
func (s stmtWrapper) QueryRow(args ...interface{}) scanner         { return s.Stmt.QueryRow(args...) }
func (s stmtWrapper) Close() error                                 { return s.Stmt.Close() }

// stmt defines the minimal interface used from *sql.Stmt.
type stmt interface {
	Exec(args ...interface{}) (sql.Result, error)
	QueryRow(args ...interface{}) scanner
	Close() error
}

// PGStore stores prompts and answers in PostgreSQL.
type PGStore struct {
	db          *sql.DB
	setStmt     stmt
	getStmt     stmt
	similarStmt stmt
}

// NewPGStore opens a PostgreSQL connection.
func NewPGStore(conn string) (*PGStore, error) {
	db, err := sql.Open("postgres", conn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	setStmt, err := prepareFunc(db, `INSERT INTO cache(prompt, embedding, answer)
        VALUES ($1, $2, $3)
        ON CONFLICT (prompt) DO UPDATE SET embedding = EXCLUDED.embedding, answer = EXCLUDED.answer`)
	if err != nil {
		db.Close()
		return nil, err
	}
	getStmt, err := prepareFunc(db, `SELECT answer FROM cache WHERE prompt = $1`)
	if err != nil {
		setStmt.Close()
		db.Close()
		return nil, err
	}
	similarStmt, err := prepareFunc(db, `SELECT answer FROM cache ORDER BY embedding <-> $1 LIMIT 1`)
	if err != nil {
		setStmt.Close()
		getStmt.Close()
		db.Close()
		return nil, err
	}
	return &PGStore{db: db, setStmt: setStmt, getStmt: getStmt, similarStmt: similarStmt}, nil
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
	if s.setStmt == nil {
		var err error
		s.setStmt, err = prepareFunc(s.db, `INSERT INTO cache(prompt, embedding, answer)
        VALUES ($1, $2, $3)
        ON CONFLICT (prompt) DO UPDATE SET embedding = EXCLUDED.embedding, answer = EXCLUDED.answer`)
		if err != nil {
			return err
		}
	}
	_, err := s.setStmt.Exec(prompt, embedding, answer)
	return err
}

// Get retrieves the answer for a prompt.
func (s *PGStore) Get(prompt string) (string, bool, error) {
	if s.getStmt == nil {
		var err error
		s.getStmt, err = prepareFunc(s.db, `SELECT answer FROM cache WHERE prompt = $1`)
		if err != nil {
			return "", false, err
		}
	}
	row := s.getStmt.QueryRow(prompt)
	var ans string
	if err := row.Scan(&ans); err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}
	return ans, true, nil
}

// GetByEmbedding retrieves the nearest answer for an embedding using pgvector operators.
func (s *PGStore) GetByEmbedding(embed []float32) (string, bool, error) {
	if s.similarStmt == nil {
		var err error
		s.similarStmt, err = prepareFunc(s.db, `SELECT answer FROM cache ORDER BY embedding <-> $1 LIMIT 1`)
		if err != nil {
			return "", false, err
		}
	}
	row := s.similarStmt.QueryRow(embed)
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
	if s.setStmt != nil {
		s.setStmt.Close()
	}
	if s.getStmt != nil {
		s.getStmt.Close()
	}
	if s.similarStmt != nil {
		s.similarStmt.Close()
	}
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}
