// Package storage provides a PostgreSQL-backed cache implementation.

package storage

import (
   "database/sql"
   "fmt"
   "time"
)

// scanner defines the minimal interface for sql.Row
// so it can be mocked in tests.
type scanner interface {
	Scan(dest ...interface{}) error
}
// initStatements prepares SQL statements for the store.
func (s *PGStore) initStatements() error {
   var err error
   s.setStmt, err = prepareFunc(s.db, `INSERT INTO cache(prompt, embedding, answer)
       VALUES ($1, $2, $3)
       ON CONFLICT (prompt) DO UPDATE SET embedding = EXCLUDED.embedding, answer = EXCLUDED.answer`)
   if err != nil {
       return err
   }
   s.getStmt, err = prepareFunc(s.db, `SELECT answer FROM cache WHERE prompt = $1`)
   if err != nil {
       s.setStmt.Close()
       return err
   }
   s.similarStmt, err = prepareFunc(s.db, `SELECT answer FROM cache ORDER BY embedding <-> $1 LIMIT 1`)
   if err != nil {
       s.setStmt.Close()
       s.getStmt.Close()
       return err
   }
   return nil
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
		return nil, fmt.Errorf("open database: %w", err)
	}
	// verify connection
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping database: %w", err)
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

   // initialize prepared statements to avoid race conditions
   store := &PGStore{db: db}
   if err := store.initStatements(); err != nil {
       db.Close()
       return nil, err
   }
   return store, nil
}

// Init creates the table if it doesn't exist.
func (s *PGStore) Init() error {
	_, err := execFunc(s.db, `CREATE TABLE IF NOT EXISTS cache (
        prompt TEXT PRIMARY KEY,
        embedding VECTOR(1536),
        answer TEXT
    )`)
	if err != nil {
		return fmt.Errorf("init table: %w", err)
	}
	return nil
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
	if err != nil {
		return fmt.Errorf("set prompt=%q: %w", prompt, err)
	}
	return nil
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
		return "", false, fmt.Errorf("get prompt=%q: %w", prompt, err)
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
		return "", false, fmt.Errorf("get by embedding: %w", err)
	}
	return ans, true, nil
}

// Flush removes all cached rows.
// Flush removes all cached rows.
func (s *PGStore) Flush() error {
	// If no DB is configured (e.g. in tests), nothing to do
	if s.db == nil {
		return nil
	}
	_, err := execFunc(s.db, `DELETE FROM cache`)
	if err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return nil
}

// ImportData bulk loads prompts with their embeddings and answers.
func (s *PGStore) ImportData(prompts []string, embeddings [][]float32, answers []string) error {
	for i, p := range prompts {
		var e []float32
		if i < len(embeddings) {
			e = embeddings[i]
		}
		var a string
		if i < len(answers) {
			a = answers[i]
		}
		if err := s.Set(p, e, a); err != nil {
			return err
		}
	}
	return nil
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
