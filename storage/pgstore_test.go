package storage

import (
	"database/sql"
	"errors"
	"testing"
)

type fakeResult struct{}

func (fakeResult) LastInsertId() (int64, error) { return 0, nil }
func (fakeResult) RowsAffected() (int64, error) { return 0, nil }

type fakeScanner struct {
	answer string
	err    error
}

func (f fakeScanner) Scan(dest ...interface{}) error {
	if f.err != nil {
		return f.err
	}
	if len(dest) != 1 {
		return errors.New("dest len")
	}
	ptr, ok := dest[0].(*string)
	if !ok {
		return errors.New("type")
	}
	*ptr = f.answer
	return nil
}

type fakeDB struct {
	expectedPrompt string
	answer         string
}

func (f *fakeDB) Exec(query string, args ...interface{}) (sql.Result, error) {
	return fakeResult{}, nil
}

func (f *fakeDB) QueryRow(query string, args ...interface{}) scanner {
	return fakeScanner{answer: f.answer}
}

type fakeStmt struct{ db *fakeDB }

func (s *fakeStmt) Exec(args ...interface{}) (sql.Result, error) { return s.db.Exec("", args...) }
func (s *fakeStmt) QueryRow(args ...interface{}) scanner         { return s.db.QueryRow("", args...) }
func (s *fakeStmt) Close() error                                 { return nil }

func TestPGStoreSetGet(t *testing.T) {
	db := &fakeDB{expectedPrompt: "p", answer: "a"}
	store := &PGStore{setStmt: &fakeStmt{db}, getStmt: &fakeStmt{db}, similarStmt: &fakeStmt{db}}
	if err := store.Set("p", nil, "a"); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	val, ok, err := store.Get("p")
	if err != nil || !ok || val != "a" {
		t.Fatalf("Get failed: %v %v %v", val, ok, err)
	}
	val, ok, err = store.GetByEmbedding(nil)
	if err != nil || !ok || val != "a" {
		t.Fatalf("GetByEmbedding failed: %v %v %v", val, ok, err)
	}
}

func TestPGStoreImportFlush(t *testing.T) {
	db := &fakeDB{}
	store := &PGStore{setStmt: &fakeStmt{db}, getStmt: &fakeStmt{db}, similarStmt: &fakeStmt{db}}
	prompts := []string{"p1", "p2"}
	answers := []string{"a1", "a2"}
	if err := store.ImportData(prompts, nil, answers); err != nil {
		t.Fatalf("import: %v", err)
	}
	if err := store.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
}

func TestPGStoreClose(t *testing.T) {
	store := &PGStore{setStmt: &fakeStmt{}, getStmt: &fakeStmt{}, similarStmt: &fakeStmt{}}
	if err := store.Close(); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
}
