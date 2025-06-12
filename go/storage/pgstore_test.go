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
	if len(args) > 0 && args[0] == f.expectedPrompt {
		return fakeScanner{answer: f.answer}
	}
	return fakeScanner{err: sql.ErrNoRows}
}

func TestPGStoreSetGet(t *testing.T) {
	db := &fakeDB{expectedPrompt: "p", answer: "a"}
	// override funcs
	oldExec := execFunc
	oldQuery := queryRowFunc
	execFunc = func(_ *sql.DB, q string, args ...interface{}) (sql.Result, error) {
		return db.Exec(q, args...)
	}
	queryRowFunc = func(_ *sql.DB, q string, args ...interface{}) scanner {
		return db.QueryRow(q, args...)
	}
	defer func() { execFunc = oldExec; queryRowFunc = oldQuery }()

	store := &PGStore{}
	if err := store.Set("p", nil, "a"); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	val, ok, err := store.Get("p")
	if err != nil || !ok || val != "a" {
		t.Fatalf("Get failed: %v %v %v", val, ok, err)
	}
}
