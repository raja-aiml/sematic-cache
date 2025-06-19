package reduction

import (
	"sync"
)

// ModelLocker provides thread-safe access to ML models
type ModelLocker struct {
	mu sync.RWMutex
}

// NewModelLocker creates a new model locker
func NewModelLocker() *ModelLocker {
	return &ModelLocker{}
}

// Lock acquires an exclusive lock for model updates
func (m *ModelLocker) Lock() {
	m.mu.Lock()
}

// Unlock releases the exclusive lock
func (m *ModelLocker) Unlock() {
	m.mu.Unlock()
}

// RLock acquires a shared lock for model reads
func (m *ModelLocker) RLock() {
	m.mu.RLock()
}

// RUnlock releases the shared lock
func (m *ModelLocker) RUnlock() {
	m.mu.RUnlock()
}