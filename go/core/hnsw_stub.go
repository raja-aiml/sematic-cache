//go:build !hnsw
// +build !hnsw

package core

// HNSWIndex is a stub for an ANNIndex backed by an HNSW graph.
// To enable real HNSW functionality, build with tag 'hnsw' and provide an
// implementation in core/hnsw_hnsw.go that uses your chosen HNSW library.
type HNSWIndex struct{}

// NewHNSWIndex is a stub that panics by default. To enable, build with '-tags hnsw'.
func NewHNSWIndex(dim, M, efConstruction, efSearch int) ANNIndex {
   panic("HNSW index is not implemented: compile with '-tags hnsw' and provide hnsw_hnsw.go")
}