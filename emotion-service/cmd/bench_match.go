package main

import (
	"fmt"
	"time"
	"emovec/internal/matching"
	"emovec/internal/store"
	"emovec/internal/transform"
	"emovec/internal/safetensors"
)

func main() {
	// Load store from safetensors
	sf, _ := safetensors.Load("data/model.safetensors")
	storedVecs, _, _, _ := sf.ReadF32Matrix("layers.0.weight")
	labelsT, _, _, _ := sf.ReadF32Matrix("layers.1.weight")
	
	B := transform.GenerateTransformMatrix(transform.TransformSeed, transform.TotalRows, transform.VectorDim)
	transform.InverseTransform(storedVecs, B)
	perm := transform.GeneratePermutation(transform.TransformSeed, transform.TotalRows)
	invPerm := transform.InversePermutation(perm)
	recoveredVecs := transform.UnshuffleRows(storedVecs, transform.TotalRows, transform.VectorDim, invPerm)
	realVecs := transform.ExtractRealRows(recoveredVecs, transform.VectorDim)
	transform.L2NormalizeRows(realVecs, transform.RealRows, transform.VectorDim)
	
	allLabels := make([]float32, transform.TotalRows*store.NumDims)
	for r := 0; r < transform.TotalRows; r++ {
		for c := 0; c < store.NumDims; c++ {
			allLabels[r*store.NumDims+c] = labelsT[c*transform.TotalRows+r]
		}
	}
	recoveredLabels := transform.UnshuffleRows(allLabels, transform.TotalRows, store.NumDims, invPerm)
	realLabels := make([]float32, transform.RealRows*store.NumDims)
	copy(realLabels, recoveredLabels[:transform.RealRows*store.NumDims])
	
	dims := []string{"joy","anger","sadness","fear","disgust","surprise","trust","anticipation"}
	
	// Fake query vector (1024D)
	query := make([]float32, store.VectorDim)
	for i := range query { query[i] = 0.1 }
	matching.L2Normalize(query)
	
	// Benchmark: 10000 iterations
	N := 10000
	start := time.Now()
	for i := 0; i < N; i++ {
		matching.Match(realVecs, realLabels, dims, query, 7, 0.5, store.VectorDim, store.NumDims)
	}
	elapsed := time.Since(start)
	
	fmt.Printf("Matching: %d iterations in %v\n", N, elapsed)
	fmt.Printf("  Per query: %v\n", elapsed/time.Duration(N))
	fmt.Printf("  QPS: %.0f\n", float64(N)/elapsed.Seconds())
}
