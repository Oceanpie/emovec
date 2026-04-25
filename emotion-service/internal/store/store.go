package store

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"

	"emovec/internal/matching"
	"emovec/internal/safetensors"
	"emovec/internal/transform"
)

const (
	VectorDim = 1024
	NumDims   = 8
)

// Store holds pre-loaded prototype vectors and labels.
// Immutable after loading — safe for concurrent read access.
type Store struct {
	Vectors        []float32 // [NumEntries * VectorDim] row-major, L2-normalized
	Labels         []float32 // [NumEntries * NumDims] row-major (plutchik scheme)
	LabelsOriginal []float32 // [NumEntries * NumDims] row-major (original scheme)
	Titles         []string  // [NumEntries]
	Dims           []string  // [NumDims] e.g. ["joy","anger",...]
	DimsOriginal   []string  // [NumDims] e.g. ["高兴","愤怒",...]
	NumEntries     int
}

type labelsFile struct {
	Dimensions         []string `json:"dimensions"`
	DimensionsOriginal []string `json:"dimensions_original"`
	Entries            []struct {
		RowIndex      int                `json:"row_index"`
		Title         string             `json:"title"`
		LabelPlutchik map[string]float64 `json:"label_plutchik"`
		LabelOriginal map[string]float64 `json:"label_original"`
	} `json:"entries"`
}

// LoadStore reads binary vectors and labels JSON, pre-normalizes all vectors.
func LoadStore(vectorsPath, labelsPath string, logger *slog.Logger) (*Store, error) {
	// Load vectors (raw float32 LE, 153*1024 elements)
	vecData, err := os.ReadFile(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}

	// Load labels JSON
	labelData, err := os.ReadFile(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	var lf labelsFile
	if err := json.Unmarshal(labelData, &lf); err != nil {
		return nil, fmt.Errorf("parse labels: %w", err)
	}

	nEntries := len(lf.Entries)
	expectedSize := nEntries * VectorDim * 4
	if len(vecData) != expectedSize {
		return nil, fmt.Errorf("vectors file size mismatch: got %d, expected %d (%d entries × %d dims × 4 bytes)",
			len(vecData), expectedSize, nEntries, VectorDim)
	}

	// Convert raw bytes to float32 slice
	nFloats := nEntries * VectorDim
	vectors := make([]float32, nFloats)
	for i := 0; i < nFloats; i++ {
		bits := binary.LittleEndian.Uint32(vecData[i*4 : (i+1)*4])
		vectors[i] = math.Float32frombits(bits)
	}

	// L2-normalize each vector (pre-compute so runtime doesn't need to)
	for i := 0; i < nEntries; i++ {
		row := vectors[i*VectorDim : (i+1)*VectorDim]
		matching.L2Normalize(row)
	}

	// Build label matrices
	dims := lf.Dimensions
	dimsOriginal := lf.DimensionsOriginal
	labels := make([]float32, nEntries*NumDims)
	labelsOriginal := make([]float32, nEntries*NumDims)
	titles := make([]string, nEntries)

	for i, e := range lf.Entries {
		titles[i] = e.Title
		for d, dimName := range dims {
			v, ok := e.LabelPlutchik[dimName]
			if !ok {
				v = 0.0
			}
			labels[i*NumDims+d] = float32(v)
		}
		for d, dimName := range dimsOriginal {
			v, ok := e.LabelOriginal[dimName]
			if !ok {
				v = 0.0
			}
			labelsOriginal[i*NumDims+d] = float32(v)
		}
	}

	logger.Info("prototype store loaded",
		"entries", nEntries,
		"vector_dim", VectorDim,
		"dims", dims,
	)

	return &Store{
		Vectors:        vectors,
		Labels:         labels,
		LabelsOriginal: labelsOriginal,
		Titles:         titles,
		Dims:           dims,
		DimsOriginal:   dimsOriginal,
		NumEntries:     nEntries,
	}, nil
}

// LoadStoreFromSafetensors loads prototype data from a safetensors file.
// The file is disguised as a 2-layer MLP:
//   - layers.0.weight (1024, 1024): prototype + fill vectors, element-wise transformed
//   - layers.1.weight (8, 1024): labels (transposed: [8, 1024])
//
// Loading procedure:
//  1. Read layers.0.weight and layers.1.weight
//  2. Generate transform matrix B from seed (splitmix64 PRNG)
//  3. Inverse transform: recovered = stored / B
//  4. Generate permutation from seed+1000, unshuffle rows
//  5. Extract first 153 rows (real prototypes)
//  6. L2 normalize vectors
//  7. Build label matrices from dims JSON
func LoadStoreFromSafetensors(modelPath, dimsPath string, seed int, logger *slog.Logger) (*Store, error) {
	// Load safetensors
	sf, err := safetensors.Load(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load safetensors: %w", err)
	}

	// Read layers.0.weight (1024, 1024)
	storedVecs, rows, cols, err := sf.ReadF32Matrix("layers.0.weight")
	if err != nil {
		return nil, fmt.Errorf("read layers.0.weight: %w", err)
	}
	if rows != transform.TotalRows || cols != transform.VectorDim {
		return nil, fmt.Errorf("layers.0.weight shape (%d,%d), expected (%d,%d)",
			rows, cols, transform.TotalRows, transform.VectorDim)
	}

	// Read layers.1.weight (8, 1024) — labels transposed
	labelsT, labelRows, labelCols, err := sf.ReadF32Matrix("layers.1.weight")
	if err != nil {
		return nil, fmt.Errorf("read layers.1.weight: %w", err)
	}
	if labelRows != NumDims || labelCols != transform.TotalRows {
		return nil, fmt.Errorf("layers.1.weight shape (%d,%d), expected (%d,%d)",
			labelRows, labelCols, NumDims, transform.TotalRows)
	}

	// Transpose labels from (8, 1024) to (1024, 8)
	allLabels := make([]float32, transform.TotalRows*NumDims)
	for r := 0; r < transform.TotalRows; r++ {
		for c := 0; c < NumDims; c++ {
			allLabels[r*NumDims+c] = labelsT[c*transform.TotalRows+r]
		}
	}

	logger.Info("safetensors loaded",
		"vectors_shape", fmt.Sprintf("(%d,%d)", rows, cols),
		"labels_shape", fmt.Sprintf("(%d,%d)", labelRows, labelCols),
	)

	// Inverse transform
	B := transform.GenerateTransformMatrix(seed, transform.TotalRows, transform.VectorDim)
	transform.InverseTransform(storedVecs, B)

	// Unshuffle
	perm := transform.GeneratePermutation(seed, transform.TotalRows)
	invPerm := transform.InversePermutation(perm)
	recoveredVecs := transform.UnshuffleRows(storedVecs, transform.TotalRows, transform.VectorDim, invPerm)
	recoveredLabels := transform.UnshuffleRows(allLabels, transform.TotalRows, NumDims, invPerm)

	// Extract real rows (first 153)
	realVecs := transform.ExtractRealRows(recoveredVecs, transform.VectorDim)
	realLabels := make([]float32, transform.RealRows*NumDims)
	copy(realLabels, recoveredLabels[:transform.RealRows*NumDims])

	// L2 normalize vectors
	transform.L2NormalizeRows(realVecs, transform.RealRows, transform.VectorDim)

	// Load dims JSON for dimension names
	dims, dimsOriginal, err := loadDims(dimsPath)
	if err != nil {
		return nil, fmt.Errorf("load dims: %w", err)
	}

	nEntries := transform.RealRows

	logger.Info("prototype store recovered from safetensors",
		"entries", nEntries,
		"vector_dim", VectorDim,
		"dims", dims,
	)

	return &Store{
		Vectors:        realVecs,
		Labels:         realLabels,
		LabelsOriginal: realLabels, // Same for now (safetensors only stores plutchik)
		Titles:         make([]string, nEntries),
		Dims:           dims,
		DimsOriginal:   dimsOriginal,
		NumEntries:     nEntries,
	}, nil
}

// loadDims reads dimension names from the prototype_labels.json file.
func loadDims(path string) (dims []string, dimsOriginal []string, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read dims file: %w", err)
	}
	var lf labelsFile
	if err := json.Unmarshal(data, &lf); err != nil {
		return nil, nil, fmt.Errorf("parse dims file: %w", err)
	}
	return lf.Dimensions, lf.DimensionsOriginal, nil
}

// GetLabels returns the label matrix for the given scheme ("plutchik" or "original").
func (s *Store) GetLabels(scheme string) []float32 {
	if scheme == "original" {
		return s.LabelsOriginal
	}
	return s.Labels
}

// GetDims returns the dimension names for the given scheme.
func (s *Store) GetDims(scheme string) []string {
	if scheme == "original" {
		return s.DimsOriginal
	}
	return s.Dims
}
