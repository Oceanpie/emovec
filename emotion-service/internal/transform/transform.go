package transform

import "math"

// SplitMix64 is a deterministic 64-bit PRNG using the splitmix64 algorithm.
// Produces IDENTICAL sequences to the Python SplitMix64 class in pack_prepare.py.
type SplitMix64 struct {
	state uint64
}

// NewSplitMix64 creates a new SplitMix64 PRNG with the given seed.
func NewSplitMix64(seed int) *SplitMix64 {
	return &SplitMix64{state: uint64(seed) & 0xFFFFFFFFFFFFFFFF}
}

func (s *SplitMix64) nextUint64() uint64 {
	s.state += 0x9E3779B97F4A7C15
	z := s.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z = z ^ (z >> 31)
	return z
}

// NextFloat returns a float64 in [0, 1).
func (s *SplitMix64) NextFloat() float64 {
	return float64(s.nextUint64()>>11) / float64(uint64(1)<<53)
}

// UniformFloat32 fills dst with uniform random values in [low, high).
func (s *SplitMix64) UniformFloat32(dst []float32, low, high float32) {
	rng := high - low
	for i := range dst {
		dst[i] = low + rng*float32(s.NextFloat())
	}
}

// Permutation generates a permutation of n elements using Fisher-Yates shuffle.
func (s *SplitMix64) Permutation(n int) []int {
	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j := int(s.NextFloat() * float64(i+1))
		if j > i {
			j = i
		}
		result[i], result[j] = result[j], result[i]
	}
	return result
}

// InversePermutation computes the inverse of a permutation.
func InversePermutation(perm []int) []int {
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	return inv
}

const (
	VectorDim  = 1024
	TotalRows  = 1024
	RealRows   = 153
	NumDims    = 8
	TransformB = 0.9
	TransformR = 0.2 // range = high - low = 1.1 - 0.9
	TransformSeed = 42
)

// GenerateTransformMatrix creates the element-wise transform matrix B.
// Values are uniformly distributed in [0.9, 1.1].
// Uses SplitMix64 for cross-language reproducibility with Python.
func GenerateTransformMatrix(seed, rows, cols int) []float32 {
	rng := NewSplitMix64(seed)
	n := rows * cols
	B := make([]float32, n)
	rng.UniformFloat32(B, TransformB, TransformB+TransformR)
	return B
}

// GeneratePermutation creates a permutation of n elements from seed+1000.
// Uses SplitMix64 for cross-language reproducibility with Python.
func GeneratePermutation(seed, n int) []int {
	rng := NewSplitMix64(seed + 1000)
	return rng.Permutation(n)
}

// InverseTransform applies element-wise division to recover original data.
// recovered[i] = stored[i] / B[i]
func InverseTransform(stored, B []float32) {
	for i := range stored {
		stored[i] = stored[i] / B[i]
	}
}

// UnshuffleRows reorders rows of a row-major matrix using the inverse permutation.
func UnshuffleRows(data []float32, rows, cols int, invPerm []int) []float32 {
	result := make([]float32, len(data))
	for dstRow, srcRow := range invPerm {
		copy(result[dstRow*cols:(dstRow+1)*cols], data[srcRow*cols:(srcRow+1)*cols])
	}
	return result
}

// L2NormalizeRows normalizes each row of a row-major matrix in-place.
func L2NormalizeRows(data []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		row := data[i*cols : (i+1)*cols]
		var sum float64
		for _, v := range row {
			sum += float64(v) * float64(v)
		}
		norm := float32(math.Sqrt(sum))
		if norm < 1e-10 {
			norm = 1e-10
		}
		for j := range row {
			row[j] = row[j] / norm
		}
	}
}

// ExtractRealRows returns the first RealRows rows from a row-major matrix.
func ExtractRealRows(data []float32, cols int) []float32 {
	n := RealRows * cols
	result := make([]float32, n)
	copy(result, data[:n])
	return result
}
