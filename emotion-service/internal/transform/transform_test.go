package transform

import (
	"math"
	"testing"
)

func TestSplitMix64Sequence(t *testing.T) {
	// Verify sequence is deterministic: two instances with same seed produce same output
	rng1 := NewSplitMix64(42)
	rng2 := NewSplitMix64(42)
	for i := 0; i < 100; i++ {
		v1 := rng1.nextUint64()
		v2 := rng2.nextUint64()
		if v1 != v2 {
			t.Errorf("non-deterministic at position %d: %d != %d", i, v1, v2)
		}
	}
}

func TestSplitMix64FloatRange(t *testing.T) {
	rng := NewSplitMix64(42)
	for i := 0; i < 10000; i++ {
		v := rng.NextFloat()
		if v < 0.0 || v >= 1.0 {
			t.Errorf("value out of range [0,1): %f", v)
		}
	}
}

func TestUniformFloat32(t *testing.T) {
	rng := NewSplitMix64(42)
	data := make([]float32, 1000)
	rng.UniformFloat32(data, 0.9, 1.1)

	min, max := float32(2.0), float32(0.0)
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		if v < 0.9 || v >= 1.1 {
			t.Errorf("value out of range [0.9, 1.1): %f", v)
		}
	}
	t.Logf("Uniform [0.9, 1.1) range: min=%.4f, max=%.4f", min, max)
}

func TestPermutation(t *testing.T) {
	rng := NewSplitMix64(42)
	perm := rng.Permutation(10)

	// Check it's a valid permutation
	seen := make(map[int]bool)
	for _, v := range perm {
		if v < 0 || v >= 10 {
			t.Errorf("value out of range: %d", v)
		}
		if seen[v] {
			t.Errorf("duplicate value: %d", v)
		}
		seen[v] = true
	}
	if len(seen) != 10 {
		t.Errorf("expected 10 unique values, got %d", len(seen))
	}
}

func TestInversePermutation(t *testing.T) {
	rng := NewSplitMix64(42)
	perm := rng.Permutation(100)
	invPerm := InversePermutation(perm)

	// Applying perm then invPerm should be identity
	for i := 0; i < 100; i++ {
		if invPerm[perm[i]] != i {
			t.Errorf("inverse permutation broken at %d: perm[%d]=%d, invPerm[%d]=%d",
				i, i, perm[i], perm[i], invPerm[perm[i]])
		}
	}
}

func TestGenerateTransformMatrix(t *testing.T) {
	B := GenerateTransformMatrix(42, 4, 4)

	if len(B) != 16 {
		t.Errorf("expected 16 elements, got %d", len(B))
	}

	// All values should be in [0.9, 1.1)
	for i, v := range B {
		if float64(v) < 0.9 || float64(v) >= 1.1 {
			t.Errorf("B[%d]=%f out of range [0.9, 1.1)", i, v)
		}
	}

	// Deterministic
	B2 := GenerateTransformMatrix(42, 4, 4)
	for i := range B {
		if B[i] != B2[i] {
			t.Errorf("non-deterministic at B[%d]: %f != %f", i, B[i], B2[i])
		}
	}

	t.Logf("First 4 values: %v", B[:4])
}

func TestInverseTransformRoundTrip(t *testing.T) {
	// Generate B, create fake data, transform, inverse transform, check
	B := GenerateTransformMatrix(42, 4, 4)

	original := []float32{1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 7.0, 8.0}

	// Transform: stored = original * B
	stored := make([]float32, len(original))
	for i := range original {
		stored[i] = original[i] * B[i]
	}

	// Inverse: recovered = stored / B
	recovered := make([]float32, len(stored))
	copy(recovered, stored)
	InverseTransform(recovered, B)

	for i := range original {
		diff := math.Abs(float64(recovered[i]) - float64(original[i]))
		if diff > 1e-6 {
			t.Errorf("round-trip error at %d: original=%f, recovered=%f, diff=%e",
				i, original[i], recovered[i], diff)
		}
	}
}

func TestUnshuffleRows(t *testing.T) {
	// Create a 4x3 matrix
	data := []float32{
		0, 1, 2, // row 0
		3, 4, 5, // row 1
		6, 7, 8, // row 2
		9, 10, 11, // row 3
	}
	// Permutation: [0->2, 1->0, 2->3, 3->1]
	perm := []int{2, 0, 3, 1}
	invPerm := InversePermutation(perm)

	result := UnshuffleRows(data, 4, 3, invPerm)

	// After unshuffle: invPerm = [1, 3, 0, 2]
	// result[0] = data[1] = [3,4,5]
	// result[1] = data[3] = [9,10,11]
	// result[2] = data[0] = [0,1,2]
	// result[3] = data[2] = [6,7,8]
	expected := []float32{3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8}

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("unshuffle[%d]: got %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestL2NormalizeRows(t *testing.T) {
	data := []float32{
		3, 4, 0, // ||row|| = 5, normalized = [0.6, 0.8, 0]
		0, 0, 0, // ||row|| = 0, should use epsilon
	}
	L2NormalizeRows(data, 2, 3)

	// Row 0: [3/5, 4/5, 0] = [0.6, 0.8, 0]
	if math.Abs(float64(data[0])-0.6) > 1e-6 {
		t.Errorf("data[0]: got %f, want 0.6", data[0])
	}
	if math.Abs(float64(data[1])-0.8) > 1e-6 {
		t.Errorf("data[1]: got %f, want 0.8", data[1])
	}

	// Row 1: should be all near-zero (divided by epsilon)
	// Just check it doesn't contain NaN or Inf
	for i := 3; i < 6; i++ {
		if math.IsNaN(float64(data[i])) || math.IsInf(float64(data[i]), 0) {
			t.Errorf("data[%d] is invalid: %f", i, data[i])
		}
	}
}
