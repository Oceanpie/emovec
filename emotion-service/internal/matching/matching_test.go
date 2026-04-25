package matching

import (
	"math"
	"testing"
)

func TestL2Normalize(t *testing.T) {
	// [3, 4] → norm 5 → [0.6, 0.8]
	v := []float32{3, 4}
	L2Normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("L2Normalize([3,4]) = %v, want [0.6, 0.8]", v)
	}
	// Verify norm is 1
	norm := L2Norm(v)
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("norm after L2Normalize = %v, want 1.0", norm)
	}

	// Zero vector → no NaN, norm clamped to NormEpsilon
	z := []float32{0, 0, 0}
	L2Normalize(z)
	for i, x := range z {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Errorf("L2Normalize zero vector produced NaN/Inf at index %d: %v", i, x)
		}
	}
	// Result should be [0, 0, 0] since 0/epsilon = 0
	for _, x := range z {
		if x != 0 {
			t.Errorf("L2Normalize zero vector should yield zeros, got %v", z)
			break
		}
	}
}

func TestDotProduct(t *testing.T) {
	// Orthogonal: [1,0] · [0,1] = 0
	if v := DotProduct([]float32{1, 0}, []float32{0, 1}); math.Abs(v) > 1e-10 {
		t.Errorf("DotProduct([1,0],[0,1]) = %v, want 0", v)
	}

	// Parallel: [1,0] · [1,0] = 1
	if v := DotProduct([]float32{1, 0}, []float32{1, 0}); math.Abs(v-1.0) > 1e-10 {
		t.Errorf("DotProduct([1,0],[1,0]) = %v, want 1", v)
	}

	// General: [1,2,3] · [4,5,6] = 4+10+18 = 32
	if v := DotProduct([]float32{1, 2, 3}, []float32{4, 5, 6}); math.Abs(v-32.0) > 1e-10 {
		t.Errorf("DotProduct([1,2,3],[4,5,6]) = %v, want 32", v)
	}
}

func TestTopK(t *testing.T) {
	sims := []float64{0.5, 0.9, 0.3, 0.7, 0.1}
	values, indices := TopK(sims, 3)

	expectedValues := []float64{0.9, 0.7, 0.5}
	expectedIndices := []int{1, 3, 0}

	for i := range values {
		if math.Abs(values[i]-expectedValues[i]) > 1e-10 {
			t.Errorf("TopK values[%d] = %v, want %v", i, values[i], expectedValues[i])
		}
		if indices[i] != expectedIndices[i] {
			t.Errorf("TopK indices[%d] = %v, want %v", i, indices[i], expectedIndices[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	// [0.8, 0.6, 0.3] tau=0.5
	// max = 0.8
	// exps: exp(0/0.5)=1, exp(-0.2/0.5)=exp(-0.4), exp(-0.5/0.5)=exp(-1)
	input := []float64{0.8, 0.6, 0.3}
	result := Softmax(input, 0.5)

	// Verify sum to 1
	var sum float64
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Softmax sum = %v, want 1.0", sum)
	}

	// Verify descending order (largest input → largest weight)
	if result[0] <= result[1] || result[1] <= result[2] {
		t.Errorf("Softmax should be descending for descending input, got %v", result)
	}

	// Verify exact values
	exp0 := 1.0
	exp1 := math.Exp(-0.4)
	exp2 := math.Exp(-1.0)
	total := exp0 + exp1 + exp2
	if math.Abs(result[0]-exp0/total) > 1e-10 {
		t.Errorf("Softmax[0] = %v, want %v", result[0], exp0/total)
	}
	if math.Abs(result[1]-exp1/total) > 1e-10 {
		t.Errorf("Softmax[1] = %v, want %v", result[1], exp1/total)
	}
	if math.Abs(result[2]-exp2/total) > 1e-10 {
		t.Errorf("Softmax[2] = %v, want %v", result[2], exp2/total)
	}

	// Single value → [1.0]
	single := Softmax([]float64{2.0}, 0.5)
	if len(single) != 1 || math.Abs(single[0]-1.0) > 1e-10 {
		t.Errorf("Softmax single = %v, want [1.0]", single)
	}

	// Empty → nil
	if empty := Softmax(nil, 0.5); empty != nil {
		t.Errorf("Softmax nil = %v, want nil", empty)
	}
}

func TestWeightedSum(t *testing.T) {
	// 3 entries, 2 dimensions
	// labels: [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
	// weights: [0.5, 0.3, 0.2]
	// indices: [0, 1, 2]
	// Expected: dim0 = 0.5*1.0 + 0.3*0.0 + 0.2*0.5 = 0.6
	//           dim1 = 0.5*0.0 + 0.3*1.0 + 0.2*0.5 = 0.4
	labels := []float32{1.0, 0.0, 0.0, 1.0, 0.5, 0.5}
	dims := []string{"a", "b"}
	weights := []float64{0.5, 0.3, 0.2}
	indices := []int{0, 1, 2}

	scores := WeightedSum(labels, dims, weights, indices, 2)

	if math.Abs(scores["a"]-0.6) > 1e-10 {
		t.Errorf("WeightedSum a = %v, want 0.6", scores["a"])
	}
	if math.Abs(scores["b"]-0.4) > 1e-10 {
		t.Errorf("WeightedSum b = %v, want 0.4", scores["b"])
	}
}

func TestMatch(t *testing.T) {
	// 3 prototype vectors in 4D (already normalized)
	// proto0: [1,0,0,0], proto1: [0,1,0,0], proto2: [0,0,1,0]
	protoVecs := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	// Query: [1,0,0,0] — matches proto0 exactly
	queryVec := []float32{1, 0, 0, 0}

	// 2 dims, labels: [[1.0,0.0],[0.0,1.0],[0.5,0.5]]
	labels := []float32{1.0, 0.0, 0.0, 1.0, 0.5, 0.5}
	dims := []string{"a", "b"}

	result := Match(protoVecs, labels, dims, queryVec, 2, 0.5, 4, 2)

	// Top match should be index 0 with similarity 1.0
	if len(result.Matches) != 2 {
		t.Fatalf("Match returned %d matches, want 2", len(result.Matches))
	}
	if math.Abs(result.Matches[0].Similarity-1.0) > 1e-10 {
		t.Errorf("Match[0].Similarity = %v, want 1.0", result.Matches[0].Similarity)
	}
	if result.Matches[0].Rank != 1 {
		t.Errorf("Match[0].Rank = %d, want 1", result.Matches[0].Rank)
	}

	// Second match should be proto1 or proto2 with similarity 0
	if math.Abs(result.Matches[1].Similarity) > 1e-10 {
		t.Errorf("Match[1].Similarity = %v, want 0.0", result.Matches[1].Similarity)
	}

	// Weights should sum to 1
	var weightSum float64
	for _, m := range result.Matches {
		weightSum += m.Weight
	}
	if math.Abs(weightSum-1.0) > 1e-10 {
		t.Errorf("Weight sum = %v, want 1.0", weightSum)
	}

	// With similarity 1.0 and 0.0, softmax(tau=0.5):
	// exp((1-1)/0.5)=1, exp((0-1)/0.5)=exp(-2)
	// weight0 = 1/(1+exp(-2)), weight1 = exp(-2)/(1+exp(-2))
	exp2 := math.Exp(-2)
	w0 := 1.0 / (1.0 + exp2)
	w1 := exp2 / (1.0 + exp2)

	if math.Abs(result.Matches[0].Weight-w0) > 1e-10 {
		t.Errorf("Match[0].Weight = %v, want %v", result.Matches[0].Weight, w0)
	}
	if math.Abs(result.Matches[1].Weight-w1) > 1e-10 {
		t.Errorf("Match[1].Weight = %v, want %v", result.Matches[1].Weight, w1)
	}

	// Score for dim "a" should be w0*1.0 + w1*0.0 = w0
	if math.Abs(result.Scores["a"]-(math.Round(w0*10000)/10000)) > 1e-10 {
		t.Errorf("Scores[a] = %v, want %v", result.Scores["a"], math.Round(w0*10000)/10000)
	}
}
