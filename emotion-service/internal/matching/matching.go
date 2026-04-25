package matching

import (
	"math"
	"sort"
)

// NormEpsilon prevents division by zero in L2Normalize.
const NormEpsilon = 1e-10

// MatchResult holds the full matching result.
type MatchResult struct {
	Scores  map[string]float64
	Matches []MatchInfo
}

// MatchInfo holds details for a single matched prototype.
type MatchInfo struct {
	Rank       int
	Title      string
	Similarity float64
	Weight     float64
}

// L2Normalize normalizes a float32 vector in-place using float64 accumulator.
// Clamps norm to NormEpsilon to prevent division by zero (matches Python: max(norm, 1e-10)).
func L2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm < NormEpsilon {
		norm = NormEpsilon
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
}

// L2Norm returns the L2 norm of a float32 vector using float64 accumulator.
func L2Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

// DotProduct computes the dot product of two float32 vectors using float64 accumulator.
func DotProduct(a, b []float32) float64 {
	var sum float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

// TopK finds indices of k largest values in sims.
// Returns values and indices sorted descending.
// Uses simple sort — O(n log n) is fine for n=153.
func TopK(sims []float64, k int) (values []float64, indices []int) {
	n := len(sims)
	if k > n {
		k = n
	}
	type iv struct {
		i int
		v float64
	}
	all := make([]iv, n)
	for i, v := range sims {
		all[i] = iv{i, v}
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].v > all[j].v // descending
	})
	values = make([]float64, k)
	indices = make([]int, k)
	for i := 0; i < k; i++ {
		values[i] = all[i].v
		indices[i] = all[i].i
	}
	return
}

// Softmax applies temperature-scaled softmax.
// For numerical stability, subtracts max before exp.
func Softmax(values []float64, tau float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	exps := make([]float64, len(values))
	var sum float64
	for i, v := range values {
		exps[i] = math.Exp((v - maxVal) / tau)
		sum += exps[i]
	}
	result := make([]float64, len(values))
	for i := range exps {
		result[i] = exps[i] / sum
	}
	return result
}

// WeightedSum computes weighted sum of labels for top-K matches.
// labels: flat slice of [n_entries * nDims]float32 in row-major order
// dims: dimension names (nDims elements)
// weights: softmax weights for top-K
// indices: top-K indices into labels
// nDims: number of dimensions
// Returns map[string]float64 with each value rounded to 4 decimal places.
func WeightedSum(labels []float32, dims []string, weights []float64, indices []int, nDims int) map[string]float64 {
	scores := make(map[string]float64)
	for d := 0; d < nDims; d++ {
		var val float64
		for i, idx := range indices {
			labelVal := float64(labels[idx*nDims+d])
			val += weights[i] * labelVal
		}
		scores[dims[d]] = math.Round(val*10000) / 10000
	}
	return scores
}

// Match performs the full matching pipeline on a pre-normalized query vector.
// protoVecs: pre-normalized prototype vectors [n_entries * vectorDim]float32 row-major
// labels: label values [n_entries * nDims]float32 row-major
// dims: dimension names
// queryVec: pre-normalized query vector
// topK: number of top matches to consider
// tau: softmax temperature
// vectorDim: dimension of embedding vectors
// nDims: number of label dimensions (8)
func Match(protoVecs []float32, labels []float32, dims []string, queryVec []float32, topK int, tau float64, vectorDim int, nDims int) MatchResult {
	nEntries := len(protoVecs) / vectorDim
	sims := make([]float64, nEntries)
	for i := 0; i < nEntries; i++ {
		protoRow := protoVecs[i*vectorDim : (i+1)*vectorDim]
		sims[i] = DotProduct(queryVec, protoRow)
	}

	topValues, topIndices := TopK(sims, topK)
	weights := Softmax(topValues, tau)
	scores := WeightedSum(labels, dims, weights, topIndices, nDims)

	matches := make([]MatchInfo, len(topIndices))
	for i := range topIndices {
		matches[i] = MatchInfo{
			Rank:       i + 1,
			Similarity: topValues[i],
			Weight:     weights[i],
		}
	}

	return MatchResult{
		Scores:  scores,
		Matches: matches,
	}
}
