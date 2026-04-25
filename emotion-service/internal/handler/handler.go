package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"emovec/internal/config"
	"emovec/internal/embedding"
	"emovec/internal/matching"
	"emovec/internal/store"
)

// Analyzer handles emotion analysis requests.
type Analyzer struct {
	store  *store.Store
	client *embedding.Client
	cfg    config.MatchingConfig
	logger *slog.Logger
}

// NewAnalyzer creates a new Analyzer.
func NewAnalyzer(s *store.Store, c *embedding.Client, cfg config.MatchingConfig, logger *slog.Logger) *Analyzer {
	return &Analyzer{
		store:  s,
		client: c,
		cfg:    cfg,
		logger: logger,
	}
}

// AnalyzeRequest is the JSON request body for POST /analyze.
type AnalyzeRequest struct {
	Text   string   `json:"text,omitempty"`
	Texts  []string `json:"texts,omitempty"`
	Scheme *string  `json:"scheme,omitempty"`
	TopK   *int     `json:"top_k,omitempty"`
	Tau    *float64 `json:"tau,omitempty"`
}

// SingleResponse is the response for a single text request.
type SingleResponse struct {
	OK     bool               `json:"ok"`
	Result map[string]float64 `json:"result,omitempty"`
	Error  string             `json:"error,omitempty"`
}

// BatchResponse is the response for a batch request.
type BatchResponse struct {
	Results []SingleResult `json:"results"`
}

// SingleResult is one item in the batch response.
type SingleResult struct {
	OK     bool               `json:"ok"`
	Result map[string]float64 `json:"result,omitempty"`
	Error  string             `json:"error,omitempty"`
}

// HealthResponse is the response for GET /health.
type HealthResponse struct {
	Status string `json:"status"`
}

// HandleAnalyze handles POST /analyze
func (a *Analyzer) HandleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, SingleResponse{OK: false, Error: "invalid JSON"})
		return
	}

	// Determine effective params
	scheme := a.cfg.Scheme
	if req.Scheme != nil {
		scheme = *req.Scheme
	}
	topK := a.cfg.TopK
	if req.TopK != nil && *req.TopK > 0 {
		topK = *req.TopK
	}
	tau := a.cfg.Tau
	if req.Tau != nil && *req.Tau > 0 {
		tau = *req.Tau
	}

	// Single text
	if req.Text != "" && len(req.Texts) == 0 {
		start := time.Now()
		result, err := a.analyzeOne(r.Context(), req.Text, scheme, topK, tau)
		if err != nil {
			a.logger.Error("analyze failed", "error", err, "text_len", len(req.Text))
			writeJSON(w, http.StatusOK, SingleResponse{OK: false, Error: err.Error()})
			return
		}
		a.logger.Info("analyzed", "text_len", len(req.Text), "duration", time.Since(start))
		writeJSON(w, http.StatusOK, SingleResponse{OK: true, Result: result})
		return
	}

	// Batch texts
	if len(req.Texts) > 0 && req.Text == "" {
		start := time.Now()

		embedResults := a.client.EmbedBatch(r.Context(), req.Texts)

		results := make([]SingleResult, len(req.Texts))
		for i, er := range embedResults {
			if er.Err != nil {
				results[i] = SingleResult{OK: false, Error: er.Err.Error()}
				continue
			}
			scores, err := a.matchOne(er.Vector, scheme, topK, tau)
			if err != nil {
				results[i] = SingleResult{OK: false, Error: err.Error()}
				continue
			}
			results[i] = SingleResult{OK: true, Result: scores}
		}

		a.logger.Info("batch analyzed", "count", len(req.Texts), "duration", time.Since(start))
		writeJSON(w, http.StatusOK, BatchResponse{Results: results})
		return
	}

	// Invalid: both or neither
	writeJSON(w, http.StatusBadRequest, SingleResponse{
		OK:    false,
		Error: "provide either 'text' (string) or 'texts' (array), not both",
	})
}

// HandleHealth handles GET /health
func HandleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
}

func (a *Analyzer) analyzeOne(ctx context.Context, text string, scheme string, topK int, tau float64) (map[string]float64, error) {
	vec, err := a.client.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}
	return a.matchOne(vec, scheme, topK, tau)
}

func (a *Analyzer) matchOne(vec []float32, scheme string, topK int, tau float64) (map[string]float64, error) {
	// L2 normalize the query vector (copy first to avoid mutating)
	queryVec := make([]float32, len(vec))
	copy(queryVec, vec)
	matching.L2Normalize(queryVec)

	labels := a.store.GetLabels(scheme)
	dims := a.store.GetDims(scheme)

	result := matching.Match(
		a.store.Vectors, labels, dims, queryVec,
		topK, tau,
		store.VectorDim, store.NumDims,
	)
	return result.Scores, nil
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
