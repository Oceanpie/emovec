package handler

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"emovec/internal/config"
	"emovec/internal/embedding"
	"emovec/internal/store"
	"emovec/internal/transform"
)

func setupTestAnalyzer(t *testing.T) (*Analyzer, *httptest.Server) {
	t.Helper()

	// Mock embedding server — returns a 1024-dim non-zero vector
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		emb := make([]float64, 1024)
		for i := range emb {
			emb[i] = 0.01 * float64(i%100)
		}
		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": emb, "index": 0},
			},
			"model": "test",
		}
		json.NewEncoder(w).Encode(resp)
	}))

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	s, err := store.LoadStoreFromSafetensors(
		"../../data/model.safetensors",
		"../../data/prototype_labels.json",
		transform.TransformSeed,
		logger,
	)
	if err != nil {
		t.Fatalf("load store: %v", err)
	}

	client := embedding.NewClient([]embedding.ProviderConfig{
		{Name: "mock", BaseURL: mockServer.URL, APIKey: "test", Model: "test", BatchSize: 64, Timeout: 5 * time.Second},
	}, "test instruction", logger)

	cfg := config.MatchingConfig{TopK: 7, Tau: 0.5, Scheme: "plutchik"}

	return NewAnalyzer(s, client, cfg, logger), mockServer
}

func TestHandleAnalyze_SingleText(t *testing.T) {
	analyzer, mockServer := setupTestAnalyzer(t)
	defer mockServer.Close()

	body, _ := json.Marshal(AnalyzeRequest{Text: "我真的很想你"})
	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(body))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp SingleResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if !resp.OK {
		t.Fatalf("expected ok, got error: %s", resp.Error)
	}
	if len(resp.Result) != 8 {
		t.Errorf("result dims = %d, want 8", len(resp.Result))
	}
}

func TestHandleAnalyze_BatchText(t *testing.T) {
	analyzer, mockServer := setupTestAnalyzer(t)
	defer mockServer.Close()

	body, _ := json.Marshal(AnalyzeRequest{Texts: []string{"hello", "world"}})
	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(body))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp BatchResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Results) != 2 {
		t.Fatalf("results count = %d, want 2", len(resp.Results))
	}
	for i, r := range resp.Results {
		if !r.OK {
			t.Errorf("results[%d] not ok: %s", i, r.Error)
		}
		if len(r.Result) != 8 {
			t.Errorf("results[%d] dims = %d, want 8", i, len(r.Result))
		}
	}
}

func TestHandleAnalyze_InvalidRequest(t *testing.T) {
	analyzer, _ := setupTestAnalyzer(t)

	// Both text and texts provided
	body, _ := json.Marshal(AnalyzeRequest{Text: "hello", Texts: []string{"world"}})
	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(body))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleAnalyze_EmptyBody(t *testing.T) {
	analyzer, _ := setupTestAnalyzer(t)

	// Neither text nor texts provided
	body, _ := json.Marshal(AnalyzeRequest{})
	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(body))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleAnalyze_InvalidJSON(t *testing.T) {
	analyzer, _ := setupTestAnalyzer(t)

	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleAnalyze_MethodNotAllowed(t *testing.T) {
	analyzer, _ := setupTestAnalyzer(t)

	req := httptest.NewRequest(http.MethodGet, "/analyze", nil)
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandleAnalyze_OverrideParams(t *testing.T) {
	analyzer, mockServer := setupTestAnalyzer(t)
	defer mockServer.Close()

	scheme := "original"
	topK := 5
	tau := 0.3
	body, _ := json.Marshal(AnalyzeRequest{
		Text:   "test",
		Scheme: &scheme,
		TopK:   &topK,
		Tau:    &tau,
	})
	req := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(body))
	w := httptest.NewRecorder()

	analyzer.HandleAnalyze(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp SingleResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if !resp.OK {
		t.Fatalf("expected ok, got error: %s", resp.Error)
	}
	// Original scheme has Chinese dimension names
	if _, ok := resp.Result["高兴"]; !ok {
		t.Errorf("expected '高兴' in result for original scheme, got keys: %v", resp.Result)
	}
}

func TestHandleHealth(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	HandleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp HealthResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "ok" {
		t.Errorf("status = %q, want ok", resp.Status)
	}
}
