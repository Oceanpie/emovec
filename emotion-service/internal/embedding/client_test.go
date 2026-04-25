package embedding

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
)

func mockEmbedHandler(dim int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		// Support both single string and array input
		var count int
		switch v := req["input"].(type) {
		case string:
			count = 1
		case []interface{}:
			count = len(v)
		default:
			count = 1
		}

		data := make([]map[string]interface{}, count)
		for i := range data {
			data[i] = map[string]interface{}{
				"embedding": makeFloatSlice(dim),
				"index":     i,
			}
		}
		resp := map[string]interface{}{
			"data":  data,
			"model": "test-model",
		}
		json.NewEncoder(w).Encode(resp)
	}
}

func makeFloatSlice(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 0.01 * float64(i%100)
	}
	return v
}

func TestEmbed_SingleProvider(t *testing.T) {
	server := httptest.NewServer(mockEmbedHandler(1024))
	defer server.Close()

	client := NewClient([]ProviderConfig{
		{Name: "test", BaseURL: server.URL, APIKey: "test-key", Model: "test-model", BatchSize: 64, Timeout: 5 * time.Second},
	}, "test instruction", slog.New(slog.NewTextHandler(os.Stdout, nil)))

	vec, err := client.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(vec) != 1024 {
		t.Errorf("len(vec) = %d, want 1024", len(vec))
	}
}

func TestEmbed_FallbackOn429(t *testing.T) {
	callCount := 0
	failServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.WriteHeader(429)
		w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer failServer.Close()

	okServer := httptest.NewServer(mockEmbedHandler(1024))
	defer okServer.Close()

	client := NewClient([]ProviderConfig{
		{Name: "fail", BaseURL: failServer.URL, APIKey: "key1", Model: "m1", BatchSize: 64, Timeout: 5 * time.Second},
		{Name: "ok", BaseURL: okServer.URL, APIKey: "key2", Model: "m2", BatchSize: 64, Timeout: 5 * time.Second},
	}, "test instruction", slog.New(slog.NewTextHandler(os.Stdout, nil)))

	vec, err := client.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(vec) != 1024 {
		t.Errorf("len(vec) = %d, want 1024", len(vec))
	}
	if callCount != 1 {
		t.Errorf("fail server called %d times, want 1 (should fallback immediately on 429)", callCount)
	}
}

func TestEmbed_AllProvidersFail(t *testing.T) {
	failServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error":"internal"}`))
	}))
	defer failServer.Close()

	client := NewClient([]ProviderConfig{
		{Name: "fail", BaseURL: failServer.URL, APIKey: "key", Model: "m", BatchSize: 64, Timeout: 2 * time.Second},
	}, "test instruction", slog.New(slog.NewTextHandler(os.Stdout, nil)))

	_, err := client.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}
}

func TestEmbedBatch(t *testing.T) {
	server := httptest.NewServer(mockEmbedHandler(1024))
	defer server.Close()

	client := NewClient([]ProviderConfig{
		{Name: "test", BaseURL: server.URL, APIKey: "test-key", Model: "test-model", BatchSize: 64, Timeout: 5 * time.Second},
	}, "test instruction", slog.New(slog.NewTextHandler(os.Stdout, nil)))

	results := client.EmbedBatch(context.Background(), []string{"hello", "world", "test"})
	if len(results) != 3 {
		t.Fatalf("len(results) = %d, want 3", len(results))
	}
	for i, r := range results {
		if r.Err != nil {
			t.Errorf("results[%d].Err = %v", i, r.Err)
		}
		if len(r.Vector) != 1024 {
			t.Errorf("results[%d].Vector len = %d, want 1024", i, len(r.Vector))
		}
	}
}

func TestEmbedBatch_Empty(t *testing.T) {
	client := NewClient(nil, "test", slog.New(slog.NewTextHandler(os.Stdout, nil)))
	results := client.EmbedBatch(context.Background(), nil)
	if len(results) != 0 {
		t.Errorf("len(results) = %d, want 0", len(results))
	}
}

func TestEmbed_InputFormat(t *testing.T) {
	var gotInput string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)
		gotInput = req["input"].(string)

		resp := map[string]interface{}{
			"data": []map[string]interface{}{{"embedding": makeFloatSlice(10), "index": 0}},
			"model": "test",
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient([]ProviderConfig{
		{Name: "test", BaseURL: server.URL, APIKey: "key", Model: "m", BatchSize: 64, Timeout: 5 * time.Second},
	}, "my instruction", slog.New(slog.NewTextHandler(os.Stdout, nil)))

	client.Embed(context.Background(), "hello")

	expected := "Instruct: my instruction\nQuery:hello"
	if gotInput != expected {
		t.Errorf("input = %q, want %q", gotInput, expected)
	}
}

func TestEmbedBatch_PerProviderBatchSize(t *testing.T) {
	var mu sync.Mutex
	var apiCalls [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)
		inputs := req["input"].([]interface{})
		var inputStrs []string
		for _, v := range inputs {
			inputStrs = append(inputStrs, v.(string))
		}
		mu.Lock()
		apiCalls = append(apiCalls, inputStrs)
		mu.Unlock()

		data := make([]map[string]interface{}, len(inputs))
		for i := range inputs {
			data[i] = map[string]interface{}{
				"embedding": makeFloatSlice(8),
				"index":     i,
			}
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data":  data,
			"model": "test",
		})
	}))
	defer server.Close()

	client := NewClient([]ProviderConfig{
		{Name: "test", BaseURL: server.URL, APIKey: "key", Model: "m", BatchSize: 2, Timeout: 5 * time.Second},
	}, "inst", slog.New(slog.NewTextHandler(io.Discard, nil)))

	results := client.EmbedBatch(context.Background(), []string{"a", "b", "c", "d", "e"})
	if len(results) != 5 {
		t.Fatalf("len(results) = %d, want 5", len(results))
	}
	for i, r := range results {
		if r.Err != nil {
			t.Errorf("results[%d].Err = %v", i, r.Err)
		}
	}

	if len(apiCalls) != 3 {
		t.Fatalf("apiCalls = %d, want 3", len(apiCalls))
	}
	// Collect batch sizes (order may vary due to concurrent execution).
	sizes := make([]int, len(apiCalls))
	for i, call := range apiCalls {
		sizes[i] = len(call)
	}
	// Sort to get deterministic comparison.
	sort.Ints(sizes)
	if sizes[0] != 1 || sizes[1] != 2 || sizes[2] != 2 {
		t.Errorf("batch sizes: %v, want [1,2,2]", sizes)
	}
}
