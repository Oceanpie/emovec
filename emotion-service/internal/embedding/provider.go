package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"golang.org/x/sync/errgroup"
)

// provider is an unexported embedding API provider (OpenAI-compatible).
type provider struct {
	ProviderConfig
	client *http.Client
}

// callAPI is the shared HTTP helper for both single and batch embedding requests.
func (p *provider) callAPI(ctx context.Context, reqBody map[string]interface{}) (*apiResponse, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := strings.TrimRight(p.BaseURL, "/") + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, &httpError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var result apiResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	return &result, nil
}

// embedOne calls callAPI with a single input string and extracts one vector.
func (p *provider) embedOne(ctx context.Context, input string) ([]float32, error) {
	reqBody := map[string]interface{}{
		"model":           p.Model,
		"input":           input,
		"encoding_format": "float",
	}
	resp, err := p.callAPI(ctx, reqBody)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("empty response from embedding API")
	}
	vec := make([]float32, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

// embedBatch embeds multiple inputs, internally chunking by BatchSize.
// It uses errgroup with SetLimit for sub-chunk concurrency.
func (p *provider) embedBatch(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	batchSize := p.BatchSize
	if batchSize <= 0 {
		batchSize = 8
	}

	// If fits in one batch, just send it directly.
	if len(inputs) <= batchSize {
		return p.sendBatchRequest(ctx, inputs)
	}

	// Split into sub-chunks and dispatch concurrently.
	type chunk struct{ start, end int }
	var chunks []chunk
	for i := 0; i < len(inputs); i += batchSize {
		end := i + batchSize
		if end > len(inputs) {
			end = len(inputs)
		}
		chunks = append(chunks, chunk{i, end})
	}

	results := make([][][]float32, len(chunks))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(5)

	for ci, ch := range chunks {
		ci, ch := ci, ch
		g.Go(func() error {
			vecs, err := p.sendBatchRequest(gctx, inputs[ch.start:ch.end])
			if err != nil {
				return err
			}
			results[ci] = vecs
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	// Reassemble in order.
	all := make([][]float32, 0, len(inputs))
	for _, r := range results {
		all = append(all, r...)
	}
	return all, nil
}

// sendBatchRequest sends a single batch API request and returns vectors in input order.
func (p *provider) sendBatchRequest(ctx context.Context, inputs []string) ([][]float32, error) {
	reqBody := map[string]interface{}{
		"model":           p.Model,
		"input":           inputs,
		"encoding_format": "float",
	}
	resp, err := p.callAPI(ctx, reqBody)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(inputs) {
		return nil, &batchCountError{Got: len(resp.Data), Want: len(inputs)}
	}
	sort.Slice(resp.Data, func(i, j int) bool {
		return resp.Data[i].Index < resp.Data[j].Index
	})
	vecs := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vec := make([]float32, len(d.Embedding))
		for j, v := range d.Embedding {
			vec[j] = float32(v)
		}
		vecs[i] = vec
	}
	return vecs, nil
}
