package embedding

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

const (
	maxRetries = 3
	retryDelay = 10 * time.Second
)

// EmbedResult holds the result of embedding a single text.
type EmbedResult struct {
	Vector []float32
	Err    error
}

// ProviderConfig holds configuration for a single embedding provider.
type ProviderConfig struct {
	Name      string
	BaseURL   string
	APIKey    string
	Model     string
	BatchSize int
	Timeout   time.Duration
}

// Client is an OpenAI-compatible embedding API client with multi-provider fallback.
type Client struct {
	providers   []provider
	instruction string
	logger      *slog.Logger
}

// apiResponse matches the OpenAI embeddings API response format.
type apiResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
}

// httpError wraps an HTTP error response.
type httpError struct {
	StatusCode int
	Body       string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Body)
}

// batchCountError indicates a batch response had wrong number of entries.
type batchCountError struct {
	Got  int
	Want int
}

func (e *batchCountError) Error() string {
	return fmt.Sprintf("batch response count mismatch: got %d, want %d", e.Got, e.Want)
}

// NewClient creates a new embedding client with the given provider configs.
func NewClient(configs []ProviderConfig, instruction string, logger *slog.Logger) *Client {
	providers := make([]provider, len(configs))
	for i, cfg := range configs {
		timeout := cfg.Timeout
		if timeout == 0 {
			timeout = 30 * time.Second
		}
		providers[i] = provider{
			ProviderConfig: cfg,
			client: &http.Client{
				Timeout: timeout,
				Transport: &http.Transport{
					MaxIdleConnsPerHost: 10,
					IdleConnTimeout:     90 * time.Second,
				},
			},
		}
	}
	return &Client{
		providers:   providers,
		instruction: instruction,
		logger:      logger,
	}
}

// Embed embeds a single text, trying providers in fallback order with retries.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	apiInput := fmt.Sprintf("Instruct: %s\nQuery:%s", c.instruction, text)
	return retryWithFallback(ctx, c.providers, c.logger, func(ctx context.Context, p provider) ([]float32, error) {
		return p.embedOne(ctx, apiInput)
	})
}

// EmbedBatch embeds multiple texts with batch API support and provider fallback.
// Returns one EmbedResult per input text, maintaining order.
// Individual failures are marked per-position (partial failure OK).
func (c *Client) EmbedBatch(ctx context.Context, texts []string) []EmbedResult {
	results := make([]EmbedResult, len(texts))
	if len(texts) == 0 {
		return results
	}

	prepared := make([]string, len(texts))
	for i, t := range texts {
		prepared[i] = fmt.Sprintf("Instruct: %s\nQuery:%s", c.instruction, t)
	}

	// Try batch with fallback across providers.
	vecs, err := retryWithFallback(ctx, c.providers, c.logger, func(ctx context.Context, p provider) ([][]float32, error) {
		return p.embedBatch(ctx, prepared)
	})

	if err != nil {
		// All providers failed for batch — fall back to individual calls.
		c.logger.Warn("batch embedding failed, falling back to individual calls", "error", err)
		for i, input := range prepared {
			vec, err := retryWithFallback(ctx, c.providers, c.logger, func(ctx context.Context, p provider) ([]float32, error) {
				return p.embedOne(ctx, input)
			})
			if err != nil {
				results[i] = EmbedResult{Err: err}
			} else {
				results[i] = EmbedResult{Vector: vec}
			}
		}
		return results
	}

	for i, vec := range vecs {
		results[i] = EmbedResult{Vector: vec}
	}
	return results
}

// retryWithFallback is a generic retry/fallback function that tries each provider
// with up to maxRetries attempts, falling back to the next provider on non-retryable errors.
func retryWithFallback[T any](
	ctx context.Context,
	providers []provider,
	logger *slog.Logger,
	fn func(ctx context.Context, p provider) (T, error),
) (T, error) {
	var zero T
	var lastErr error
	for _, p := range providers {
		for attempt := 0; attempt <= maxRetries; attempt++ {
			result, err := fn(ctx, p)
			if err == nil {
				return result, nil
			}
			lastErr = err
			if isNonRetryable(err) {
				logger.Warn("provider error, falling back", "provider", p.Name, "error", err)
				break // try next provider
			}
			if attempt < maxRetries {
				logger.Warn("provider error, retrying", "provider", p.Name, "attempt", attempt+1, "error", err)
				select {
				case <-ctx.Done():
					return zero, ctx.Err()
				case <-time.After(retryDelay):
				}
			}
		}
	}
	return zero, fmt.Errorf("all embedding providers failed: %w", lastErr)
}

// isHTTPError checks if err is an httpError with the given status code.
func isHTTPError(err error, code int) bool {
	if he, ok := err.(*httpError); ok {
		return he.StatusCode == code
	}
	return false
}

// isNonRetryable returns true if the error should not be retried
// (immediate fallback to next provider or individual calls).
func isNonRetryable(err error) bool {
	if isHTTPError(err, 400) || isHTTPError(err, 401) || isHTTPError(err, 403) || isHTTPError(err, 429) {
		return true
	}
	if _, ok := err.(*batchCountError); ok {
		return true
	}
	return false
}
