package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"emovec/internal/config"
	"emovec/internal/embedding"
	"emovec/internal/handler"
	"emovec/internal/matching"
	"emovec/internal/store"
	"emovec/internal/transform"

	"gopkg.in/yaml.v3"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	schemeFlag := flag.String("scheme", "", "emotion scheme (plutchik|original), overrides config")
	topKFlag := flag.Int("top-k", 0, "top-K for matching, overrides config")
	tauFlag := flag.Float64("tau", 0, "softmax temperature, overrides config")
	flag.Parse()

	// Remaining args after flags = text to analyze (CLI mode)
	cliText := flag.Arg(0)

	// 1. Init logger: CLI mode → stderr + WARN, server mode → stdout + INFO
	logLevel := slog.LevelInfo
	logWriter := os.Stdout
	if cliText != "" {
		logLevel = slog.LevelWarn
		logWriter = os.Stderr
	}
	logger := slog.New(slog.NewTextHandler(logWriter, &slog.HandlerOptions{Level: logLevel}))
	slog.SetDefault(logger)

	// 2. Auto-load .env (set env vars if not already set)
	loadDotEnv(logger)
	cfgData, err := os.ReadFile(*configPath)
	if err != nil {
		logger.Error("failed to load config", "error", err, "path", *configPath)
		os.Exit(1)
	}
	expanded := os.ExpandEnv(string(cfgData))
	var cfg config.Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		logger.Error("failed to parse config", "error", err)
		os.Exit(1)
	}
	if err := cfg.Validate(); err != nil {
		logger.Error("invalid config", "error", err)
		os.Exit(1)
	}
	logger.Info("config loaded",
		"addr", cfg.Server.Addr,
		"providers", len(cfg.Embedding.Providers),
		"scheme", cfg.Matching.Scheme,
	)

	// 3. Load prototype store (embedded safetensors → filesystem fallback)
	logger.Info("loading prototype store...")
	var st *store.Store
	if cfg.Data.ModelPath != "" {
		// Try embedded safetensors first
		modelData, embedErr := embeddedData.ReadFile(cfg.Data.ModelPath)
		if embedErr == nil {
			// Write to temp file for safetensors parser (needs file seek)
			tmpFile, tmpErr := os.CreateTemp("", "model-*.safetensors")
			if tmpErr == nil {
				tmpFile.Write(modelData)
				tmpFile.Close()
				defer os.Remove(tmpFile.Name())

				// Also try embedded dims
				var dimsPath string
				if dimsData, dErr := embeddedData.ReadFile(cfg.Data.LabelsPath); dErr == nil {
					dimsTmp, _ := os.CreateTemp("", "labels-*.json")
					if dimsTmp != nil {
						dimsTmp.Write(dimsData)
						dimsTmp.Close()
						defer os.Remove(dimsTmp.Name())
						dimsPath = dimsTmp.Name()
					}
				}
				if dimsPath == "" {
					dimsPath = cfg.Data.LabelsPath
				}

				st, err = store.LoadStoreFromSafetensors(
					tmpFile.Name(), dimsPath,
					transform.TransformSeed, logger,
				)
				logger.Info("loaded from embedded safetensors")
			}
		}
		if st == nil {
			// Fallback to filesystem
			st, err = store.LoadStoreFromSafetensors(
				cfg.Data.ModelPath, cfg.Data.LabelsPath,
				transform.TransformSeed, logger,
			)
		}
	} else {
		// Load from raw binary (Phase 1: clean)
		st, err = store.LoadStore(cfg.Data.VectorsPath, cfg.Data.LabelsPath, logger)
	}
	if err != nil {
		logger.Error("failed to load store", "error", err)
		os.Exit(1)
	}

	// 4. Init embedding client
	providerConfigs := make([]embedding.ProviderConfig, len(cfg.Embedding.Providers))
	for i, p := range cfg.Embedding.Providers {
		providerConfigs[i] = embedding.ProviderConfig{
			Name:      p.Name,
			BaseURL:   p.BaseURL,
			APIKey:    p.APIKey,
			Model:     p.Model,
			BatchSize: p.BatchSize,
			Timeout:   p.Timeout,
		}
	}
	client := embedding.NewClient(providerConfigs, cfg.Embedding.Instruction, logger)

	// 5. Resolve effective matching params (CLI flags override config)
	scheme := cfg.Matching.Scheme
	if *schemeFlag != "" {
		scheme = *schemeFlag
	}
	topK := cfg.Matching.TopK
	if *topKFlag > 0 {
		topK = *topKFlag
	}
	tau := cfg.Matching.Tau
	if *tauFlag > 0 {
		tau = *tauFlag
	}

	// 6. CLI mode or server mode
	if cliText != "" {
		runCLI(cliText, st, client, scheme, topK, tau, logger)
		return
	}

	// Server mode
	analyzer := handler.NewAnalyzer(st, client, cfg.Matching, logger)

	// 6. Setup HTTP routes
	mux := http.NewServeMux()
	mux.HandleFunc("POST /analyze", analyzer.HandleAnalyze)
	mux.HandleFunc("GET /health", handler.HandleHealth)

	// 7. Start server with graceful shutdown
	readTimeout := cfg.Server.ReadTimeout
	writeTimeout := cfg.Server.WriteTimeout
	if readTimeout == 0 {
		readTimeout = 15 * time.Second
	}
	if writeTimeout == 0 {
		writeTimeout = 30 * time.Second
	}

	srv := &http.Server{
		Addr:         cfg.Server.Addr,
		Handler:      mux,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		logger.Info("server starting",
			"addr", srv.Addr,
			"go_version", runtime.Version(),
		)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	logger.Info("shutting down", "signal", sig)

	// Graceful shutdown with 30s timeout
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("server forced to shutdown", "error", err)
	}
	logger.Info("server stopped")
}

// runCLI runs a single analysis and prints JSON to stdout.
func runCLI(text string, st *store.Store, client *embedding.Client, scheme string, topK int, tau float64, logger *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vec, err := client.Embed(ctx, text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedding failed: %v\n", err)
		os.Exit(1)
	}

	// L2 normalize query
	queryVec := make([]float32, len(vec))
	copy(queryVec, vec)
	matching.L2Normalize(queryVec)

	labels := st.GetLabels(scheme)
	dims := st.GetDims(scheme)

	result := matching.Match(
		st.Vectors, labels, dims, queryVec,
		topK, tau,
		store.VectorDim, store.NumDims,
	)

	output := map[string]interface{}{
		"ok":     true,
		"text":   text,
		"scheme": scheme,
		"result": result.Scores,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "json encode failed: %v\n", err)
		os.Exit(1)
	}
}

// loadDotEnv reads .env from current dir or parent dirs, sets env vars (no override).
func loadDotEnv(logger *slog.Logger) {
	dir, _ := os.Getwd()
	for i := 0; i < 5; i++ {
		p := dir + "/.env"
		if data, err := os.ReadFile(p); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "#") || !strings.Contains(line, "=") {
					continue
				}
				k, v, _ := strings.Cut(line, "=")
				k = strings.TrimSpace(k)
				v = strings.TrimSpace(v)
				os.Setenv(k, v)
			}
			logger.Debug("loaded .env", "path", p)
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
}
