package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeTestFile(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "test.yaml")
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	return p
}

const validConfig = `
server:
  addr: ":9090"
  read_timeout: 10s
  write_timeout: 20s
embedding:
  providers:
    - name: test
      base_url: "http://localhost:8000/v1"
      api_key: "test-key"
      model: "Qwen/Test"
      batch_size: 32
      timeout: 15s
  instruction: "test instruction"
matching:
  top_k: 5
  tau: 0.3
  scheme: "plutchik"
data:
  vectors_path: "data/vectors.bin"
  labels_path: "data/labels.json"
`

func TestLoadValidConfig(t *testing.T) {
	p := writeTestFile(t, validConfig)
	cfg, err := Load(p)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.Server.Addr != ":9090" {
		t.Errorf("Server.Addr = %q, want :9090", cfg.Server.Addr)
	}
	if cfg.Server.ReadTimeout != 10*time.Second {
		t.Errorf("ReadTimeout = %v, want 10s", cfg.Server.ReadTimeout)
	}
	if len(cfg.Embedding.Providers) != 1 {
		t.Fatalf("Providers count = %d, want 1", len(cfg.Embedding.Providers))
	}
	if cfg.Embedding.Providers[0].Name != "test" {
		t.Errorf("Provider.Name = %q, want test", cfg.Embedding.Providers[0].Name)
	}
	if cfg.Embedding.Providers[0].BatchSize != 32 {
		t.Errorf("Provider.BatchSize = %d, want 32", cfg.Embedding.Providers[0].BatchSize)
	}
	if cfg.Matching.TopK != 5 {
		t.Errorf("TopK = %d, want 5", cfg.Matching.TopK)
	}
	if cfg.Matching.Tau != 0.3 {
		t.Errorf("Tau = %f, want 0.3", cfg.Matching.Tau)
	}
	if cfg.Matching.Scheme != "plutchik" {
		t.Errorf("Scheme = %q, want plutchik", cfg.Matching.Scheme)
	}
	if cfg.Data.VectorsPath != "data/vectors.bin" {
		t.Errorf("VectorsPath = %q, want data/vectors.bin", cfg.Data.VectorsPath)
	}
}

func TestEnvVarExpansion(t *testing.T) {
	os.Setenv("TEST_EMOTION_API_KEY", "expanded-key-123")
	defer os.Unsetenv("TEST_EMOTION_API_KEY")

	content := `
server:
  addr: ":8080"
  read_timeout: 5s
  write_timeout: 10s
embedding:
  providers:
    - name: envtest
      base_url: "http://localhost/v1"
      api_key: "${TEST_EMOTION_API_KEY}"
      model: "test"
      batch_size: 8
      timeout: 5s
  instruction: "test"
matching:
  top_k: 3
  tau: 0.5
  scheme: "original"
data:
  vectors_path: "v.bin"
  labels_path: "l.json"
`
	p := writeTestFile(t, content)
	cfg, err := Load(p)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.Embedding.Providers[0].APIKey != "expanded-key-123" {
		t.Errorf("APIKey = %q, want expanded-key-123", cfg.Embedding.Providers[0].APIKey)
	}
}

func TestValidateNoProviders(t *testing.T) {
	content := `
server:
  addr: ":8080"
embedding:
  providers: []
  instruction: "test"
matching:
  top_k: 3
  tau: 0.5
  scheme: "plutchik"
data:
  vectors_path: "v.bin"
  labels_path: "l.json"
`
	p := writeTestFile(t, content)
	_, err := Load(p)
	if err == nil {
		t.Fatal("expected error for no providers")
	}
	if err.Error() != "at least one embedding provider required" {
		t.Errorf("error = %q, want provider error", err.Error())
	}
}

func TestValidateInvalidScheme(t *testing.T) {
	content := `
server:
  addr: ":8080"
embedding:
  providers:
    - name: x
      base_url: "http://x"
      api_key: "k"
      model: "m"
      batch_size: 1
      timeout: 1s
  instruction: "t"
matching:
  top_k: 3
  tau: 0.5
  scheme: "bogus"
data:
  vectors_path: "v"
  labels_path: "l"
`
	p := writeTestFile(t, content)
	_, err := Load(p)
	if err == nil {
		t.Fatal("expected error for invalid scheme")
	}
	want := `invalid scheme "bogus": must be plutchik or original`
	if err.Error() != want {
		t.Errorf("error = %q, want %q", err.Error(), want)
	}
}

func TestValidateZeroTopK(t *testing.T) {
	content := `
server:
  addr: ":8080"
embedding:
  providers:
    - name: x
      base_url: "http://x"
      api_key: "k"
      model: "m"
      batch_size: 1
      timeout: 1s
  instruction: "t"
matching:
  top_k: 0
  tau: 0.5
  scheme: "plutchik"
data:
  vectors_path: "v"
  labels_path: "l"
`
	p := writeTestFile(t, content)
	_, err := Load(p)
	if err == nil {
		t.Fatal("expected error for zero top_k")
	}
	if err.Error() != "top_k must be positive" {
		t.Errorf("error = %q, want top_k error", err.Error())
	}
}

func TestFileNotFound(t *testing.T) {
	_, err := Load("nonexistent.yaml")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}
