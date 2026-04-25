package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	Matching  MatchingConfig  `yaml:"matching"`
	Data      DataConfig      `yaml:"data"`
}

type ServerConfig struct {
	Addr         string        `yaml:"addr"`
	ReadTimeout  time.Duration `yaml:"read_timeout"`
	WriteTimeout time.Duration `yaml:"write_timeout"`
}

type EmbeddingConfig struct {
	Providers   []ProviderConfig `yaml:"providers"`
	Instruction string           `yaml:"instruction"`
}

type ProviderConfig struct {
	Name      string        `yaml:"name"`
	BaseURL   string        `yaml:"base_url"`
	APIKey    string        `yaml:"api_key"`
	Model     string        `yaml:"model"`
	BatchSize int           `yaml:"batch_size"`
	Timeout   time.Duration `yaml:"timeout"`
}

type MatchingConfig struct {
	TopK   int     `yaml:"top_k"`
	Tau    float64 `yaml:"tau"`
	Scheme string  `yaml:"scheme"`
}

type DataConfig struct {
	VectorsPath string `yaml:"vectors_path"`
	LabelsPath  string `yaml:"labels_path"`
	ModelPath   string `yaml:"model_path"` // safetensors path (Phase 2), empty = use raw binary
}

func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	expanded := os.ExpandEnv(string(raw))
	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *Config) Validate() error {
	if len(c.Embedding.Providers) == 0 {
		return fmt.Errorf("at least one embedding provider required")
	}
	if c.Matching.Scheme != "plutchik" && c.Matching.Scheme != "original" {
		return fmt.Errorf("invalid scheme %q: must be plutchik or original", c.Matching.Scheme)
	}
	if c.Matching.TopK <= 0 {
		return fmt.Errorf("top_k must be positive")
	}
	if c.Matching.Tau <= 0 {
		return fmt.Errorf("tau must be positive")
	}
	return nil
}
