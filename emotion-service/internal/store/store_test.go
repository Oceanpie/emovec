package store

import (
	"log/slog"
	"os"
	"testing"

	"emovec/internal/matching"
	"emovec/internal/transform"
)

func loadTestStore(t *testing.T) *Store {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	s, err := LoadStoreFromSafetensors(
		"../../data/model.safetensors",
		"../../data/prototype_labels.json",
		transform.TransformSeed,
		logger,
	)
	if err != nil {
		t.Fatalf("LoadStoreFromSafetensors failed: %v", err)
	}
	return s
}

func TestLoadStore(t *testing.T) {
	s := loadTestStore(t)

	if s.NumEntries != 153 {
		t.Errorf("NumEntries = %d, want 153", s.NumEntries)
	}
	if len(s.Vectors) != 153*1024 {
		t.Errorf("len(Vectors) = %d, want %d", len(s.Vectors), 153*1024)
	}
	if len(s.Labels) != 153*8 {
		t.Errorf("len(Labels) = %d, want %d", len(s.Labels), 153*8)
	}
	if len(s.Dims) != 8 {
		t.Errorf("len(Dims) = %d, want 8", len(s.Dims))
	}
}

func TestVectorsNormalized(t *testing.T) {
	s := loadTestStore(t)

	// Check all vectors are unit-length
	for i := 0; i < s.NumEntries; i++ {
		row := s.Vectors[i*1024 : (i+1)*1024]
		norm := matching.L2Norm(row)
		if norm < 0.999 || norm > 1.001 {
			t.Errorf("vector %d has norm %f, expected ~1.0", i, norm)
		}
	}
}

func TestKnownLabelValues(t *testing.T) {
	s := loadTestStore(t)

	// ABHIMAN is entry 0, anger=0.75 in plutchik
	angerIdx := -1
	for i, d := range s.Dims {
		if d == "anger" {
			angerIdx = i
			break
		}
	}
	if angerIdx < 0 {
		t.Fatal("anger dimension not found")
	}

	got := s.Labels[0*8+angerIdx] // entry 0, anger dimension
	if got != 0.75 {
		t.Errorf("ABHIMAN anger = %f, want 0.75", got)
	}

	// Note: Titles are not populated in safetensors path (not stored in the file)
}

func TestGetLabelsByScheme(t *testing.T) {
	s := loadTestStore(t)

	plutchik := s.GetLabels("plutchik")
	original := s.GetLabels("original")
	if len(plutchik) != 153*8 {
		t.Errorf("plutchik labels len = %d, want %d", len(plutchik), 153*8)
	}
	if len(original) != 153*8 {
		t.Errorf("original labels len = %d, want %d", len(original), 153*8)
	}

	// Check dims
	pdims := s.GetDims("plutchik")
	odims := s.GetDims("original")
	if pdims[0] != "joy" {
		t.Errorf("plutchik dims[0] = %q, want joy", pdims[0])
	}
	if odims[0] != "高兴" {
		t.Errorf("original dims[0] = %q, want 高兴", odims[0])
	}
}
