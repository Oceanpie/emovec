package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeTestSafetensors creates a minimal safetensors file for testing.
func writeTestSafetensors(t *testing.T, path string) {
	t.Helper()

	// Create two small tensors: (3, 4) and (4,)
	weightData := make([]float32, 12) // 3x4
	for i := range weightData {
		weightData[i] = float32(i) * 0.1
	}
	biasData := []float32{1.0, 2.0, 3.0, 4.0}

	// Convert to bytes
	weightBytes := make([]byte, len(weightData)*4)
	for i, v := range weightData {
		binary.LittleEndian.PutUint32(weightBytes[i*4:], math.Float32bits(v))
	}
	biasBytes := make([]byte, len(biasData)*4)
	for i, v := range biasData {
		binary.LittleEndian.PutUint32(biasBytes[i*4:], math.Float32bits(v))
	}

	// Build header
	header := map[string]interface{}{
		"__metadata__": map[string]string{"format": "pt"},
		"weight": map[string]interface{}{
			"dtype":       "F32",
			"shape":       []int{3, 4},
			"data_offsets": []int{0, len(weightBytes)},
		},
		"bias": map[string]interface{}{
			"dtype":       "F32",
			"shape":       []int{4},
			"data_offsets": []int{len(weightBytes), len(weightBytes) + len(biasBytes)},
		},
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	// Pad to 8-byte alignment with spaces (safetensors spec)
	padding := (8 - len(headerJSON)%8) % 8
	for i := 0; i < padding; i++ {
		headerJSON = append(headerJSON, ' ')
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Write header size
	binary.Write(f, binary.LittleEndian, uint64(len(headerJSON)))
	// Write header
	f.Write(headerJSON)
	// Write data
	f.Write(weightBytes)
	f.Write(biasBytes)
}

func TestLoadAndRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.safetensors")
	writeTestSafetensors(t, path)

	sf, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Check tensor names
	names := sf.TensorNames()
	if len(names) != 2 {
		t.Errorf("expected 2 tensors, got %d", len(names))
	}
	if !sf.HasTensor("weight") || !sf.HasTensor("bias") {
		t.Error("missing expected tensors")
	}

	// Read weight matrix
	data, rows, cols, err := sf.ReadF32Matrix("weight")
	if err != nil {
		t.Fatalf("ReadF32Matrix weight failed: %v", err)
	}
	if rows != 3 || cols != 4 {
		t.Errorf("weight shape: got (%d,%d), want (3,4)", rows, cols)
	}
	if len(data) != 12 {
		t.Errorf("weight data: got %d elements, want 12", len(data))
	}
	// Check first value
	if math.Abs(float64(data[0])-0.0) > 1e-6 {
		t.Errorf("weight[0]: got %f, want 0.0", data[0])
	}
	// Check last value
	if math.Abs(float64(data[11])-1.1) > 1e-6 {
		t.Errorf("weight[11]: got %f, want 1.1", data[11])
	}

	// Read bias vector
	biasData, shape, err := sf.ReadF32("bias")
	if err != nil {
		t.Fatalf("ReadF32 bias failed: %v", err)
	}
	if len(shape) != 1 || shape[0] != 4 {
		t.Errorf("bias shape: got %v, want [4]", shape)
	}
	if len(biasData) != 4 {
		t.Errorf("bias data: got %d elements, want 4", len(biasData))
	}
	if math.Abs(float64(biasData[0])-1.0) > 1e-6 {
		t.Errorf("bias[0]: got %f, want 1.0", biasData[0])
	}
}

func TestMissingTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.safetensors")
	writeTestSafetensors(t, path)

	sf, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	_, _, err = sf.ReadF32("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent tensor")
	}
}

func TestRealSafetensorsFile(t *testing.T) {
	// Test with the actual model.safetensors from the data directory
	path := filepath.Join("..", "..", "data", "model.safetensors")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("model.safetensors not found (run pack_prepare.py pack first)")
	}

	sf, err := Load(path)
	if err != nil {
		t.Fatalf("Load real safetensors failed: %v", err)
	}

	// Check expected tensors
	expectedTensors := []string{"layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"}
	for _, name := range expectedTensors {
		if !sf.HasTensor(name) {
			t.Errorf("missing tensor %q", name)
		}
	}

	// Read layers.0.weight (should be 1024x1024)
	w0, rows, cols, err := sf.ReadF32Matrix("layers.0.weight")
	if err != nil {
		t.Fatalf("ReadF32Matrix layers.0.weight failed: %v", err)
	}
	if rows != 1024 || cols != 1024 {
		t.Errorf("layers.0.weight shape: got (%d,%d), want (1024,1024)", rows, cols)
	}
	if len(w0) != 1024*1024 {
		t.Errorf("layers.0.weight data: got %d elements, want %d", len(w0), 1024*1024)
	}

	// Read layers.1.weight (should be 8x1024)
	w1, rows, cols, err := sf.ReadF32Matrix("layers.1.weight")
	if err != nil {
		t.Fatalf("ReadF32Matrix layers.1.weight failed: %v", err)
	}
	if rows != 8 || cols != 1024 {
		t.Errorf("layers.1.weight shape: got (%d,%d), want (8,1024)", rows, cols)
	}
	if len(w1) != 8*1024 {
		t.Errorf("layers.1.weight data: got %d elements, want %d", len(w1), 8*1024)
	}

	// Read bias vectors
	b0, _, err := sf.ReadF32("layers.0.bias")
	if err != nil {
		t.Fatalf("ReadF32 layers.0.bias failed: %v", err)
	}
	if len(b0) != 1024 {
		t.Errorf("layers.0.bias: got %d elements, want 1024", len(b0))
	}

	b1, _, err := sf.ReadF32("layers.1.bias")
	if err != nil {
		t.Fatalf("ReadF32 layers.1.bias failed: %v", err)
	}
	if len(b1) != 8 {
		t.Errorf("layers.1.bias: got %d elements, want 8", len(b1))
	}
}
