package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// TensorInfo describes a single tensor in the safetensors file header.
type TensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"` // [start, end] relative to data buffer
}

// File represents a parsed safetensors file.
type File struct {
	tensors  map[string]TensorInfo
	data     []byte  // raw data buffer (everything after header)
	offset   int64   // byte offset where data buffer starts in the file
	filePath string // original file path for lazy loading
}

// Load reads and parses a safetensors file.
// The header is parsed immediately; tensor data is read lazily.
func Load(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer f.Close()

	// Read header size (uint64 LE)
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("read header size: %w", err)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := f.Read(headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Parse header
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %q: %w", name, err)
		}
		tensors[name] = info
	}

	dataOffset := int64(8 + headerSize)

	return &File{
		tensors:  tensors,
		data:     nil, // lazy loaded
		offset:   dataOffset,
		filePath: path,
	}, nil
}

// TensorNames returns all tensor names in the file.
func (sf *File) TensorNames() []string {
	names := make([]string, 0, len(sf.tensors))
	for name := range sf.tensors {
		names = append(names, name)
	}
	return names
}

// HasTensor checks if a tensor with the given name exists.
func (sf *File) HasTensor(name string) bool {
	_, ok := sf.tensors[name]
	return ok
}

// TensorShape returns the shape of the named tensor.
func (sf *File) TensorShape(name string) ([]int, error) {
	info, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	return info.Shape, nil
}

// ReadF32 reads a tensor as float32 data.
// Returns the flat float32 slice and the tensor shape.
func (sf *File) ReadF32(name string) ([]float32, []int, error) {
	info, ok := sf.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	if info.Dtype != "F32" {
		return nil, nil, fmt.Errorf("tensor %q has dtype %q, expected F32", name, info.Dtype)
	}

	// Load data if not yet loaded
	if sf.data == nil {
		if err := sf.loadData(); err != nil {
			return nil, nil, err
		}
	}

	start := info.DataOffsets[0]
	end := info.DataOffsets[1]
	raw := sf.data[start:end]

	// Convert raw bytes to float32
	nFloats := len(raw) / 4
	result := make([]float32, nFloats)
	for i := 0; i < nFloats; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		result[i] = math.Float32frombits(bits)
	}

	return result, info.Shape, nil
}

// ReadF32Matrix reads a 2D tensor as float32 data.
// Returns row-major float32 slice, rows, cols.
func (sf *File) ReadF32Matrix(name string) (data []float32, rows int, cols int, err error) {
	flat, shape, err := sf.ReadF32(name)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(shape) != 2 {
		return nil, 0, 0, fmt.Errorf("tensor %q has %d dimensions, expected 2", name, len(shape))
	}
	return flat, shape[0], shape[1], nil
}

// loadData reads the entire data buffer into memory.
func (sf *File) loadData() error {
	f, err := os.Open(sf.filePath)
	if err != nil {
		return fmt.Errorf("open for data read: %w", err)
	}
	defer f.Close()

	// Seek to data start
	if _, err := f.Seek(sf.offset, 0); err != nil {
		return fmt.Errorf("seek to data: %w", err)
	}

	// Read all remaining data
	stat, err := f.Stat()
	if err != nil {
		return fmt.Errorf("stat file: %w", err)
	}
	dataSize := stat.Size() - sf.offset
	sf.data = make([]byte, dataSize)
	if _, err := f.Read(sf.data); err != nil {
		return fmt.Errorf("read data: %w", err)
	}

	return nil
}
