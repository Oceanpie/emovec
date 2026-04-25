package main

import "embed"

//go:embed data/model.safetensors data/prototype_labels.json
var embeddedData embed.FS
