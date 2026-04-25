# Building a Prototype Library for Emotion Analysis

This guide explains the methodology for building an emotional prototype library from scratch — the core data that powers the zero-shot emotion matching pipeline.

> **Note**: This repository contains the algorithm and service code only. Follow this guide to construct your own prototype library. No pre-built data is included.

## Overview

The system maps Chinese text to 8D emotion vectors using **prototype-based nearest-neighbor matching**:

1. Define a set of emotional prototypes (e.g., from a taxonomy of human emotions)
2. Label each prototype with an 8D emotion intensity vector via LLM
3. Encode each prototype's description into an embedding vector
4. At query time, encode the input text and find the nearest prototypes by cosine similarity
5. Weighted aggregate the prototypes' 8D labels → final output

## Step-by-Step Process

### Step 1: Source Emotional Concepts

Select a source of well-defined emotional concepts. Each concept should have a clear description in natural language.

**Recommended sources:**

| Source | Description |
|--------|-------------|
| *The Book of Human Emotions* by Tiffany Watt Smith | ~150 culturally diverse emotion entries, each with detailed descriptions |
| Plutchik's Wheel of Emotions | 8 primary + 24 secondary emotions with established theoretical framework |
| Academic emotion taxonomies | e.g., WordNet-Affect, EmoLex, NRC Emotion Lexicon |

**Entry format:**

Each emotional concept should be a standalone document with:
- A unique title (e.g., "Saudade", "Anger", "Amae")
- A natural language description explaining the emotional experience
- Optional: cross-references to related emotions

### Step 2: Label Prototypes with 8D Emotion Scores

Each prototype entry needs a soft label: an 8-dimensional vector where each dimension represents the intensity of a specific emotion.

**Label schemes** (choose one):

**Plutchik 8D** (psychologically grounded, with polarity constraints):

| # | Dimension | Opposite |
|---|-----------|----------|
| 1 | joy | ↔ sadness |
| 2 | anger | ↔ fear |
| 3 | sadness | ↔ joy |
| 4 | fear | ↔ anger |
| 5 | disgust | ↔ trust |
| 6 | surprise | ↔ anticipation |
| 7 | trust | ↔ disgust |
| 8 | anticipation | ↔ surprise |

**Original 8D** (includes East Asian culturally relevant dimensions):

| # | Dimension |
|---|-----------|
| 1 | 高兴 (happiness) |
| 2 | 愤怒 (anger) |
| 3 | 悲伤 (sadness) |
| 4 | 害怕 (fear) |
| 5 | 厌恶 (disgust) |
| 6 | 忧郁 (melancholy) |
| 7 | 惊讶 (surprise) |
| 8 | 平静 (calmness) |

**Labeling strategy — LLM-as-Labeler:**

Use a large language model (e.g., DeepSeek Chat, GPT-4) to generate the 8D labels:

```
You are an expert in cross-cultural emotion theory. Rate the following
emotion entry on each of 8 dimensions (0.00–1.00).

Dimension definitions:
- joy: positive愉悦, smiling, sharing, wanting to approach
- anger: hostile reaction to obstacles/frustration
- sadness: low mood, loss, withdrawal
- fear: anticipation of threat, fight/flight
- disgust: aversion, revulsion
- surprise: startle, novelty reaction
- trust: security, reliance, openness
- anticipation: forward-looking, expectation

Rules:
1. Rate each dimension 0.00–1.00, precise to 2 decimal places
2. Pure basic emotions → ~0.95 on that dimension
3. Mixed/cultural emotions → multiple dimensions non-zero
4. Most dimensions should be 0 for most entries
5. Output ONLY valid JSON

Schema: {"joy": float, "anger": float, "sadness": float, "fear": float,
         "disgust": float, "surprise": float, "trust": float, "anticipation": float}
```

**Quality measures:**

| Measure | Implementation |
|---------|---------------|
| Multi-round averaging | Run N independent labeling rounds (e.g., 5), take mean per dimension |
| Variance tracking | Flag entries with high variance (stddev > 0.15) for manual review |
| Redirect detection | Automatically skip entries marked as redirects (*see: XXX*) |
| Idempotency | Support incremental labeling — skip already-processed entries |
| Periodic saving | Save intermediate results (e.g., every 5 entries) to prevent data loss |

### Step 3: Encode Prototypes into Embedding Vectors

Use an embedding model to convert each prototype's description into a vector.

**Recommended embedding models:**

| Model | Dimensions | Notes |
|-------|-----------|-------|
| Qwen3-Embedding-8B | 4096 (MRL supports truncation) | Strong Chinese performance, C-MTEB 76.97 |
| Qwen3-Embedding-0.6B | 1024 | Lighter, evaluation shows parity with 8B for this task |
| text-embedding-3-large | 3072 | OpenAI, good general performance |

**Instruction strategy:**

Apply a retrieval-style instruction to the **query side only** (not to the prototype entries):

```
Instruction: "Given a Chinese emotional utterance, retrieve the most
relevant emotion description that matches the emotional state expressed"

Format: Instruct: {instruction}\nQuery:{query}
```

- **Entry side**: use the raw description text directly, no instruction prefix
- **Query side**: prepend the instruction prefix before encoding
- The instruction guides the embedding model to focus on emotional content

**Encoding process:**

```
For each prototype entry:
  1. Read the natural language description
  2. Call embedding API with the description (no instruction prefix)
  3. Store the returned vector

Configuration:
  - Provider fallback: primary → backup (e.g., ModelScope → SiliconFlow)
  - Concurrency: 5 parallel requests
  - Batch: process all entries, store as (N_entries × D) float32 array
```

### Step 4: Build and Validate the Prototype Store

**Store format:**

| Item | Format | Shape |
|------|--------|-------|
| Vectors | numpy .npy | (N_entries, D_dimensions), float32 |
| Labels | JSON | N_entries × 8D scores |
| Metadata | JSON | Model info, entry mapping, dimensions |

**Validation:**

- Verify all entries have matching vectors and labels
- Check for NaN/Inf in vectors
- Validate label values are in [0, 1] range
- Confirm embedding dimensions match expectations

### Step 5: Implement the Matching Pipeline

The core algorithm (implemented in `src/matching.py` and `emovec/internal/matching/`):

```
Input: query text
  1. Encode: embedding(text, instruction, dim=D) → query_raw
  2. Normalize: query_vec = query_raw / ||query_raw||
               proto_norm = proto_vectors / ||proto_vectors||
  3. Cosine Sim: sims = proto_norm @ query_vec        # dot product
  4. Top-K: top_indices = argsort(sims)[::-1][:K]
  5. Softmax: weights = softmax(sims[top_indices], tau)
  6. Weighted Sum: output[d] = sum(weights[i] × labels[i][d])
Output: 8D emotion intensity vector
```

**Key parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| dim | 512–1024 | Embedding dimension (truncate if MRL-supported) |
| K | 7 | Number of nearest prototypes |
| τ | 0.5 | Softmax temperature (lower = sharper) |

### Step 6: Package for the Go Service (Optional)

The Go service (`emovec/`) expects prototype data in safetensors format:

| Tensor Name | Shape | Content |
|-------------|-------|---------|
| `layers.0.weight` | N×D | Prototype vectors |
| `layers.1.weight` | 8×N | Prototype 8D labels (transposed) |

For data obfuscation (optional), the service supports:
- **Matrix padding**: Pad the N×D matrix to D×D using real embedding vectors from unrelated text
- **Row permutation**: Shuffle rows using a seed-derived permutation
- **Multiplicative transform**: Apply element-wise transform (K×B, where B is derived from a seed)
- **Obfuscation**: Package data to resemble a 2-layer MLP weight file

See [docs/部署设计.md](部署设计.md) for detailed obfuscation design.

## Complete Reference Pipeline

```bash
# 1. Parse source material into individual entries
#    (Extract emotional concepts from the book/taxonomy)

# 2. Generate 8D labels via LLM
#    (Label each entry with Plutchik or Original scheme, 5 rounds averaging)

# 3. Encode entries into embedding vectors
python scripts/build_prototype.py

# 4. Verify the prototype store
#    (Check dimensions, validate labels, test a few queries)

# 5. Run queries
python scripts/match.py "你的输入文本"
```

> The scripts referenced above are data-processing tools not included in this repository. Implement equivalent logic based on this guide's methodology.

## Notes

- The prototype library is **model-version-specific**: upgrading the embedding model requires re-encoding all prototypes
- Labels and vectors are independent: you can update labels without re-encoding
- 153–200 prototypes is a good range: enough for coverage, small enough for sub-millisecond matching
- The matching step is computationally negligible (< 1ms) — the bottleneck is always the embedding API call
