# One-Shot Data Selection for Medical Image Classification via Graph Coverage

Selecting the most informative samples to annotate from large medical image datasets, without requiring any model training. We build a neighborhood graph over pretrained image representations and use multi-hop connectivity to measure how well a small subset covers the full dataset. Samples are selected greedily to maximize this coverage, ensuring the chosen subset captures the underlying structure of the data.

## Setup

```bash
pip install torch torchvision medmnist numpy pandas scikit-learn scipy tqdm faiss-gpu
```

## Usage

```bash
# Run selection + evaluation
python -m miccai.run \
    --datasets organsmnist \
    --methods graph_a2 facility fps herding random \
    --embeddings uni \
    --ratios 0.02 0.05 \
    --trials 5 -v \
    --training-paradigm iteration \
    --iterations 1000

# List available methods, datasets, embeddings
python -m miccai.run --list-methods
python -m miccai.run --list-datasets
python -m miccai.run --list-embeddings

# Ablation: compare k values
python -m miccai.run.compare_k \
    --dataset organsmnist \
    --method graph_a2 \
    --embedding uni \
    --k-values 5 10 20 \
    --ratio 0.02 --train --trials 3

# Ablation: compare global vs per-class graph
python -m miccai.run.compare_global \
    --dataset organsmnist \
    --method graph_a2 \
    --embedding uni \
    --ratio 0.02 --train --trials 3
```

## Methods

**One-shot (embedding-based):**
- `graph_a1` ... `graph_a5` — Graph kernel coverage (1-hop to 5-hop)
- `facility` — Greedy facility location on cosine similarity
- `fps` — Farthest point sampling
- `herding` — Iterative mean-matching

**Training-based:**
- `eva` — Error variability across epochs
- `el2n_top` — Error L2-norm scoring
- `forgetting` — Forgetting event counting

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--datasets` | MedMNIST dataset names |
| `--methods` | Selection methods |
| `--embeddings` | `uni`, `imagenet`, `trained`, `random` |
| `--ratios` | Selection ratios (e.g., 0.02 0.05) |
| `--trials` | Number of random seeds |
| `-k` | k-NN neighbors (default: 10) |
| `--k-hops` | Propagation depth (default: 2) |
| `--global` | Global graph construction |

Results are saved to `miccai/results/runs/<run_id>/`.
