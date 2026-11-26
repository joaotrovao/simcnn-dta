# SimCNN-DTA (re-implementation)

A re-implementation of SimCNN-DTA, a convolutional model that predicts how
strongly a drug binds to a protein target, from
[Shim et al., *Scientific Reports* 11, 3628 (2021)](https://doi.org/10.1038/s41598-021-83679-y).

It began as a university bioinformatics project and I later rewrote it into a
modular, reproducible pipeline: the training script runs k-fold
cross-validation, survives interruption, and reports the same four metrics the
paper uses. Both datasets from the paper are evaluated here, and the section on
differences from the paper records where this implementation departs from it and
why.

## Background
Finding out whether a candidate drug binds a given protein is slow and
expensive to do in a lab, so predicting binding affinity computationally is a
way of narrowing down which pairs are worth testing. Most deep-learning
approaches learn from the raw drug and protein sequences.

SimCNN-DTA takes a different route: instead of the molecules themselves, it
feeds the model *similarities*. Each drug is described by how similar it is to
every other drug in the dataset, and each target by how similar it is to every
other target. The outer product of those two vectors forms a 2D matrix — one
value per (other-drug, other-target) combination — which can be treated as an
image and handed to a CNN. The intuition is that if a drug resembles drugs that
bind this target, and the target resembles targets that this drug binds, the
affinity is probably high; the convolutions pick up those patterns across the
whole similarity landscape.

## Method
1. Tanimoto similarity on Morgan fingerprints (RDKit) for drugs
2. Normalised Smith–Waterman alignment scores (Biopython) for targets
3. Outer product of the two similarity vectors gives a 2D "similarity image"
4. Two conv + max-pool blocks → flatten → Dense(128) → 1 output
5. MSE loss

## Repository layout

| File | Purpose |
|---|---|
| `data.py` | Downloads Davis/KIBA via PyTDC, builds the drug and target lists |
| `similarity.py` | Computes and caches the similarity matrices |
| `model.py` | The CNN (Keras) |
| `metrics.py` | MSE, concordance index, r²m, AUPR |
| `train.py` | k-fold cross-validation, resumable |
| `results/` | Metrics (JSON) and predicted-vs-measured scatter plots |

## Setup

Python 3.11:

```bash
conda create -n simcnn-dta python=3.11
conda activate simcnn-dta
pip install -r requirements.txt
```

## Usage

```bash
python train.py --dataset davis                    # full Davis, 5 folds
python train.py --dataset kiba --max-targets 50    # KIBA on 50 targets
```

Runs are resumable: re-running the same command skips completed folds and
continues the current one from its last finished epoch. `--folds`, `--epochs`
and `--max-targets` can be lowered for a quick trial run.

## Results

5-fold cross-validation, mean ± std, compared with the values reported in the
paper:

| | Davis (this repo) | Davis (paper) | KIBA-50 (this repo) | KIBA (paper) |
|---|---|---|---|---|
| MSE | 0.350 ± 0.021 | 0.306 | 0.321 ± 0.012 | 0.178 |
| CI | 0.828 ± 0.009 | 0.855 | 0.789 ± 0.004 | 0.874 |
| r²m | 0.463 ± 0.024 | 0.595 | 0.528 ± 0.023 | 0.673 |
| AUPR | 0.629 ± 0.027 | 0.657 | 0.709 ± 0.009 | 0.782 |

## Differences from the paper
- TDC's Davis is 68 drugs × 379 targets = 25,772 pairs; the paper reports 442 targets / 30,056 pairs.
- KIBA was run on the **first 50 targets only** (2,068 drugs × 50 targets = 38,370 pairs) with pooling stride 2 — full KIBA is ~117k pairs, roughly 7 days on a laptop. So the KIBA numbers aren't directly comparable to the paper's.
- Smith–Waterman parameters (BLOSUM62, gap −11/−1) are chosen here; the paper doesn't state them.
- r²m follows Roy et al. Eq. 7; the DeepDTA reference code uses a slightly different formula (difference here was 0.002, so the gap to the paper comes from a lower r², not the formula).
- Hardware: Apple M3, 16 GB — Davis full run ≈ 7 h, KIBA-50 ≈ 13 h.

## Credits
Original university project done with Ariana Lopes. The rewrite into this
pipeline was done with AI assistance (Claude Code). Datasets accessed through
[Therapeutics Data Commons](https://tdcommons.ai/).
