"""Drug-drug and target-target similarity matrices (K1 and K2 in the paper)."""

import os
from math import sqrt

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")


def tanimoto_matrix(smiles):
    """Tanimoto similarity between RDKit topological fingerprints."""
    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps],
                    dtype=np.float32)


def smith_waterman_matrix(sequences):
    """Normalised Smith-Waterman score, Eq. (3): SW_ij / sqrt(SW_ii * SW_jj)."""
    aligner = Align.PairwiseAligner(
        mode="local",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-11,
        extend_gap_score=-1,
    )
    self_scores = [aligner.score(s, s) for s in sequences]

    n = len(sequences)
    K = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            score = aligner.score(sequences[i], sequences[j])
            K[i, j] = K[j, i] = score / sqrt(self_scores[i] * self_scores[j])
    return K


def cached(path, compute, *args):
    """Load an array from `path`, computing and saving it on the first call."""
    if os.path.exists(path):
        return np.load(path)
    print(f"Computing {path} ...")
    result = compute(*args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, result)
    return result


def similarity_matrices(dataset, cache_dir="data/similarity"):
    """Return (K1, K2) for a DTIDataset, cached on disk."""
    tag = f"{dataset.name}_{len(dataset.drugs)}d_{len(dataset.targets)}t"
    K1 = cached(f"{cache_dir}/{tag}_tanimoto.npy", tanimoto_matrix, dataset.drugs)
    K2 = cached(f"{cache_dir}/{tag}_smith_waterman.npy",
                smith_waterman_matrix, dataset.targets)
    return K1, K2
