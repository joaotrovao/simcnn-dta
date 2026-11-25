"""Loading of the Davis and KIBA datasets from Therapeutics Data Commons (TDC)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tdc.multi_pred import DTI

DATA_DIR = "data"

# Affinity cut-offs used to binarise the data for AUPR (Shim et al., 2021)
THRESHOLDS = {"davis": 7.0, "kiba": 12.1}


@dataclass
class DTIDataset:
    name: str
    drugs: list          # SMILES string of each unique drug
    targets: list        # amino-acid sequence of each unique target
    drug_idx: np.ndarray    # for each interaction, index into `drugs`
    target_idx: np.ndarray  # for each interaction, index into `targets`
    y: np.ndarray           # binding affinity of each interaction
    threshold: float

    def __len__(self):
        return len(self.y)


def load_dataset(name, max_targets=None):
    """Load a dataset as integer drug/target indices plus affinities.

    `max_targets` keeps only the first N targets (and the drugs that interact
    with them), for quick experiments on a laptop.
    """
    df = DTI(name=name.upper(), path=DATA_DIR).get_data()

    if name == "davis":
        # Kd (nM) -> pKd, Eq. (1) of the paper
        df["Y"] = -np.log10(df["Y"] / 1e9)
    # TDC's KIBA scores are already transformed as described in the paper
    # (higher = stronger binding, threshold 3.0 -> 12.1)

    if max_targets is not None:
        kept = pd.unique(df["Target_ID"])[:max_targets]
        df = df[df["Target_ID"].isin(kept)].reset_index(drop=True)

    drug_idx, drug_ids = pd.factorize(df["Drug_ID"])
    target_idx, target_ids = pd.factorize(df["Target_ID"])
    smiles = df.drop_duplicates("Drug_ID").set_index("Drug_ID")["Drug"]
    sequences = df.drop_duplicates("Target_ID").set_index("Target_ID")["Target"]

    return DTIDataset(
        name=name,
        drugs=smiles.loc[drug_ids].tolist(),
        targets=sequences.loc[target_ids].tolist(),
        drug_idx=drug_idx.astype(np.int32),
        target_idx=target_idx.astype(np.int32),
        y=df["Y"].to_numpy(np.float32),
        threshold=THRESHOLDS[name],
    )
