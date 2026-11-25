"""Evaluation metrics used in the paper: MSE, CI, r²m and AUPR."""

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score


def mse(y, y_pred):
    return float(np.mean((y - y_pred) ** 2))


def ci(y, y_pred):
    """Concordance index, Eq. (5)."""
    return float(concordance_index(y, y_pred))


def rm2(y, y_pred):
    """Modified squared correlation coefficient r²m, Eq. (7) (Roy & Roy, 2008)."""
    r2 = np.corrcoef(y, y_pred)[0, 1] ** 2
    k = np.sum(y * y_pred) / np.sum(y_pred ** 2)  # slope through the origin
    r02 = 1 - np.sum((y - k * y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2 * (1 - np.sqrt(abs(r2 - r02))))


def aupr(y, y_pred, threshold):
    """Area under precision-recall, with true labels binarised at `threshold`.

    The predictions are kept continuous: they act as the ranking scores.
    """
    return float(average_precision_score(y >= threshold, y_pred))


def evaluate(y, y_pred, threshold):
    return {
        "MSE": mse(y, y_pred),
        "CI": ci(y, y_pred),
        "rm2": rm2(y, y_pred),
        "AUPR": aupr(y, y_pred, threshold),
    }
