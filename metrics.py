"""
Blockwise, memory-friendly metrics for very long imaging logs.

What it does
------------
* Tiling-only pass (e.g., 256x256 blocks) over very long 2D masks.
* Accumulates global TP/FP/FN/TN (micro-averaging) without storing all tiles.
* Returns IoU, Dice, Precision, Recall, Specificity computed from global counts.

Inputs
------
- pred: 2D array (H, W). Either binary {0,1} mask or probability [0,1].
- gt:   2D array (H, W). Binary {0,1} ground-truth mask.
- If `pred_is_prob=True`, a threshold `thr` is applied to binarize.
- Optional `ignore_index`: mask out values in GT equal to this id.

Why micro-averaging
-------------------
* Handles large proportion of empty tiles naturally (no per-tile Dice issues).
* Reflects the full-well behavior: all pixels are counted once globally.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Dict


@dataclass
class Confusion:
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0

    def update(self, other: "Confusion") -> None:
        self.TP += other.TP
        self.FP += other.FP
        self.FN += other.FN
        self.TN += other.TN

    def to_dict(self) -> Dict[str, int]:
        return {"TP": self.TP, "FP": self.FP, "FN": self.FN, "TN": self.TN}


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d > 0 else float("nan")


def confusion_from_arrays(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pred_is_prob: bool = False,
    thr: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Confusion:
    """Compute confusion counts on equally-shaped 2D arrays.
    - pred: mask/prob, shape (H, W)
    - gt:   binary mask, shape (H, W)
    """
    if pred.dtype != np.bool_ and pred.dtype != np.uint8 and pred_is_prob is False:
        # if non-bool/non-uint8 but user says it's already binary, coerce
        pred_bin = pred.astype(np.uint8)
    elif pred_is_prob:
        pred_bin = (pred >= thr).astype(np.uint8)
    else:
        pred_bin = pred.astype(np.uint8)

    gt_bin = (gt.astype(np.uint8) > 0).astype(np.uint8)

    if ignore_index is not None:
        valid = (gt != ignore_index)
        if valid.ndim == 2:
            # mask invalid positions by zeroing both pred & gt and counting TN there
            p = pred_bin[valid]
            g = gt_bin[valid]
            # counts in valid area
            TP = int(np.sum((p == 1) & (g == 1)))
            FP = int(np.sum((p == 1) & (g == 0)))
            FN = int(np.sum((p == 0) & (g == 1)))
            TN = int(np.sum((p == 0) & (g == 0)))
            return Confusion(TP, FP, FN, TN)

    # no ignore mask or none left
    TP = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    FP = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    FN = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    TN = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
    return Confusion(TP, FP, FN, TN)


def confusion_blockwise(
    pred: np.ndarray,
    gt: np.ndarray,
    block: Tuple[int, int] = (256, 256),
    *,
    pred_is_prob: bool = False,
    thr: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Confusion:
    """Accumulate confusion counts over (H, W) by tiling without overlap."""
    assert pred.shape == gt.shape, "pred/gt shapes must match"
    H, W = gt.shape
    bh, bw = block

    total = Confusion()
    for y in range(0, H, bh):
        for x in range(0, W, bw):
            sl = np.s_[y:min(y+bh, H), x:min(x+bw, W)]
            c = confusion_from_arrays(
                pred[sl], gt[sl], pred_is_prob=pred_is_prob, thr=thr, ignore_index=ignore_index
            )
            total.update(c)
    return total


def metrics_from_confusion(c: Confusion) -> dict:
    TP, FP, FN, TN = c.TP, c.FP, c.FN, c.TN
    iou = _safe_div(TP, TP + FP + FN)
    dice = _safe_div(2 * TP, 2 * TP + FP + FN)
    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    specificity = _safe_div(TN, TN + FP)
    accuracy = _safe_div(TP + TN, TP + TN + FP + FN)
    fpr = _safe_div(FP, FP + TN)
    fnr = _safe_div(FN, FN + TP)
    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "FPR": fpr,
        "FNR": fnr,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }


def evaluate_long_log(
    pred: np.ndarray,
    gt: np.ndarray,
    block: Tuple[int, int] = (256, 256),
    *,
    pred_is_prob: bool = False,
    thr: float = 0.5,
    ignore_index: Optional[int] = None,
) -> dict:
    """Convenience wrapper: blockwise confusion -> global micro metrics.
    Returns a dict with counts and metrics.
    """
    c = confusion_blockwise(
        pred, gt, block=block, pred_is_prob=pred_is_prob, thr=thr, ignore_index=ignore_index
    )
    m = metrics_from_confusion(c)
    return m
