import sys, os, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.transform import resize

import imglogio as ilio
from utils import fill_nan, standardize, breakout_mask_to_properties
from model import DeepLabV3PlusResNet18
from metrics import evaluate_long_log


# -------------------------
# Config
# -------------------------
@dataclass
class InferConfig:
    input_path: List[str] = None  # list of CSVs, len == input_ch
    label_path: str = None  # Ground truth path for computing metrics.
    ckpt_path: str = None
    out_dir: str = None
    input_ch: int = 1
    data_shape: tuple = (256, 256)  # (h, w) the model's input size
    batch_size: int = 16
    thr: List[float] = None
    dz: float = 0.2
    opmin: float = 10.0
    device: str = "cuda:0"
    # overlap settings (fraction of patch height)
    overlap_ratio: float = 0.2  # e.g. 0.2 = 20% overlap on Z


# -------------------------
# Model loader (new model signature)
# -------------------------
def load_model(cfg: InferConfig):
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=True)
    cfgModel = ckpt["model_config"]
    model = DeepLabV3PlusResNet18(
        in_ch=cfgModel["input_ch"],
        num_classes=cfgModel["num_classes"], 
        aspp_out=cfgModel["aspp_out"], 
        rates=cfgModel["dilation_rates"]
    )
    model.load_state_dict(ckpt["model"])
    return model


# -------------------------
# Helper: make start indices with overlap on Z
# -------------------------
def make_starts(H: int, h: int, overlap_ratio: float) -> List[int]:
    """Return a list of start indices so patches [s, s+h) cover [0, H) with overlap.
    Ensures the last patch ends at H (i.e., last start = max(H-h, 0))."""
    if H <= h:
        return [0]
    overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.9))
    stride = max(1, int(round(h * (1.0 - overlap_ratio))))
    starts = list(range(0, H - h + 1, stride))
    if starts[-1] != H - h:
        starts.append(H - h)
    # de-duplicate just in case rounding produced duplicates
    starts = sorted(set(starts))
    return starts


# -------------------------
# Inference with Z-overlap and averaged stitching
# -------------------------
@torch.no_grad()
def infer(cfg: InferConfig):
    # Check
    if len(cfg.input_path) != cfg.input_ch:
        raise ValueError(
            f"The number of input file paths is {len(cfg.input_path)} but the number of input channels is {cfg.input_ch}"
        )

    # Load model
    sys.stdout.write(f"\rLoading model checkpoint from {cfg.ckpt_path}...")
    model = load_model(cfg).to(cfg.device)
    model.eval()
    sys.stdout.write(" Done\n")

    # Load logs (full-length per-channel)
    H_target, W_target = cfg.data_shape
    channels = []  # list of (H_full, W_full) arrays
    for i in range(cfg.input_ch):
        sys.stdout.write(f"\rLoading data from {cfg.input_path[i]} [{i+1}/{cfg.input_ch}]...")
        log = np.load(cfg.input_path[i], allow_pickle=True)
        val = log["data"]
        sys.stdout.write(" Done\n")
        channels.append(val.astype(np.float32))

    H_full, W_full = channels[0].shape

    # Accumulators at model width (W_target). We'll resize back to W_full at the end.
    prob_accum = np.zeros((H_full, W_target), dtype=np.float64)
    weight_accum = np.zeros((H_full, W_target), dtype=np.float64)

    starts = make_starts(H_full, H_target, cfg.overlap_ratio)

    # Batched loop over starts
    t0 = time.perf_counter()
    pbar = tqdm(range(0, len(starts), cfg.batch_size), desc="Inference (with %.1f%% overlap)" % (cfg.overlap_ratio*100))
    for bi in pbar:
        batch_starts = starts[bi : bi + cfg.batch_size]
        B = len(batch_starts)
        xb = np.zeros((B, cfg.input_ch, H_target, W_target), dtype=np.float32)

        # Assemble batch
        for j, s in enumerate(batch_starts):
            s_end = s + H_target
            for c, img in enumerate(channels):
                patch = img[s:s_end, :]  # (h, W_full)
                patch = patch.copy()
                # negatives -> NaN, then fill
                patch[patch < 0] = np.nan
                if np.isnan(patch).any():
                    patch = fill_nan(patch)
                # resize width if needed (height already H_target by construction)
                if patch.shape[1] != W_target:
                    patch = resize(patch, (patch.shape[0], W_target), order=1, mode="reflect", anti_aliasing=True)
                # standardize per-patch per-channel (same as previous inference style)
                patch = standardize(patch)
                xb[j, c] = patch

        # To tensor
        xb_t = torch.from_numpy(xb).float().to(cfg.device)

        # Predict -> probs
        yb = model(xb_t)
        probs = torch.sigmoid(yb).detach().cpu().numpy()  # (B,1,H_target,W_target)

        # Accumulate with averaging over overlaps
        for j, s in enumerate(batch_starts):
            s_end = s + H_target
            prob_patch = probs[j, 0]
            prob_accum[s:s_end, :] += prob_patch
            weight_accum[s:s_end, :] += 1.0

    # Normalize accumulated probabilities
    weight_accum[weight_accum == 0] = 1.0
    prob_avg = (prob_accum / weight_accum).astype(np.float32)  # (H_full, W_target)

    # Resize to original width if needed
    if W_target != W_full:
        prob_final = resize(prob_avg, (H_full, W_full), order=1, mode="reflect", anti_aliasing=True).astype(np.float32)
    else:
        prob_final = prob_avg

    t1 = time.perf_counter()

    # Save outputs
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Probability
    sys.stdout.write("\rSaving breakout probability...")
    prob_path = os.path.join(cfg.out_dir, "probability.npz")
    np.savez_compressed(
        prob_path, 
        data=prob_final, 
        depth=log["depth"], 
        azimuth=log["azimuth"], 
        data_unit=None, 
        depth_unit=log["depth_unit"]
    )
    sys.stdout.write(" Done\n")

    # Mask
    for t in cfg.thr:
        sys.stdout.write(f"\rSaving breakout picking results @ threshold {t}...")
        mask = (prob_final > t).astype(np.uint8)
        mask_path = os.path.join(cfg.out_dir, f"mask@{t}.npz")
        np.savez_compressed(
            mask_path, 
            data=mask, 
            depth=log["depth"], 
            azimuth=log["azimuth"], 
            data_unit=None, 
            depth_unit=log["depth_unit"]
        )

        # Properties
        prop_path = os.path.join(cfg.out_dir, f"properties@thr{t}dz{cfg.dz}.csv")
        df = breakout_mask_to_properties(z=log["depth"], mask=mask, azimuth=log["azimuth"], dz=cfg.dz, opmin=cfg.opmin)
        df.to_csv(prop_path, index=False)

        # Compute segmentation metrics.
        if cfg.label_path is not None:
            gt = np.load(cfg.label_path, allow_pickle=True)["data"]
            gt = gt.astype(np.uint8) if gt.dtype != np.uint8 else gt
            m = evaluate_long_log(mask, gt, block=(256, mask.shape[1]), pred_is_prob=False)
            with open(os.path.join(cfg.out_dir, f"metrics@thr{t}.txt"), "w") as f:
                f.write(f"Model file path: {cfg.ckpt_path}\n")
                f.write(f"Predict mask file path: {mask_path}\n")
                f.write(f"Ground truth file path: {cfg.label_path}\n")
                f.write(f"Inference time: %d seconds\n" % (t1 - t0))
                for k,v in m.items():
                    f.write(f"{k}: {v}\n")
            f.close()
        
        sys.stdout.write(" Done\n")

    print("All done. Inferrence time: %d seconds" % (t1 - t0))
    print(f"Inference results have been saved to {cfg.out_dir}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    cfg = InferConfig(
        input_path=[
            "./dataset/BedrettoLab-CB1/BedrettoLab-CB1-ATV-Amplitude-20191010_up3_NM.npz",
            "./dataset/BedrettoLab-CB1/BedrettoLab-CB1-ATV-BoreholeRadius-20191010_up3_NM.npz"
        ],
        label_path = "./dataset/BedrettoLab-CB1/BedrettoLab-CB1-ATV-BreakoutMask-20191010_up3_NM.npz", 
        ckpt_path="./checkpoint/last.pt",
        out_dir="./my_output_directory",
        input_ch=2,
        data_shape=(256, 256),
        batch_size=64,
        thr=[0.5, 0.6, 0.7, 0.8, 0.9],
        dz=0.2,
        opmin=10.0,
        device="cuda:0",
        overlap_ratio=0.2,
    )
    infer(cfg)
