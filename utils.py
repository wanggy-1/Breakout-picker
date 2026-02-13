import glob, os, torch, math, sys
import numpy as np
import pandas as pd
import imglogio as ilio
from torch.utils.data import Dataset
from scipy.interpolate import NearestNDInterpolator


def ensure_ch_first(arr):
    if arr.ndim == 2:
        return arr[None, ...]
    elif arr.ndim == 3:
        return arr
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")
    

def standardize_per_channel(x):
    """
    x: [C, H, W]
    """
    c_mean = x.mean(axis=(1, 2), keepdims=True)
    c_std = x.std(axis=(1, 2), keepdims=True)
    return (x - c_mean) / (c_std + 1e-6)


def standardize(x):
    """
    x: [H, W]
    """
    return (x - x.mean()) / (x.std() + 1e-6)


def fill_nan(arr):
    """
    Fill NaN values in a 2D numpy array using the nearest neighbor interpolation.
    """
    # A boolean array indicating normal values (True) and NaN values (False).
    mask = ~np.isnan(arr)
    # Coordinates of the array.
    xx, yy = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    # Coordinates of normal values.
    xy = np.c_[xx[mask].ravel(), yy[mask].ravel()]
    # Interpolation.
    interp = NearestNDInterpolator(xy, arr[mask].ravel())
    arr_itp = interp(xx.ravel(), yy.ravel()).reshape(arr.shape)
    
    return arr_itp


def wrap_angle(delta):
    # wrap to (-pi, pi]
    return (delta + math.pi) % (2 * math.pi) - math.pi


def get_theta_radians(theta_size: int, device=None):
    # 0..2pi (excluded at end), length = theta_size
    return torch.linspace(0, 2 * math.pi, steps=theta_size + 1, device=device)[:-1]


class ATVDataset(Dataset):
    def __init__(
        self, 
        input_dir: str|list = None, 
        label_dir: str = None, 
        sample_shape: tuple = (256, 256), 
        n_channel: int = 1, 
        return_filename: bool = False
    ):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.sample_shape = sample_shape
        self.n_channel = n_channel
        self.return_filename = return_filename
        if isinstance(self.input_dir, str):
            self.input_dir = [self.input_dir]
        if self.n_channel != len(self.input_dir):
            raise ValueError("Input channel number is %d, but input dir number is %d" % (self.n_channel, len(self.input_dir)))
        
        input_list = sorted(glob.glob(os.path.join(self.input_dir[0], "*.npz")))
        self.items = []
        for x_path in input_list:
            filename = os.path.basename(x_path)
            y_path = os.path.join(self.label_dir, filename)
            if os.path.exists(y_path):
                self.items.append(filename)
        if len(self.items) == 0:
            raise RuntimeError("No matched (input, label) pairs found.")
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename = self.items[idx]
        
        # [H, W]->[C, H, W]
        x = np.zeros((self.n_channel, *self.sample_shape), dtype=np.float32)  # Input.

        # Load input.
        for ich, xdir in enumerate(self.input_dir):
            xs = np.load(os.path.join(xdir, filename), allow_pickle=True)
            xch = xs["data"]
            xch = xch.astype(np.float32)
            # Convert negative values to NaN.
            xch[xch < 0] = np.nan
            # Interpolate NaN.
            if np.isnan(xch).any():
                xch = fill_nan(xch)
            # Dump to the input array.
            x[ich, :, :] = xch.copy()

        # Load label.
        ys = np.load(os.path.join(self.label_dir, filename), allow_pickle=True)
        y = ys["data"]
        y = y.astype(np.float32)
        # Check NaN values.
        if np.isnan(y).any():
            y = fill_nan(y)
        # [H, W]->[C, H, W]
        y = y[None, ...]

        # Standardize x.
        x = standardize_per_channel(x)

        # To tensor.
        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).float()

        if self.return_filename:
            filename_nosuffix = os.path.splitext(filename)[0]
            meta = {
                "depth": ys["depth"], 
                "azimuth": ys["azimuth"], 
            }
            return x, y, filename_nosuffix, meta
        else:
            return x, y


@torch.no_grad()
def dice_score(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(pred)
    preds = (probs > thr).float()
    inter = (preds * gt).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = ((2 * inter + eps) / (denom + eps)).mean().item()
    return float(dice)


@torch.no_grad()
def iou_score(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(pred)
    preds = (probs > thr).float()
    inter = (preds * gt).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) - inter
    iou = ((inter + eps) / (union + eps)).mean().item()
    return float(iou)


@torch.no_grad()
def specificity_score(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    thr: float = 0.5, 
    fp_penalty: float = 1.0) -> float:
    """
    Pixel-wise specificity (true negative rate)
    
    Args:
        pred: raw logits, shape [B, 1, H, W]
        gt: ground truth mask, shape [B, 1, H, W]
        thr: threshold for binarization after sigmoid
        fp_penalty: fp_penalty > 1 will penalize false positives more heavily.
    """
    eps = 1e-6
    probs = torch.sigmoid(pred)
    preds = (probs > thr).float()
    tn = ((preds == 0) & (gt == 0)).sum(dim=(1, 2, 3)).float()
    fp = ((preds == 1) & (gt == 0)).sum(dim=(1, 2, 3)).float()
    spec = ((tn + eps) / (tn + fp_penalty * fp + eps)).mean().item()

    return float(spec)


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def breakout_mask_to_properties(z: np.ndarray, mask: np.ndarray, 
                                azimuth: np.ndarray, dz: float, 
                                opmin: float = 10.0):
    # Resample breakout mask.
    z, mask = ilio.resample(z, mask, dz, method='nearest', verbose=False)
    # Azimuthal sampling rate.
    dazi = 360 / mask.shape[1]
    # Extract breakouts.
    idx = np.argwhere(mask == 1)  # Array index of breakouts.
    idr = np.unique(idx[:, 0])  # Row index.
    az = []  # Breakout central azimuth.
    op = []  # Breakout opening.
    zbo = []  # Breakout measured depth.
    for i in range(len(idr)):
        idc = idx[idx[:, 0] == idr[i]][:, 1]  # Column index.
        s = ilio.split_circular_consecutive_indices(idc, n_cols=mask.shape[1])  # Split breakouts.
        for j in range(len(s)):
            pl = s[j][0]  # Left end array index of the breakout.
            pm = np.floor(np.median(s[j]))  # Central array index of the breakout.
            pr = s[j][-1]  # Right end array index of the breakout.
            if azimuth[pr] >= azimuth[pl]:
                width = azimuth[pr] - azimuth[pl]
                azi = (azimuth[pl] + width / 2) // dazi * dazi
            else:
                width = azimuth[pr] + 360 - azimuth[pl]
                azi = azimuth[pl] + width / 2
                if azi <= 360:
                    azi = azi // dazi * dazi
                else:
                    azi = (azi - 360) // dazi * dazi
            zbo.append(z[idr[i]])  # Measured depth.
            op.append(width)  # Opening.
            az.append(azi)  # Azimuth.

    # Post process.
    dfu = pd.DataFrame(columns=['Depth', 'Azimuth', 'Tilt', 'Length', 'Opening'], 
                    data=[['m', 'deg', 'deg', 'm', 'deg']])
    dfv = pd.DataFrame(columns=['Depth', 'Azimuth', 'Tilt', 'Length', 'Opening'])
    dfv['Depth'] = zbo
    dfv['Azimuth'] = az
    dfv['Tilt'] = np.ones(len(zbo)) * -999
    dfv['Length'] = np.ones(len(zbo)) * dz
    dfv['Opening'] = op
    idx = [x for x in range(len(dfv)) if dfv.loc[x, 'Opening'] < opmin]
    dfv.drop(index=idx, inplace=True)
    dfv.reset_index(drop=True, inplace=True)
    df = pd.concat([dfu, dfv], ignore_index=True)
    return df


def plot_history(history, out_dir):
    import matplotlib.pyplot as plt

    # Plot losses, metrics, and learning rate.
    # Losses.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.plot(history["epoch"], history["train_loss_sum"], label="train", lw=1.5)
    ax.plot(history["epoch"], history["val_loss_sum"], label="valid", lw=1.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses.png"), dpi=300)

    # Metrics.
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].set_title("Specificity score")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Value")
    ax[0].plot(history["epoch"], history["train_spec_score"], label="train", lw=1.5)
    ax[0].plot(history["epoch"], history["val_spec_score"], label="valid", lw=1.5)
    ax[0].legend()

    ax[1].set_title("IoU score")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Value")
    ax[1].plot(history["epoch"], history["train_iou_score"], label="train", lw=1.5)
    ax[1].plot(history["epoch"], history["val_iou_score"], label="valid", lw=1.5)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics.png"), dpi=300)

    # Learning rate.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.plot(history["epoch"], history["learning_rate"], c='k', lw=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr.png"), dpi=300)


def plot_sample(data, cmap, 
                vlim: tuple = (None, None), 
                extent: list = None, 
                depth_unit: str = "m",  
                data_unit: str = None, 
                out_path: str = None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vmin, vmax = vlim

    plt.figure()
    im = plt.imshow(data, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    plt.xticks([0, 90, 180, 270, 360])
    plt.xlabel('Azimuth (°)')
    plt.ylabel(f'Depth ({depth_unit})')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(f'Value ({data_unit})')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_pos_weight(train_loader, device="cpu"):
    """
    Compute pos_weight ONLY from positive samples (mask.sum() > 0).

    pos_weight = (#negative pixels in positive samples) / (#positive pixels)
    """
    pos_pixels = 0.0
    neg_pixels = 0.0
    n_pos_imgs = 0

    for xb, yb in train_loader:
        yb = yb.to(device)
        yb = (yb > 0.5).float()   # ensure binary

        # per-sample foreground area
        fg = yb.sum(dim=(1, 2, 3))  # [B]
        pos_mask = fg > 0

        if pos_mask.any():
            y_pos = yb[pos_mask]           # only positive images
            pos_pixels += y_pos.sum().item()
            neg_pixels += (y_pos.numel() - y_pos.sum().item())
            n_pos_imgs += int(pos_mask.sum().item())

    if pos_pixels < 1:
        raise ValueError("No positive pixels found in positive samples!")

    pos_weight = neg_pixels / pos_pixels
    return torch.tensor([pos_weight], dtype=torch.float32, device=device)

