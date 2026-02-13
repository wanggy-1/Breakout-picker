"""
DeepLabV3+ training script for breakout segmentation from acoustic image logs.
Use positive and negative samples for model training.
"""
import os, sys, time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from utils import *                      
from model import *                      


# -------------------------
# Training configuration
# -------------------------
@dataclass
class TrainConfig:
    train_input_dir: str|list = None
    train_label_dir: str = None
    valid_input_dir: str = None
    valid_label_dir: str = None
    out_dir: str = None
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    monitor_metric: str = "val_iou"  # "val_iou" or "val_spec"
    spec_fp_penalty: float = 5.0  # Penalize false positives when using specificity to evaluate the model's prediction.
    device: str = "cuda:0"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True


# -------------------------
# Model configuration
# -------------------------
@dataclass
class ModelConfig:
    input_ch: int = 2  # Number of channels of the input data.
    data_shape: tuple = (256, 256)  # Input data shape
    num_classes: int = 1  # Number of foreground classes.
    dilation_rates: tuple = (1, 3, 5, 7)  # Dilation rates of the atrous convolution kernel.
    aspp_out: int = 256  # Number of channels of the ASPP module's output.


# -------------------------
# Train loop
# -------------------------

def train(cfgTrain: TrainConfig, cfgModel: ModelConfig):
    os.makedirs(cfgTrain.out_dir, exist_ok=True)
    seed_everything(cfgTrain.seed)

    # Datasets / loaders
    train_set = ATVDataset(
        input_dir=cfgTrain.train_input_dir,
        label_dir=cfgTrain.train_label_dir,
        sample_shape=cfgModel.data_shape,
        n_channel=cfgModel.input_ch,
        return_filename=False,
    )
    valid_set = ATVDataset(
        input_dir=cfgTrain.valid_input_dir,
        label_dir=cfgTrain.valid_label_dir,
        sample_shape=cfgModel.data_shape,
        n_channel=cfgModel.input_ch,
        return_filename=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfgTrain.batch_size,
        shuffle=True,
        num_workers=cfgTrain.num_workers,
        pin_memory=cfgTrain.pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfgTrain.batch_size,
        shuffle=False,
        num_workers=cfgTrain.num_workers,
        pin_memory=cfgTrain.pin_memory,
        drop_last=False,
    )

    print(f"Training data loaded from {cfgTrain.train_input_dir}")
    print(f"Validation data loaded from {cfgTrain.valid_input_dir}")
    x0, y0 = train_set[0]
    xv, yv = valid_set[0]
    print(f"Number of training samples: {train_set.__len__()}")
    print(f"Number of validation samples: {valid_set.__len__()}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")
    print(f"Batch size: {cfgTrain.batch_size}")
    print("Training input shape:", x0.shape)
    print("Training label shape:", y0.shape)
    print("Validation input shape:", xv.shape)
    print("Validation label shape:", yv.shape)

    # Model
    model = DeepLabV3PlusResNet18(
        in_ch=cfgModel.input_ch,
        num_classes=cfgModel.num_classes,
        aspp_out=cfgModel.aspp_out, 
        rates=cfgModel.dilation_rates, 
    ).to(cfgTrain.device)

    # Loss
    pos_weight = compute_pos_weight(train_loader, device=cfgTrain.device)
    print("pos_weight = ", pos_weight.item())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgTrain.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgTrain.epochs)

    # Logging state
    best_iou = -1.0
    best_spec = -1.0
    history = {
        "epoch": [],
        "train_loss_sum": [],
        "train_spec_score": [], "train_iou_score": [],
        "val_loss_sum": [],
        "val_spec_score": [], "val_iou_score": [],
        "learning_rate": [],
    }

    for epoch in range(1, cfgTrain.epochs + 1):
        history["epoch"].append(epoch)

        # -------------------------
        # Train one epoch
        # -------------------------
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_spec_sum = 0.0; train_iou_sum = 0.0
        train_spec_n = 0; train_iou_n = 0
        for i, (xb, yb) in enumerate(train_loader):
            sys.stdout.write(f"\rEpoch: {epoch}/{cfgTrain.epochs} Training [{i+1}/{len(train_loader)}]")
            xb = xb.to(cfgTrain.device, non_blocking=True)
            yb = yb.to(cfgTrain.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            with torch.no_grad():
                is_pos = (yb.sum(dim=(1, 2, 3)) > 0)
                if is_pos.any():
                    train_iou_sum += iou_score(pred[is_pos], yb[is_pos], thr=0.5) * int(is_pos.sum().item())
                    train_iou_n += int(is_pos.sum().item())
                is_neg = ~is_pos
                if is_neg.any():
                    train_spec_sum += specificity_score(pred[is_neg], yb[is_neg], thr=0.5, fp_penalty=cfgTrain.spec_fp_penalty) * int(is_neg.sum().item())
                    train_spec_n += int(is_neg.sum().item())
        scheduler.step()
        # Aggregate train stats
        ntr = max(1, len(train_loader))
        train_loss_sum /= ntr; history["train_loss_sum"].append(train_loss_sum)
        train_iou = (train_iou_sum / train_iou_n) if train_iou_n > 0 else float("nan")
        train_spec = (train_spec_sum / train_spec_n) if train_spec_n > 0 else float("nan")
        history["train_iou_score"].append(float(train_iou))
        history["train_spec_score"].append(float(train_spec))
        sys.stdout.write("\n")

        # -------------------------
        # Validate one epoch
        # -------------------------
        model.eval()
        val_loss_sum = 0.0
        val_spec_sum = 0.0; val_iou_sum = 0.0
        val_spec_n = 0; val_iou_n = 0
        with torch.no_grad():
            for i, (xb, yb) in enumerate(valid_loader):
                sys.stdout.write(f"\rEpoch: {epoch}/{cfgTrain.epochs} Validating [{i+1}/{len(valid_loader)}]")
                xb = xb.to(cfgTrain.device, non_blocking=True)
                yb = yb.to(cfgTrain.device, non_blocking=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()
                is_pos = (yb.sum(dim=(1, 2, 3)) > 0)
                if is_pos.any():
                    val_iou_sum += iou_score(pred[is_pos], yb[is_pos], thr=0.5) * int(is_pos.sum().item())
                    val_iou_n += int(is_pos.sum().item())
                is_neg = ~is_pos
                if is_neg.any():
                    val_spec_sum += specificity_score(pred[is_neg], yb[is_neg], thr=0.5, fp_penalty=cfgTrain.spec_fp_penalty) * int(is_neg.sum().item())
                val_spec_n += int(is_neg.sum().item())
        # Aggregate validation stats.
        nva = max(1, len(valid_loader))
        val_loss_sum /= nva; history["val_loss_sum"].append(val_loss_sum)
        val_iou = (val_iou_sum / val_iou_n) if val_iou_n > 0 else float("num")
        val_spec = (val_spec_sum / val_spec_n) if val_spec_n > 0 else float("num")
        history["val_iou_score"].append(float(val_iou))
        history["val_spec_score"].append(float(val_spec))
        sys.stdout.write("\n")

        # Record time & LR
        dt = time.time() - t0
        history["learning_rate"].append(scheduler.get_last_lr()[0])

        # Console log
        print(
            f"Epoch {epoch}/{cfgTrain.epochs} | lr={scheduler.get_last_lr()[0]:.3e} | "
            f"train_loss={train_loss_sum:.4f} | "
            f"train_spec={train_spec:.4f} | train_iou={train_iou:.4f} | "
            f"val_spec={val_spec:.4f} | val_iou={val_iou:.4f}"
        )

        # Save best checkpoint
        if cfgTrain.monitor_metric == "val_spec":
            if val_spec > best_spec:
                best_spec = val_spec
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_score": best_spec,
                    "train_config": cfgTrain.__dict__,
                    "model_config": cfgModel.__dict__,
                }
                torch.save(ckpt, os.path.join(cfgTrain.out_dir, "best.pt"))
        elif cfgTrain.monitor_metric == "val_iou":
            if val_iou > best_iou:
                best_iou = val_iou
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_score": best_iou,
                    "train_config": cfgTrain.__dict__,
                    "model_config": cfgModel.__dict__,
                }
                torch.save(ckpt, os.path.join(cfgTrain.out_dir, "best.pt"))

        # Persist history each epoch
        pd.DataFrame(history).to_csv(os.path.join(cfgTrain.out_dir, "history.csv"), index=False)

    # Save last checkpoint
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
            "train_config": cfgTrain.__dict__,
            "model_config": cfgModel.__dict__,
        },
        os.path.join(cfgTrain.out_dir, "last.pt"),
    )

    # Plots
    plot_history(history, cfgTrain.out_dir)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    cfgTrain = TrainConfig(
        train_input_dir=[
           "./data/train/amplitude", 
           "./data/train/radius",
        ],
        train_label_dir="./data/trian/label",
        valid_input_dir=[
            "./data/valid/amplitude",
            "./data/valid/radius",
        ],
        valid_label_dir="./data/valid/label",
        out_dir="./checkpoint",
        device="cuda:0",
        epochs=200,
        batch_size=8,
        lr=1e-4,
        monitor_metric="val_iou", 
        spec_fp_penalty=10.0,   # Penalize false positives when using specificity to evaluate the model's prediction.
        seed=42,
        num_workers=4,
        pin_memory=True,
    )

    cfgModel = ModelConfig(
        input_ch=2,
        data_shape=(256, 256),
        num_classes=1,
        dilation_rates=(1, 3, 5, 7),
        aspp_out=256
    )

    train(cfgTrain, cfgModel)
