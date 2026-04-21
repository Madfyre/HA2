"""
train_task2.py — fine-tune ResNet50 on a 100-class ImageFolder dataset using
real DDP via torchrun.

Usage (on Imladris, inside tmux):

    CUDA_VISIBLE_DEVICES=0,4,5 torchrun --nproc_per_node=3 train_task2.py \
        --data-root ~/NES_HA\!/HA2 --epochs 40 --batch-size 64 --lr 1e-3

The output file `net_task2.pt` is the state_dict of the underlying YourNet
(not a Lightning checkpoint) and can be loaded in a notebook for evaluation:

    net = YourNet(ncl=100)
    net.load_state_dict(torch.load('net_task2.pt', map_location='cuda'))
"""

import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


# -----------------------------------------------------------------------------
# Seeds
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model — keep the same API as in the notebook so evaluate_task() works:
#   model(imgs, tgt) -> loss;   model.get_accuracy(reset=True) -> float
# -----------------------------------------------------------------------------
class YourNet(nn.Module):
    def __init__(self, ncl: int = 100):
        super().__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.net.fc = nn.Linear(self.net.fc.in_features, ncl)
        self.hit = 0
        self.tot = 0

    def _forward(self, x):
        return self.net(x)

    def forward(self, imgs, tgt=None):
        lgt = self._forward(imgs)
        if tgt is None:
            return lgt
        prd = lgt.argmax(1)
        self.hit += (prd == tgt).sum().item()
        self.tot += tgt.size(0)
        return F.cross_entropy(lgt, tgt)

    def get_accuracy(self, reset: bool = False) -> float:
        acc = self.hit / max(self.tot, 1)
        if reset:
            self.hit = 0
            self.tot = 0
        return acc


# -----------------------------------------------------------------------------
# Lightning module — only used at training time.
# Metrics are computed via plain tensor ops and synced across DDP ranks via
# sync_dist=True (the self.hit/self.tot counter in YourNet is NOT DDP-safe).
# -----------------------------------------------------------------------------
class LitResNet(pl.LightningModule):
    def __init__(
        self,
        ncl: int = 100,
        lr: float = 1e-3,
        max_epochs: int = 40,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = YourNet(ncl=ncl)

    def forward(self, x):
        return self.net._forward(x)

    def configure_optimizers(self):
        # Backbone gets 10x smaller LR than the freshly-initialised head.
        head, backbone = [], []
        for name, p in self.net.named_parameters():
            (head if name.startswith("net.fc") else backbone).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.hparams.lr * 0.1},
                {"params": head,     "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    def _shared_step(self, batch, stage: str):
        img, lbl = batch
        logits = self.net._forward(img)
        loss = F.cross_entropy(logits, lbl)
        acc = (logits.argmax(1) == lbl).float().mean()
        # sync_dist=True averages the metric across all DDP ranks for logging.
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(data_root: str, batch_size: int, num_workers: int):
    mn = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]

    tf_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
        transforms.RandomErasing(p=0.2),
    ])
    tf_v = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
    ])

    ds_t = ImageFolder(os.path.join(data_root, "train"), transform=tf_t)
    ds_v = ImageFolder(os.path.join(data_root, "val"),   transform=tf_v)

    # DataLoader runs in each DDP process; Lightning wraps the sampler with
    # DistributedSampler automatically, so each GPU sees its own 1/N slice.
    dl_t = DataLoader(
        ds_t, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    dl_v = DataLoader(
        ds_v, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    return dl_t, dl_v


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.expanduser("~/NES_HA!/HA2"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PER-GPU batch size. Effective batch = N_GPU * batch_size.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--out", default="net_task2.pt")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")   # enable TF32 on A4000/RTX 4000

    dl_t, dl_v = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    model = LitResNet(ncl=100, lr=args.lr, max_epochs=args.epochs)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch}-{val_acc:.3f}",
    )
    trainer = pl.Trainer(
        logger=CSVLogger(args.log_dir, name="task2"),
        accelerator="cuda",
        devices=-1,                 # use all GPUs visible via CUDA_VISIBLE_DEVICES
        strategy="ddp",             # REAL DDP (torchrun), NOT ddp_notebook
        max_epochs=args.epochs,
        precision="16-mixed",
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                       patience=args.patience),
            ckpt_cb,
        ],
    )

    trainer.fit(model, dl_t, dl_v)

    # Save only on rank 0; otherwise N processes race on the same file.
    if trainer.is_global_zero:
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            # Reload the best epoch's weights (not the last-epoch weights) and
            # save just the underlying YourNet state_dict for notebook eval.
            best = LitResNet.load_from_checkpoint(best_path)
            torch.save(best.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved BEST weights to {args.out}")
            print(f"[rank 0] Best val_acc = {ckpt_cb.best_model_score:.4f}")
            print(f"[rank 0] Best ckpt    = {best_path}")
        else:
            torch.save(model.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved final (last-epoch) weights to {args.out}")


if __name__ == "__main__":
    main()
