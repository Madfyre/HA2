"""
train_task2.py — fine-tune ConvNeXt-Tiny on a 100-class ImageFolder dataset
using real DDP via torchrun.

Usage (on Imladris, inside tmux):

    CUDA_VISIBLE_DEVICES=0,4,5 torchrun --nproc_per_node=3 train_task2.py \\
        --data-root ~/NES_HA\\!/HA2 --epochs 35 --batch-size 64 --lr 1e-3

Target: >= 0.4 val_acc in ~1 hour on 3x A4000.

The output file `net_task2.pt` is the state_dict of the underlying YourNet
and can be loaded in a notebook for evaluation on a single GPU:

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
# Model
#
# ConvNeXt-Tiny's classifier is:
#   Sequential(
#     (0): LayerNorm2d
#     (1): Flatten
#     (2): Linear(in_features=768, out_features=1000)
#   )
# So we replace classifier[2] to get the right number of output classes.
#
# Interface (forward / get_accuracy) is kept compatible with evaluate_task()
# from the notebook.
# -----------------------------------------------------------------------------
class YourNet(nn.Module):
    def __init__(self, ncl: int = 100):
        super().__init__()
        self.net = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        in_features = self.net.classifier[2].in_features
        self.net.classifier[2] = nn.Linear(in_features, ncl)
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
# LightningModule
# -----------------------------------------------------------------------------
class LitConvNeXt(pl.LightningModule):
    def __init__(
        self,
        ncl: int = 100,
        lr: float = 1e-3,
        max_epochs: int = 35,
        warmup_epochs: int = 2,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        backbone_lr_mult: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = YourNet(ncl=ncl)

    def forward(self, x):
        return self.net._forward(x)

    def configure_optimizers(self):
        # Fresh classifier head -> full LR.
        # Pretrained backbone  -> lr * backbone_lr_mult (0.1 by default).
        head_params, backbone_params = [], []
        for name, p in self.net.named_parameters():
            if name.startswith("net.classifier"):
                head_params.append(p)
            else:
                backbone_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.lr * self.hparams.backbone_lr_mult},
                {"params": head_params,     "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # Warmup (linear 0 -> full LR over `warmup_epochs` epochs),
        # then cosine anneal to 0 over the remaining epochs.
        warmup = LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs),
        )
        sched = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    def _shared_step(self, batch, stage: str):
        img, lbl = batch
        logits = self.net._forward(img)
        loss = F.cross_entropy(
            logits, lbl, label_smoothing=self.hparams.label_smoothing
        )
        acc = (logits.argmax(1) == lbl).float().mean()
        # sync_dist=True averages the logged metric across DDP ranks.
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

    # AutoAugment(IMAGENET) — stronger, principled aug covered in Seminar 4
    # (where it's used with the CIFAR10 policy). We pick IMAGENET because
    # our inputs are 224x224 and the backbone was pretrained on ImageNet.
    tf_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
        transforms.RandomErasing(p=0.25),
    ])
    tf_v = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
    ])

    ds_t = ImageFolder(os.path.join(data_root, "train"), transform=tf_t)
    ds_v = ImageFolder(os.path.join(data_root, "val"),   transform=tf_v)

    # DataLoader runs per-process. Lightning auto-wraps the sampler with
    # DistributedSampler so each GPU sees its own 1/N shard of the data.
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
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.expanduser("~/NES_HA!/HA2"))
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64,
                   help="PER-GPU batch size. Effective batch = N_GPU * batch_size.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--out", default="net_task2.pt")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")   # enable TF32 matmul on Ampere

    dl_t, dl_v = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    model = LitConvNeXt(
        ncl=100,
        lr=args.lr,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1,
        filename="best-{epoch}-{val_acc:.3f}",
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    early = pl.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=args.patience
    )

    trainer = pl.Trainer(
        logger=CSVLogger(args.log_dir, name="task2"),
        accelerator="cuda",
        devices=-1,                # all GPUs visible via CUDA_VISIBLE_DEVICES
        strategy="ddp",            # REAL DDP via torchrun, NOT ddp_notebook
        max_epochs=args.epochs,
        precision="16-mixed",
        sync_batchnorm=True,
        callbacks=[ckpt_cb, early, lr_cb],
    )

    trainer.fit(model, dl_t, dl_v)

    # Rank 0 saves; other ranks exit.
    if trainer.is_global_zero:
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            best = LitConvNeXt.load_from_checkpoint(best_path)
            torch.save(best.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved BEST weights to {args.out}")
            print(f"[rank 0] Best val_acc = {ckpt_cb.best_model_score:.4f}")
            print(f"[rank 0] Best ckpt    = {best_path}")
        else:
            torch.save(model.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved final (last-epoch) weights to {args.out}")


if __name__ == "__main__":
    main()
