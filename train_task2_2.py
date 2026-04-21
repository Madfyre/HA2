"""
    CUDA_VISIBLE_DEVICES=0,4,5 torchrun --nproc_per_node=3 train_task2_2.py \\
        --data-root ~/NES_HA\\!/HA2 --epochs 50 --batch-size 32 --lr 8e-4
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class YourNet(nn.Module):
    def __init__(self, ncl: int = 100, stochastic_depth_prob: float = 0.1):
        super().__init__()
        self.net = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
            stochastic_depth_prob=stochastic_depth_prob,
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


class LitConvNeXt(pl.LightningModule):
    def __init__(
        self,
        ncl: int = 100,
        lr: float = 8e-4,
        max_epochs: int = 50,
        warmup_epochs: int = 3,
        weight_decay: float = 5e-2,
        label_smoothing: float = 0.1,
        backbone_lr_mult: float = 0.1,
        stochastic_depth_prob: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = YourNet(ncl=ncl, stochastic_depth_prob=stochastic_depth_prob)

    def forward(self, x):
        return self.net._forward(x)

    def configure_optimizers(self):
        head, backbone_decay, backbone_nodecay = [], [], []
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("net.classifier"):
                head.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                backbone_nodecay.append(p)
            else:
                backbone_decay.append(p)

        lr = self.hparams.lr
        bb_lr = lr * self.hparams.backbone_lr_mult
        opt = torch.optim.AdamW(
            [
                {"params": backbone_decay, "lr": bb_lr, "weight_decay": self.hparams.weight_decay},
                {"params": backbone_nodecay, "lr": bb_lr, "weight_decay": 0.0},
                {"params": head, "lr": lr, "weight_decay": self.hparams.weight_decay},
            ],
        )
        warmup = LinearLR(
            opt, start_factor=1e-3, end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs),
        )
        sched = SequentialLR(
            opt, schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )
        return {"optimizer": opt, "lr_scheduler": sched}


    def _shared_step(self, batch, stage: str):
        img, lbl = batch
        logits = self.net._forward(img)
        loss = F.cross_entropy(
            logits, lbl, label_smoothing=self.hparams.label_smoothing
        )
        acc = (logits.argmax(1) == lbl).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, sync_dist=True)
        return loss


    def training_step(self, batch, _):
        return self._shared_step(batch, "train")


    def validation_step(self, batch, _):
        self._shared_step(batch, "val")


def build_dataloaders(data_root: str, batch_size: int, num_workers: int, img_size: int):
    mn = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]

    tf_t = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
        transforms.RandomErasing(p=0.25),
    ])

    tf_v = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mn, sd),
    ])

    ds_t = ImageFolder(os.path.join(data_root, "train"), transform=tf_t)
    ds_v = ImageFolder(os.path.join(data_root, "val"),   transform=tf_v)

    dl_t = DataLoader(
        ds_t, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
    )
    dl_v = DataLoader(
        ds_v, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
    )
    return dl_t, dl_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.expanduser("~/NES_HA!/HA2"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16,
                   help="PER-GPU batch size. Effective = N_GPU * batch_size.")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Peak head LR. Backbone uses lr * 0.1.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--img-size", type=int, default=232)
    p.add_argument("--stochastic-depth-prob", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--out", default="net_task2.pt")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--accumulate-grad-batches", type=int, default=2,
                   help="Bump if you can't fit the batch size you want.")

    args = p.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")  # TF32 matmul

    dl_t, dl_v = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers, args.img_size
    )
    model = LitConvNeXt(
        ncl=100,
        lr=args.lr,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        stochastic_depth_prob=args.stochastic_depth_prob,
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
        logger=CSVLogger(args.log_dir, name="task2_base"),
        accelerator="cuda",
        devices=-1,
        strategy="ddp",
        max_epochs=args.epochs,
        precision="16-mixed",
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, early, lr_cb],
    )

    trainer.fit(model, dl_t, dl_v)

    if trainer.is_global_zero:
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            best = LitConvNeXt.load_from_checkpoint(best_path)
            torch.save(best.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved BEST weights to {args.out}")
            print(f"[rank 0] Best val_acc = {ckpt_cb.best_model_score:.4f}")
            print(f"[rank 0] Best ckpt = {best_path}")
        else:
            torch.save(model.net.state_dict(), args.out)
            print(f"\n[rank 0] Saved final (last-epoch) weights to {args.out}")


if __name__ == "__main__":
    main()
