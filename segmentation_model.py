#!/usr/bin/env python3
# segmentation_model_refactored.py
"""
Segmentation trainer, optimized to use the single-channel texture-enhanced images
produced by prepare_dataset.py. Uses ACA-Atrous-ResUNet only and includes
AMP (mixed precision), robust sampler, gradient accumulation, and logging.
"""
from __future__ import annotations
import argparse
import os
import random
from typing import Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torchvision

# ----------------------------
# Dataset
# ----------------------------
class BreastSegDataset(Dataset):
    """
    Expects CSV with `image_file_path` and `roi_mask_file_path`.
    Input image is a single-channel texture-enhanced grayscale PNG produced by prepare_dataset.py.
    """
    def __init__(self, csv_file: str, resize: Tuple[int, int] = (512, 512), augment: bool = False):
        df = pd.read_csv(csv_file)
        self.images: List[str] = df["image_file_path"].tolist()
        self.masks: List[str] = df["roi_mask_file_path"].tolist()
        self.resize = resize
        self.augment = augment
        self.transform = self._get_transforms()

    def _get_transforms(self) -> A.Compose:
        # We resize first so Albumentations shape checks pass reliably
        common = [
            A.Resize(self.resize[0], self.resize[1]),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ]
        if self.augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                *common
            ]
            # keep shape checks on (should be consistent because of Resize)
            return A.Compose(aug, additional_targets={"mask": "mask"})
        else:
            return A.Compose(common, additional_targets={"mask": "mask"})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        mask = (mask > 0).astype(np.uint8) * 255  # ensure binary 0/255

        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]          # 1 x H x W, normalized (-1..1)
        mask_t = augmented["mask"].unsqueeze(0).float() / 255.0  # 1 x H x W (0/1)

        return img_t.float(), mask_t.float()

# ----------------------------
# ACA block, UNet pieces, and ACAAtrousResUNet (unchanged but included)
# ----------------------------
class ACAModule(nn.Module):
    def __init__(self, skip_channels, gate_channels, reduction=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, max(skip_channels // reduction,1), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(skip_channels // reduction,1), skip_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(skip_channels + gate_channels, max(skip_channels // reduction,1), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(skip_channels // reduction,1), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels), nn.ReLU(inplace=True)
        )
    def forward(self, skip, gate):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)
        concat = torch.cat([skip, gate], dim=1)
        ca = self.ca(concat)
        sa = self.spatial(concat)
        refined = skip * ca * sa + skip
        return self.fuse(refined)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UpACA(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.aca = ACAModule(skip_channels=skip_ch, gate_channels=in_ch)
        self.conv = DoubleConv(skip_ch + in_ch, out_ch, dropout=dropout)
    def forward(self, x_decoder, x_encoder):
        x = self.up(x_decoder)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        skip_ref = self.aca(x_encoder, x)
        out = torch.cat([skip_ref, x], dim=1)
        return self.conv(out)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1,6,12,18)):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False) for r in rates])
        self.bn = nn.BatchNorm2d(out_ch * len(rates))
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Sequential(nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        feats = [blk(x) for blk in self.blocks]
        x = torch.cat(feats, dim=1)
        x = self.relu(self.bn(x))
        x = self.project(x)
        return x

class ACAAtrousResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # SMP Unet encoder but we will only use this single architecture
        self.encoder = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=out_ch)
        encoder_channels = self.encoder.encoder.out_channels
        self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])
        self.up_aca1 = UpACA(in_ch=encoder_channels[-2], out_ch=encoder_channels[-3], skip_ch=encoder_channels[-2])
        self.up_aca2 = UpACA(in_ch=encoder_channels[-3], out_ch=encoder_channels[-4], skip_ch=encoder_channels[-3])
        self.up_aca3 = UpACA(in_ch=encoder_channels[-4], out_ch=encoder_channels[-5], skip_ch=encoder_channels[-4])
        self.up_aca4 = UpACA(in_ch=encoder_channels[-5], out_ch=encoder_channels[-5], skip_ch=encoder_channels[-5])
        self.outc = nn.Conv2d(in_channels=encoder_channels[-5], out_channels=out_ch, kernel_size=1)
    def forward(self, x):
        feats = self.encoder.encoder(x)
        e1, e2, e3, e4, bottleneck = feats[1], feats[2], feats[3], feats[4], feats[5]
        d5 = self.aspp(bottleneck)
        d4 = self.up_aca1(d5, e4)
        d3 = self.up_aca2(d4, e3)
        d2 = self.up_aca3(d3, e2)
        d1 = self.up_aca4(d2, e1)
        logits = self.outc(d1)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# ----------------------------
# Loss & helpers
# ----------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        inter = (inputs_sigmoid * targets_flat).sum()
        dice_score = (2. * inter + self.smooth) / (inputs_sigmoid.sum() + targets_flat.sum() + self.smooth)
        return bce + (1 - dice_score)

def l1_regularization(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    return l1_lambda * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

def dice_per_sample(preds, targets, smooth=1e-5):
    inter = (preds * targets).sum((1,2,3))
    denom = preds.sum((1,2,3)) + targets.sum((1,2,3))
    return (2.*inter + smooth) / (denom + smooth)

# ----------------------------
# Trainer with AMP and accumulation
# ----------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.writer = SummaryWriter(log_dir=args.logdir)
        self.best_val_dice = 0.0
        self.pos_weight = torch.tensor([args.pos_weight], device=self.device)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        os.makedirs(args.outdir, exist_ok=True)

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        step = 0
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}")
        accum_steps = max(1, self.args.accum_steps)
        for imgs, masks in pbar:
            imgs = imgs.to(self.device); masks = masks.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                logits = self.model(imgs)
                loss = self.criterion(logits, masks, pos_weight=self.pos_weight)
                l1_penalty = l1_regularization(self.model, self.args.l1_lambda)
                total_loss_batch = loss + l1_penalty
                total_loss_batch = total_loss_batch / accum_steps

            # backprop with scaler if available
            if self.scaler is not None:
                self.scaler.scale(total_loss_batch).backward()
                if (step + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                total_loss_batch.backward()
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            if (step + 1) % accum_steps == 0 and self.scaler is None:
                self.optimizer.zero_grad()

            total_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1_penalty.item():.6f}")

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val E{epoch}/{self.args.epochs}")
            for imgs, masks in pbar:
                imgs = imgs.to(self.device); masks = masks.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, masks, pos_weight=self.pos_weight)
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                val_dice += dice_per_sample(probs, masks).mean().item()
                pbar.set_postfix(dice=f"{val_dice / (pbar.n + 1):.4f}")

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)

        self.scheduler.step(avg_val_dice)
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("val/dice", avg_val_dice, epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # save best
        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            torch.save(self.model.state_dict(), os.path.join(self.args.outdir, "best.pth"))
            print(f"New best model saved ({avg_val_dice:.4f})")

        torch.save(self.model.state_dict(), os.path.join(self.args.outdir, f"epoch_{epoch}.pth"))
        self._log_images(epoch)

    def _log_images(self, epoch: int):
        if epoch % 3 != 0:
            return
        imgs, masks = next(iter(self.val_loader))
        idx = random.randint(0, imgs.size(0)-1)
        img = imgs[idx:idx+1].to(self.device)
        mask = masks[idx:idx+1]
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.sigmoid(logits if not isinstance(logits, tuple) else logits[0])
        def to_rgb(x):
            return x.squeeze(0).cpu().repeat(3,1,1)
        grid = torchvision.utils.make_grid([to_rgb(img.cpu()), to_rgb(mask), to_rgb((probs>0.5).float())],
                                           nrow=3, normalize=True, scale_each=True)
        self.writer.add_image("val/sample_prediction", grid, epoch)

    def run(self):
        for epoch in range(1, self.args.epochs+1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
        self.writer.close()

# ----------------------------
# Utilities & main
# ----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Train ACA-Atrous-ResUNet on texture-enhanced single-channel images")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="final_checkpoints")
    p.add_argument("--logdir", type=str, default="runs/aca_resunet_75ep_improvised")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-weight", type=float, default=12.0)
    p.add_argument("--l1-lambda", type=float, default=4.5e-4)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--seed", type=int, default=32)
    return p.parse_args()

def seed_everything(seed: int = 32):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def setup_dataloaders(args):
    ds = BreastSegDataset(args.csv, resize=(args.img_size, args.img_size), augment=True)
    val_len = int(len(ds) * 0.2); train_len = len(ds) - val_len
    tr, vl = random_split(ds, [train_len, val_len])

    # robust image-level sampler (handles masks that might not exist or be empty)
    df = pd.read_csv(args.csv)
    train_mask_paths = [df.loc[i, "roi_mask_file_path"] for i in tr.indices]
    class_labels=[]
    for p in tqdm(train_mask_paths, desc="Preparing sampler"):
        try:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                class_labels.append(0)
            else:
                class_labels.append(1 if m.sum() > 0 else 0)
        except Exception:
            class_labels.append(0)
    counts = np.bincount(class_labels)
    counts = np.where(counts==0, 1, counts)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in class_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    vl_loader = DataLoader(vl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tr_loader, vl_loader

def create_model(device):
    model = ACAAtrousResUNet(in_ch=1, out_ch=1).to(device)
    return model

def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tr_loader, vl_loader = setup_dataloaders(args)
    model = create_model(device)

    criterion = DiceBCELoss(smooth=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                      criterion=criterion, device=device, train_loader=tr_loader, val_loader=vl_loader, args=args)
    trainer.run()

if __name__ == "__main__":
    main()
