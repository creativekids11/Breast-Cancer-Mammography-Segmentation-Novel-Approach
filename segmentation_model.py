#!/usr/bin/env python3
# segmentation_model.py
"""
Segmentation using ACAAtrousResUNet with:
 - optional conditional pectoral removal at training-time
 - using precomputed GLCM contrast heatmaps (loaded from CSV)
 - GLCM as extra channel
 - hard-example mining (upweight hardest fraction in each batch)
 - separate small-object & large-object Dice logging
 - image logging every 5 epochs (train + val)
"""
import argparse
import os
import random
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torchvision

from skimage.feature import graycomatrix, graycoprops

# ----------------------------
# Dataset
# ----------------------------
class BreastSegDataset(Dataset):
    def __init__(self, csv_file: str, resize: Tuple[int, int] = (512, 512), augment: bool = False,
                 remove_pectoral: bool = False):
        df = pd.read_csv(csv_file)
        self.images: List[str] = df["image_file_path"].tolist()
        self.masks: List[str] = df["roi_mask_file_path"].tolist()

        # require precomputed GLCM paths
        if "glcm_file_path" not in df.columns:
            raise ValueError("CSV must include 'glcm_file_path' column (from precompute_glcm.py)")
        self.glcms: List[str] = df["glcm_file_path"].tolist()

        self.resize = resize
        self.augment = augment
        self.remove_pectoral = remove_pectoral
        self.transform = self._get_transforms()

    def _get_transforms(self):
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
            return A.Compose(aug, additional_targets={"mask": "mask"})
        else:
            return A.Compose(common, additional_targets={"mask": "mask"})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        # read image + mask
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # read mask with preserved channels if any

        if img is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        # Optional conditional pectoral removal (only replace img if muscle detected)
        if self.remove_pectoral:
            img_clean, p_mask, removed = remove_pectoral_by_cc(img)
            if removed:
                img = img_clean

        # If mask has multiple channels (e.g. 3-channel PNG), convert to single channel
        if mask.ndim == 3:
            # assume mask is identical across channels or only first channel contains label
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # If mask and image sizes differ, resize mask to match image using nearest neighbor
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure binary mask (0/255)
        mask = (mask > 0).astype(np.uint8) * 255

        # Apply same resize/normalize transforms (Albumentations will resize image & mask to self.resize)
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]   # [1,H,W], float32 normalized approx -1..1 (mean=0.5,std=0.5 normalization)
        mask_t = augmented["mask"].unsqueeze(0) / 255.0  # [1,H,W] in {0,1}

        # Load precomputed GLCM
        glcm_path = self.glcms[idx]
        if not os.path.exists(glcm_path):
            raise FileNotFoundError(f"GLCM file not found: {glcm_path}")
        heat = np.load(glcm_path).astype(np.float32)

        # If heat size differs from dataset resize, resize to match
        if heat.shape != (self.resize[0], self.resize[1]):
            heat = cv2.resize(heat, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize heat to same normalization as image (-1..1 after mean=0.5/std=0.5)
        # (precompute script should produce heat in [0,1])
        heat_norm = (heat - 0.5) / 0.5
        heat_t = torch.from_numpy(heat_norm).unsqueeze(0).float()

        # Concat: input is 2 channels (raw + glcm)
        inp = torch.cat([img_t.float(), heat_t], dim=0)
        return inp, mask_t


# ----------------------------
# Pectoral removal (component-based). Returns (clean_img, mask, removed_flag)
# ----------------------------
def remove_pectoral_by_cc(img_uint8, thresh_method='otsu', min_area_frac=0.001):
    """
    Detect and remove pectoral muscle if present.
    Returns cleaned image, pectoral mask (255), and removed flag.
    """
    H, W = img_uint8.shape
    img = cv2.equalizeHist(img_uint8)
    # Otsu threshold
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # connected components
    num_labels, labels = cv2.connectedComponents(bw)
    top_strip = img[:max(1, H//8), :]
    left_sum = int(top_strip[:, :max(1, W//3)].sum())
    right_sum = int(top_strip[:, -max(1, W//3):].sum())
    corner_side = 'left' if left_sum > right_sum else 'right'
    best_label = None; best_area = 0
    for lab in range(1, num_labels):
        mask = (labels == lab).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area_frac * H * W:
            continue
        if corner_side == 'left':
            touches = mask[:max(1, H//20), :max(1, W//20)].sum() > 0 or mask[:max(1, H//10), :max(1, W//10)].sum() > 0
        else:
            touches = mask[:max(1, H//20), -max(1, W//20):].sum() > 0 or mask[:max(1, H//10), -max(1, W//10):].sum() > 0
        if touches and area > best_area:
            best_area = area; best_label = lab
    pectoral_mask = np.zeros_like(img, dtype=np.uint8)
    clean = img_uint8.copy()
    if best_label is not None:
        mask = (labels == best_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(hull_mask, hull, 255)
            pectoral_mask = hull_mask
            clean = img_uint8.copy()
            clean[pectoral_mask > 0] = 0
            return clean, pectoral_mask, True
    return img_uint8, pectoral_mask, False


# ----------------------------
# GLCM helper (kept for reference; precompute used separate script)
# ----------------------------
def glcm_contrast_map(img_uint8, win=31, levels=8, step=8):
    H, W = img_uint8.shape
    half = win // 2
    if win < 3: win = 3; half = 1
    q = (img_uint8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    angles = [0]
    dists = [1]
    ys = list(range(half, H - half, step))
    xs = list(range(half, W - half, step))
    for y in ys:
        for x in xs:
            patch = q[y - half: y + half + 1, x - half: x + half + 1]
            try:
                glcm = graycomatrix(patch, distances=dists, angles=angles, levels=levels, symmetric=True, normed=True)
                contrast = float(graycoprops(glcm, 'contrast')[0, 0])
            except Exception:
                contrast = 0.0
            heat[y, x] += contrast
            count[y, x] += 1.0
    heat = gaussian_filter(heat, sigma=max(1.0, step/2.0))
    mn = heat.min(); mx = heat.max()
    if mx - mn < 1e-8:
        return np.zeros((H, W), dtype=np.float32)
    heat = (heat - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)


# ----------------------------
# Model (ACAAtrousResUNet)
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
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
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
# Loss & metrics
# ----------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__(); self.smooth = smooth
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_sigmoid * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_sigmoid.sum() + targets_flat.sum() + self.smooth)
        return bce + (1 - dice_score)

def dice_per_sample(preds, targets, smooth=1e-5):
    inter = (preds * targets).sum((1,2,3))
    denom = preds.sum((1,2,3)) + targets.sum((1,2,3))
    return (2.*inter + smooth) / (denom + smooth)


# ----------------------------
# Trainer with hard-example mining & small/large dice
# ----------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, device, train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.writer = SummaryWriter(log_dir=args.logdir)
        self.best_val_dice = 0.0
        self.pos_weight = torch.tensor([args.pos_weight], device=self.device)
        os.makedirs(args.outdir, exist_ok=True)

    def per_sample_loss(self, logits, masks):
        B = logits.size(0)
        bce = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=self.pos_weight, reduction='none')
        bce_per = bce.view(B, -1).mean(dim=1)
        probs = torch.sigmoid(logits)
        dice = dice_per_sample(probs, masks)
        return (1.0 - dice) + bce_per

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device); masks = masks.to(self.device)
            B = imgs.shape[0]
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            sample_losses = self.per_sample_loss(logits, masks)
            k = max(1, int(B * self.args.hard_fraction))
            if k < B:
                topk_vals, topk_idx = torch.topk(sample_losses, k=k, largest=True)
                weights = torch.ones_like(sample_losses, device=self.device)
                weights[topk_idx] = self.args.hard_weight
            else:
                weights = torch.ones_like(sample_losses, device=self.device) * self.args.hard_weight
            loss = (sample_losses * weights).mean()
            l1_penalty = self.args.l1_lambda * sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
            full_loss = loss + l1_penalty
            full_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1_penalty.item():.6f}")
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        small_dice_acc = 0.0; small_cnt = 0
        large_dice_acc = 0.0; large_cnt = 0
        pbar = tqdm(self.val_loader, desc=f"Val E{epoch}/{self.args.epochs}")
        with torch.no_grad():
            for imgs, masks in pbar:
                imgs = imgs.to(self.device); masks = masks.to(self.device)
                logits = self.model(imgs)
                loss = DiceBCELoss()(logits, masks, pos_weight=self.pos_weight)
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                dice_batch = dice_per_sample(probs, masks)
                val_dice += dice_batch.mean().item()

                areas = masks.sum(dim=(1,2,3)) / (imgs.shape[2] * imgs.shape[3])
                small_mask = areas < self.args.small_area_thresh
                large_mask = ~small_mask
                if small_mask.any():
                    small_dice_acc += dice_batch[small_mask].mean().item(); small_cnt += 1
                if large_mask.any():
                    large_dice_acc += dice_batch[large_mask].mean().item(); large_cnt += 1
                pbar.set_postfix(dice=f"{val_dice / (pbar.n + 1):.4f}")

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)
        avg_small = (small_dice_acc / small_cnt) if small_cnt > 0 else 0.0
        avg_large = (large_dice_acc / large_cnt) if large_cnt > 0 else 0.0

        self.scheduler.step(avg_val_dice)
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("val/dice", avg_val_dice, epoch)
        self.writer.add_scalar("val/dice_small", avg_small, epoch)
        self.writer.add_scalar("val/dice_large", avg_large, epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f} (small: {avg_small:.4f}, large: {avg_large:.4f})")
        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            torch.save(self.model.state_dict(), os.path.join(self.args.outdir, "best.pth"))
            print(f"New best saved ({avg_val_dice:.4f})")
        torch.save(self.model.state_dict(), os.path.join(self.args.outdir, f"epoch_{epoch}.pth"))

        # log images every 5 epochs (train + val)
        if epoch % 5 == 0:
            self._log_images(epoch, split="val")
            self._log_images(epoch, split="train")

    def _log_images(self, epoch: int, split: str = "val"):
        loader = self.val_loader if split == "val" else self.train_loader
        # defensive: if loader empty
        try:
            imgs, masks = next(iter(loader))
        except StopIteration:
            return
        idx = random.randint(0, imgs.size(0) - 1)
        img = imgs[idx:idx+1].to(self.device); mask = masks[idx:idx+1].to(self.device)
        with torch.no_grad():
            pred_logits = self.model(img)
            pred_prob = torch.sigmoid(pred_logits)

        # prepare visuals: extract raw image channel (channel 0) and convert to 3-channel rgb for tensorboard
        def tensor_to_rgb(t: torch.Tensor) -> torch.Tensor:
            """
            input t shape may be [1,H,W] or [1,1,H,W] or [1,C,H,W] with C>=1 (we'll pick first channel).
            returns (3,H,W) tensor in [0,1]
            """
            t = t.detach().cpu()
            if t.ndim == 4:
                # t: [B=1, C, H, W] -> pick first channel
                ch = t[0, 0, :, :].unsqueeze(0)
            elif t.ndim == 3:
                # [1,H,W] -> already first channel
                ch = t[0, :, :].unsqueeze(0)
            else:
                raise ValueError("Unexpected tensor shape for visualization")
            # denormalize: model input normalized with mean=0.5/std=0.5 earlier -> x = (x*0.5 + 0.5)
            ch = (ch * 0.5) + 0.5
            ch = ch.clamp(0.0, 1.0)
            # repeat to 3 channels
            rgb = ch.repeat(3, 1, 1)
            return rgb

        img_raw = tensor_to_rgb(img[:, 0:1, :, :])   # pick raw image channel
        mask_rgb = mask.squeeze(0).cpu().repeat(3, 1, 1)   # mask already [1,H,W]
        pred_rgb = (pred_prob > 0.5).float().squeeze(0).cpu().repeat(3, 1, 1)

        grid = torchvision.utils.make_grid([img_raw, mask_rgb, pred_rgb], nrow=3, normalize=False, scale_each=False)
        self.writer.add_image(f"{split}/sample", grid, epoch)

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
        self.writer.close()


# ----------------------------
# CLI & helpers
# ----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--logdir", type=str, default="runs/aca_resunet_glcm_75ep")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--pos-weight", type=float, default=12.0)
    p.add_argument("--l1-lambda", type=float, default=4.5e-4)
    # glcm args (used by dataset for resizing only)
    p.add_argument("--glcm-win", type=int, default=31)
    p.add_argument("--glcm-levels", type=int, default=8)
    p.add_argument("--glcm-step", type=int, default=8)
    # hard mining
    p.add_argument("--hard-fraction", type=float, default=0.3)
    p.add_argument("--hard-weight", type=float, default=3.0)
    p.add_argument("--small-area-thresh", type=float, default=0.001)
    # optional conditional pectoral removal at training-time
    p.add_argument("--remove-pectoral-train", action="store_true",
                   help="If set, attempt conditional pectoral removal on images at dataset __getitem__ (keeps image if none detected).")
    return p.parse_args()

def setup_dataloaders(args):
    ds = BreastSegDataset(args.csv, resize=(args.img_size,args.img_size), augment=True,
                          remove_pectoral=args.remove_pectoral_train if hasattr(args,'remove_pectoral_train') else False)
    val_len = int(len(ds) * 0.2)
    train_len = len(ds) - val_len
    tr, vl = random_split(ds, [train_len, val_len])
    # image-level sampler based on presence of mask
    df = pd.read_csv(args.csv)
    mask_paths = [df.loc[i, "roi_mask_file_path"] for i in tr.indices]
    class_labels = [1 if cv2.imread(p, cv2.IMREAD_GRAYSCALE).sum() > 0 else 0 for p in tqdm(mask_paths, desc="sampler")]
    counts = np.bincount(class_labels)
    counts[counts==0] = 1
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in class_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    vl_loader = DataLoader(vl, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return tr_loader, vl_loader

def create_model(device):
    model = ACAAtrousResUNet(in_ch=2, out_ch=1).to(device)
    return model

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(args.outdir, exist_ok=True)
    tr_loader, vl_loader = setup_dataloaders(args)
    model = create_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, device=device,
                      train_loader=tr_loader, val_loader=vl_loader, args=args)
    trainer.run()

if __name__ == "__main__":
    main()
