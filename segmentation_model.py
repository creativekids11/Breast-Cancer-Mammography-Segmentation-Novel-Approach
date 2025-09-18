#!/usr/bin/env python3
# segmentation_model_rl_glcm_glrlm.py
"""
Segmentation with ACA-Atrous UNet + Actor-Critic Policy
-------------------------------------------------------
Two-phase training:
  Phase 1 → Policy only (segmentation frozen as oracle)
    - optionally: Phase1 trains only CLAHE (gamma fixed)
  Phase 2 → Policy + segmentation model jointly (CLAHE+Gamma)

Changes vs original:
 - Phase1 can be run with only CLAHE action (use --phase1-clahe-only)
 - Replaced weaker handcrafted features (contrast/edge/clahe_hist) with
   GLCM + a simple GLRLM feature set for stronger texture descriptors.
 - Logging retained.
"""
import argparse, os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from math import pi

# =========================================================
# Dataset
# =========================================================
class BreastSegDataset(Dataset):
    def __init__(self, csv_file, resize=(512, 512), augment=False):
        df = pd.read_csv(csv_file)
        self.images = df["image_file_path"].tolist()
        self.masks = df["roi_mask_file_path"].tolist()
        self.resize, self.augment = resize, augment
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.Rotate(limit=20,p=0.5), A.RandomBrightnessContrast(p=0.2),
                A.Resize(resize[0], resize[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()
            ], additional_targets={"mask":"mask"})
        else:
            self.aug = A.Compose([
                A.Resize(resize[0], resize[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()
            ], additional_targets={"mask":"mask"})
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.resize[::-1])
        mask = cv2.resize(mask, self.resize[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (mask>0).astype(np.uint8)*255
        aug = self.aug(image=img, mask=mask)
        return aug["image"].float(), aug["mask"].unsqueeze(0).float()/255.0

# ----------------------------
# ACA block, UNet, ASPP, etc. (UNCHANGED except name)
# ----------------------------
class ACAModule(nn.Module):
    def __init__(self, skip_channels, gate_channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
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
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, bilinear=True, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if bilinear else nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_ch + in_ch, out_ch, dropout=dropout)
    def forward(self, x_decoder, x_encoder):
        x = self.up(x_decoder)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x_encoder, x], dim=1)
        return self.conv(out)

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
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [blk(x) for blk in self.blocks]
        x = torch.cat(feats, dim=1)
        x = self.relu(self.bn(x))
        x = self.project(x)
        return x

class ACAAtrousUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.aspp = ASPP(base_ch*8, base_ch*2)
        self.up1 = UpACA(base_ch*8, base_ch*4, base_ch*8, dropout=True)
        self.up2 = UpACA(base_ch*4, base_ch*2, base_ch*4, dropout=True)
        self.up3 = UpACA(base_ch*2, base_ch, base_ch*2, dropout=True)
        self.up4 = UpACA(base_ch, base_ch, base_ch, dropout=True)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*8)
        self.up1 = Up(base_ch*8, base_ch*4, base_ch*8, dropout=True)
        self.up2 = Up(base_ch*4, base_ch*2, base_ch*4, dropout=True)
        self.up3 = Up(base_ch*2, base_ch, base_ch*2, dropout=True)
        self.up4 = Up(base_ch, base_ch, base_ch, dropout=True)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1)
        x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5, x4); u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2); u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

class ConnectUNets(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.net1 = UNet(in_ch, out_ch, base_ch)
        self.net2 = UNet(in_ch + out_ch, out_ch, base_ch)
    def forward(self, x):
        pred1 = torch.sigmoid(self.net1(x))
        inp2 = torch.cat([x, pred1], dim=1)
        pred2 = self.net2(inp2)
        return pred2, pred1

class ACAAtrousResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.encoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )
        encoder_channels = self.encoder.encoder.out_channels
        self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])
        self.up_aca1 = UpACA(in_ch=encoder_channels[-2], out_ch=encoder_channels[-3], skip_ch=encoder_channels[-2])
        self.up_aca2 = UpACA(in_ch=encoder_channels[-3], out_ch=encoder_channels[-4], skip_ch=encoder_channels[-3])
        self.up_aca3 = UpACA(in_ch=encoder_channels[-4], out_ch=encoder_channels[-5], skip_ch=encoder_channels[-4])
        self.up_aca4 = UpACA(in_ch=encoder_channels[-5], out_ch=encoder_channels[-5], skip_ch=encoder_channels[-5])
        self.outc = nn.Conv2d(in_channels=encoder_channels[-5], out_channels=out_ch, kernel_size=1)
    def forward(self, x):
        encoder_features = self.encoder.encoder(x)
        e1, e2, e3, e4, bottleneck = encoder_features[1], encoder_features[2], encoder_features[3], encoder_features[4], encoder_features[5]
        d5 = self.aspp(bottleneck)
        d4 = self.up_aca1(d5, e4)
        d3 = self.up_aca2(d4, e3)
        d2 = self.up_aca3(d3, e2)
        d1 = self.up_aca4(d2, e1)
        logits = self.outc(d1)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# =========================================================
# Losses & Metrics
# =========================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5): super().__init__(); self.smooth=smooth
    def forward(self, inputs, targets, pos_weight=torch.tensor([10.0])):
        pos_weight=pos_weight.to(inputs.device)
        bce=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        inputs_sig=torch.sigmoid(inputs); inter=(inputs_sig.view(-1)*targets.view(-1)).sum()
        dice=(2.*inter+self.smooth)/(inputs_sig.sum()+targets.sum()+self.smooth)
        return 1-dice+bce(inputs,targets)

def dice_per_sample(preds, targets, smooth=1e-5):
    inter=(preds*targets).sum((1,2,3)); denom=preds.sum((1,2,3))+targets.sum((1,2,3))
    return (2.*inter+smooth)/(denom+smooth)

# =========================================================
# Helper: CLAHE + Gamma (OpenCV, numpy)
# =========================================================
def apply_clahe_gamma_np(imgs, clahe_clips, gammas):
    B,H,W=imgs.shape; out=np.zeros((B,H,W),np.float32)
    for i in range(B):
        clip=float(np.clip(clahe_clips[i],0.1,10.0)); gamma=float(np.clip(gammas[i],0.1,5.0))
        clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)); imc=clahe.apply(imgs[i])
        imc=imc.astype(np.float32)/255.0; out[i]=np.power(imc,gamma)
    return out

# -------------------------
# Additional handcrafted features for policy stats
# - replaced some simpler features with GLCM + GLRLM
# -------------------------
def compute_lbp(x_tensor, P=8, R=1):
    B = x_tensor.size(0)
    device = x_tensor.device
    lbp_bins = P + 2
    lbp_list = []
    try:
        for i in range(B):
            img = (x_tensor[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
            lbp = local_binary_pattern(img, P, R, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, P + 1))
            hist = torch.from_numpy(hist.astype(np.float32))
            hist = hist / (hist.sum() + 1e-8)
            lbp_list.append(hist)
        return torch.stack(lbp_list, dim=0).to(device)  # B x (P+2)
    except Exception:
        return torch.zeros((B, lbp_bins), dtype=torch.float32, device=device)

def compute_glcm_features(x_tensor, levels=8, distances=[1], angles=None):
    """Compute GLCM properties for each image.
    Returns tensor B x (num_props * num_angles)
    """
    if angles is None:
        angles = [0, pi/4, pi/2, 3*pi/4]
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    B = x_tensor.size(0)
    device = x_tensor.device
    out_list = []
    for i in range(B):
        img = (x_tensor[i].squeeze().cpu().numpy() * (levels-1)).astype(np.uint8)
        try:
            glcm = graycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            feats = []
            for p in props:
                val = graycoprops(glcm, p)  # shape (len(distances), len(angles))
                feats.append(val.flatten())
            feats = np.concatenate(feats, axis=0).astype(np.float32)
        except Exception:
            feats = np.zeros(len(props) * len(angles), dtype=np.float32)
        out_list.append(torch.from_numpy(feats))
    return torch.stack(out_list, dim=0).to(device)  # B x (6*len(angles))

def compute_glrlm_features(x_tensor, levels=8):
    """Compute GLRLM-based features using horizontal + vertical runs.
    Produces 5 features per image: SRE, LRE, GLN, RLN, RP (approximations).
    """
    B = x_tensor.size(0)
    device = x_tensor.device
    feats_all = []
    for i in range(B):
        img = (x_tensor[i].squeeze().cpu().numpy() * (levels-1)).astype(np.uint8)
        H, W = img.shape
        max_run = max(H, W)
        # helper: compute run-length matrix for given axis
        def runs_in_direction(arr, axis):
            if axis == 0:
                arr_proc = arr.T
            else:
                arr_proc = arr
            R = np.zeros((levels, max_run+1), dtype=np.float32)
            total = 0
            for row in arr_proc:
                curr = row[0]
                run = 1
                for pix in row[1:]:
                    if pix == curr:
                        run += 1
                    else:
                        R[curr, run] += 1
                        total += 1
                        curr = pix
                        run = 1
                R[curr, run] += 1
                total += 1
            return R, total
        R_h, th = runs_in_direction(img, axis=1)
        R_v, tv = runs_in_direction(img, axis=0)
        R = R_h + R_v
        total_runs = th + tv
        if total_runs == 0:
            feats_all.append(torch.zeros(5, dtype=torch.float32)); continue
        runs = np.arange(max_run+1, dtype=np.float32)
        sre = (R / (runs[None, :]**2 + 1e-8)).sum() / total_runs
        lre = (R * (runs[None, :]**2)).sum() / total_runs
        gln = (R.sum(axis=1)**2).sum() / (R.sum() + 1e-8)
        rln = (R.sum(axis=0)**2).sum() / (R.sum() + 1e-8)
        rp = total_runs / (H * W + 1e-8)
        feats = np.array([sre, lre, gln, rln, rp], dtype=np.float32)
        feats_all.append(torch.from_numpy(feats))
    return torch.stack(feats_all, dim=0).to(device)  # B x 5

# =========================================================
# Policy + Value Networks (updated base_stats size)
# =========================================================
class PolicyNetwork(nn.Module):
    def __init__(self, hidden=128, history_len=4, lbp_P=8, clahe_hist_bins=32, glcm_levels=8, glcm_angles=4):
        super().__init__()
        # encode downsampled image + seg (2 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # sizes for handcrafted stats
        hist_bins = 32
        lbp_bins = lbp_P + 2
        glcm_props = 6
        glcm_angles = glcm_angles
        glcm_size = glcm_props * glcm_angles
        glrlm_size = 5
        # base stats: hist(32) + entropy(1) + lbp(lbp_bins) + glcm(glcm_size) + glrlm(5)
        base_stats_size = hist_bins + 1 + lbp_bins + glcm_size + glrlm_size

        self.history_len = history_len
        history_size = history_len * 2  # two action dims (clahe, gamma) per timestep

        stats_input_size = base_stats_size + history_size

        # new stats fc to accept extended features
        self.stats_fc = nn.Sequential(
            nn.Linear(stats_input_size, hidden // 2), nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + hidden // 2, hidden), nn.ReLU()
        )
        # still output 2 dims (clahe, gamma) — during phase1 we will optionally use only first dim
        self.mean_head = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        # history buffer registered as buffer so it moves to device with model
        # shape: history_len x 2 (stored on model device)
        self.register_buffer('history_buffer', torch.zeros(self.history_len, 2, dtype=torch.float32))

    def forward(self, obs, stats):
        B = obs.size(0)
        feat = self.encoder(obs).view(B, -1)
        history = self.history_buffer.view(1, -1).expand(B, -1)  # B x (history_len*2)
        stats_combined = torch.cat([stats, history.to(stats.device)], dim=1)
        feat_stats = self.stats_fc(stats_combined)
        h = self.fusion(torch.cat([feat, feat_stats], dim=1))
        mean = self.mean_head(h)
        log_std_clamped = torch.clamp(self.log_std, -6.0, 1.0)
        std = torch.exp(log_std_clamped).unsqueeze(0).expand_as(mean)
        return mean, std

    def update_history(self, actions):
        """actions: either (B,2) real-valued scaled actions or (2,) single vector.
           We store the batch-mean real actions in the ring buffer (as real values).
        """
        if actions is None:
            return
        if actions.dim() == 2:
            a = actions.mean(dim=0).detach().cpu()
        else:
            a = actions.detach().cpu()
        hb_cpu = self.history_buffer.detach().cpu()
        hb_cpu = torch.cat([hb_cpu[1:], a.view(1, 2)], dim=0)
        self.history_buffer.copy_(hb_cpu.to(self.history_buffer.device))

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 1)
    def forward(self, obs):
        feat = self.encoder(obs).view(obs.size(0), -1)
        return self.fc(feat).squeeze(1)

# =========================================================
# Training (Phase 1 + Phase 2) with CLAHE-only Phase1 option
# =========================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.logdir)

    ds = BreastSegDataset(args.csv, resize=(args.img_size, args.img_size), augment=True)
    val_len, train_len = int(len(ds) * 0.2), len(ds) - int(len(ds) * 0.2)
    tr, vl = random_split(ds, [train_len, val_len])
    tr_loader = DataLoader(tr, batch_size=args.batch_size,
                           sampler=WeightedRandomSampler([1.0] * len(tr), len(tr), True),
                           num_workers=4, pin_memory=True)

    seg = ACAAtrousResUNet(in_ch=1, out_ch=1).to(device)
    if os.path.exists("checkpoints/best.pth"):
        seg.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))
    seg.eval()
    [setattr(p, "requires_grad", False) for p in seg.parameters()]

    # set realistic action ranges here:
    clahe_min, clahe_max = 0.5, 4.0
    gamma_min, gamma_max = 0.7, 2.5
    mid_gamma = (gamma_min + gamma_max) / 2.0

    policy = PolicyNetwork(hidden=128, history_len=args.history_len, lbp_P=args.lbp_p,
                           clahe_hist_bins=args.clahe_bins, glcm_levels=args.glcm_levels).to(device)
    critic = ValueNetwork().to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.policy_lr)
    seg_opt = torch.optim.SGD(seg.parameters(), lr=args.seg_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    crit = DiceBCELoss()

    def compute_histograms(x_tensor, bins=32, vmin=0.0, vmax=1.0):
        B = x_tensor.size(0)
        hist_list = []
        for i in range(B):
            h = torch.histc(x_tensor[i].view(-1), bins=bins, min=vmin, max=vmax)
            hist_list.append(h)
        hist = torch.stack(hist_list, dim=0)
        return hist

    # ----------------- Phase 1 -----------------
    print("== Phase 1: Policy only (Actor-Critic) ==")
    for ep in range(args.policy_epochs):
        policy.train(); critic.train()
        tot_loss = tot_reward = 0
        for step, (x, y) in enumerate(tqdm(tr_loader, desc=f"PolE{ep}")):
            x, y = x.to(device), y.to(device)
            B = x.shape[0]

            x_denorm = (x * 0.5 + 0.5).clamp(0, 1)
            x_uint8 = (x_denorm.squeeze(1).cpu().numpy() * 255).astype(np.uint8)

            # Downsample + seg feedback from frozen seg
            with torch.no_grad():
                seg_probs = torch.sigmoid(seg(x))

            obs = torch.cat([
                F.interpolate(x_denorm, size=(64,64)),
                F.interpolate(seg_probs, size=(64,64))
            ], dim=1)  # B x 2 x 64 x 64

            # --- New handcrafted stats: hist + entropy + LBP + GLCM + GLRLM ---
            hist = compute_histograms(x_denorm, bins=32, vmin=0.0, vmax=1.0).to(x.device)  # B x 32
            hist = hist / (hist.sum(1, keepdim=True) + 1e-8)
            entropy = -(hist * (hist + 1e-8).log()).sum(1, keepdim=True)  # B x 1

            lbp = compute_lbp(x_denorm, P=args.lbp_p, R=1)  # B x (P+2)

            glcm = compute_glcm_features(x_denorm, levels=args.glcm_levels)  # B x (6*angles)
            glrlm = compute_glrlm_features(x_denorm, levels=args.glcm_levels)  # B x 5

            stats = torch.cat([hist, entropy, lbp, glcm, glrlm], dim=1).float().to(device)

            # Policy + value
            mean, std = policy(obs, stats)

            # If requested: Phase1 trains only CLAHE (gamma fixed).
            if args.phase1_clahe_only:
                mean_clahe = mean[:, [0]]  # B x 1
                std_clahe = std[:, [0]]
                dist = torch.distributions.Normal(mean_clahe, std_clahe)
                z = dist.rsample()  # B x 1
                logp = dist.log_prob(z).sum(1)
                entropy_bonus = dist.entropy().sum(1).mean()
                z_sig = torch.sigmoid(z)  # in (0,1)

                # scaled real actions
                clahe = clahe_min + z_sig[:,0] * (clahe_max - clahe_min)  # B
                gamma = torch.full((B,), mid_gamma, device=device)     # fixed midpoint gamma

                # update policy history with real-valued scaled actions
                actions_real = torch.stack([clahe, gamma], dim=1)  # B x 2
                policy.update_history(actions_real.detach())

            else:
                dist = torch.distributions.Normal(mean, std)
                z = dist.rsample()  # B x 2
                logp = dist.log_prob(z).sum(1)
                entropy_bonus = dist.entropy().sum(1).mean()
                z_sig = torch.sigmoid(z)
                # scale to real world ranges
                clahe = clahe_min + z_sig[:,0] * (clahe_max - clahe_min)
                gamma = gamma_min + z_sig[:,1] * (gamma_max - gamma_min)
                # update policy history with real-valued scaled actions
                actions_real = torch.stack([clahe, gamma], dim=1)
                policy.update_history(actions_real.detach())

            # apply adjustments
            adj = apply_clahe_gamma_np(x_uint8, clahe.detach().cpu().numpy(), gamma.detach().cpu().numpy())
            adj_t = torch.tensor(adj, dtype=torch.float32).unsqueeze(1).to(device)
            adj_t = (adj_t - 0.5) / 0.5  # normalized to [-1,1]

            with torch.no_grad():
                seg_p = torch.sigmoid(seg(adj_t))

            # per-sample BCE
            bce_per_sample = F.binary_cross_entropy(seg_p, y, reduction='none').mean((1,2,3))
            dice_vals = dice_per_sample(seg_p, y)  # B

            # compute dullness penalty
            adj_denorm = (adj_t * 0.5 + 0.5).squeeze(1)  # B x H x W in [0,1]
            brightness = adj_denorm.view(B, -1).mean(1)   # B
            contrast = adj_denorm.view(B, -1).std(1)     # B

            if args.auto_dull_thresholds:
                # compute batch-level threshold from original input brightness mean
                batch_orig_brightness_mean = x_denorm.view(B, -1).mean(1).mean()  # scalar
                brightness_thresh_eff = batch_orig_brightness_mean * args.brightness_target_factor
            else:
                brightness_thresh_eff = torch.tensor(args.brightness_threshold, device=device)

            contrast_thresh_eff = torch.tensor(args.contrast_threshold, device=device)

            bright_deficit = torch.clamp(brightness_thresh_eff - brightness, min=0.0)
            contrast_deficit = torch.clamp(contrast_thresh_eff - contrast, min=0.0)
            dull_penalty = args.brightness_penalty_weight * bright_deficit + args.contrast_penalty_weight * contrast_deficit
            # ensure same device/type
            dull_penalty = dull_penalty.to(device)

            # reward shaping: dice - 0.1 * bce - dull_penalty
            rewards = dice_vals - 0.1 * bce_per_sample.detach() - dull_penalty.detach()

            # normalize rewards across batch
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = rewards.to(device)

            value = critic(obs)
            adv = rewards - value.detach()
            actor_loss = -(logp * adv).mean()
            critic_loss = F.mse_loss(value, rewards)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            policy_opt.zero_grad(); critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            policy_opt.step(); critic_opt.step()

            # logging (keep minimal images to avoid IO bottleneck)
            if step == 0:
                writer.add_image("phase1/input_image_0", x_denorm[0].detach().cpu(), global_step=ep*len(tr_loader)+step, dataformats='CHW')
                writer.add_image("phase1/adjusted_image_0", ((adj_t * 0.5 + 0.5)[0]).detach().cpu(), global_step=ep*len(tr_loader)+step, dataformats='CHW')
                writer.add_images("phase1/policy_output_seg", seg_p.detach().cpu(), ep)

                writer.add_scalar("phase1/mean_clahe", mean[:,0].mean().item(), ep)
                writer.add_scalar("phase1/std_clahe", std[:,0].mean().item(), ep)
                writer.add_scalar("phase1/entropy_bonus", entropy_bonus.item(), ep)
                writer.add_scalar("phase1/hist_entropy_mean", entropy.mean().item(), ep)
                writer.add_scalar("phase1/brightness_mean", brightness.mean().item(), ep)
                writer.add_scalar("phase1/contrast_mean", contrast.mean().item(), ep)
                writer.add_scalar("phase1/dull_penalty_mean", dull_penalty.mean().item(), ep)
                # log history mean (real values)
                writer.add_scalar("phase1/policy_history_mean", policy.history_buffer.mean().item(), ep)

            tot_loss += loss.item(); tot_reward += rewards.mean().item()
        print(f"[P1 Ep{ep}] Loss={tot_loss/len(tr_loader):.4f} Reward={tot_reward/len(tr_loader):.4f}")

    torch.save(policy.state_dict(), os.path.join(args.outdir,"policy_phase1.pth"))

    # ----------------- Phase 2 (joint) -----------------
    print("== Phase 2: Joint training ==")
    [setattr(p, "requires_grad", True) for p in seg.parameters()]
    for ep in range(args.joint_epochs):
        seg.train(); policy.train(); critic.train()
        seg_loss_tot = pol_loss_tot = 0
        for step, (x, y) in enumerate(tqdm(tr_loader, desc=f"JointE{ep}")):
            x, y = x.to(device), y.to(device)
            B = x.shape[0]

            x_denorm = (x * 0.5 + 0.5).clamp(0, 1)
            x_uint8 = (x_denorm.squeeze(1).cpu().numpy() * 255).astype(np.uint8)

            with torch.no_grad():
                seg_probs = torch.sigmoid(seg(x))

            obs = torch.cat([
                F.interpolate(x_denorm, size=(64,64)),
                F.interpolate(seg_probs, size=(64,64))
            ], dim=1)

            hist = compute_histograms(x_denorm, bins=32, vmin=0.0, vmax=1.0).to(x.device)
            hist = hist / (hist.sum(1, keepdim=True) + 1e-8)
            entropy = -(hist * (hist + 1e-8).log()).sum(1, keepdim=True)
            lbp = compute_lbp(x_denorm, P=args.lbp_p, R=1)
            glcm = compute_glcm_features(x_denorm, levels=args.glcm_levels)
            glrlm = compute_glrlm_features(x_denorm, levels=args.glcm_levels)
            stats = torch.cat([hist, entropy, lbp, glcm, glrlm], dim=1).float().to(device)

            mean, std = policy(obs, stats)
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
            logp = dist.log_prob(z).sum(1)
            entropy_bonus = dist.entropy().sum(1).mean()
            z_sig = torch.sigmoid(z)
            # scale to real world ranges and update history with real-valued actions
            clahe = clahe_min + z_sig[:,0] * (clahe_max - clahe_min)
            gamma = gamma_min + z_sig[:,1] * (gamma_max - gamma_min)
            actions_real = torch.stack([clahe, gamma], dim=1)
            policy.update_history(actions_real.detach())

            adj = apply_clahe_gamma_np(x_uint8, clahe.cpu().numpy(), gamma.cpu().numpy())
            adj_t = torch.tensor(adj).unsqueeze(1).to(device)
            adj_t = (adj_t - 0.5) / 0.5

            seg_logits = seg(adj_t)
            seg_loss = crit(seg_logits, y)
            seg_p = torch.sigmoid(seg_logits)

            # per-sample BCE
            bce_per_sample = F.binary_cross_entropy(seg_p, y, reduction='none').mean((1,2,3))
            dice_vals = dice_per_sample(seg_p, y)

            # compute dullness penalty
            adj_denorm = (adj_t * 0.5 + 0.5).squeeze(1)  # B x H x W in [0,1]
            brightness = adj_denorm.view(B, -1).mean(1)   # B
            contrast = adj_denorm.view(B, -1).std(1)     # B

            if args.auto_dull_thresholds:
                batch_orig_brightness_mean = x_denorm.view(B, -1).mean(1).mean()
                brightness_thresh_eff = batch_orig_brightness_mean * args.brightness_target_factor
            else:
                brightness_thresh_eff = torch.tensor(args.brightness_threshold, device=device)

            contrast_thresh_eff = torch.tensor(args.contrast_threshold, device=device)

            bright_deficit = torch.clamp(brightness_thresh_eff - brightness, min=0.0)
            contrast_deficit = torch.clamp(contrast_thresh_eff - contrast, min=0.0)
            dull_penalty = args.brightness_penalty_weight * bright_deficit + args.contrast_penalty_weight * contrast_deficit
            dull_penalty = dull_penalty.to(device)

            rewards = dice_vals - 0.1 * bce_per_sample.detach() - dull_penalty.detach()
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = rewards.to(device)

            value = critic(obs)
            adv = rewards - value.detach()
            pol_loss = -(logp * adv).mean()
            critic_loss = F.mse_loss(value, rewards)
            loss = seg_loss + pol_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            seg_opt.zero_grad(); policy_opt.zero_grad(); critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            seg_opt.step(); policy_opt.step(); critic_opt.step()

            if step == 0:
                writer.add_images("phase2/input_images", x_denorm, ep)
                writer.add_images("phase2/adjusted_images", (adj_t * 0.5 + 0.5), ep)
                writer.add_images("phase2/seg_output", seg_p, ep)
                writer.add_scalar("phase2/entropy_bonus", entropy_bonus.item(), ep)
                writer.add_scalar("phase2/mean_clahe", mean[:,0].mean().item(), ep)
                writer.add_scalar("phase2/mean_gamma", mean[:,1].mean().item(), ep)
                writer.add_scalar("phase2/hist_entropy_mean", entropy.mean().item(), ep)
                writer.add_scalar("phase2/clahe_glcm_mean", glcm.mean().item(), ep)
                writer.add_scalar("phase2/glrlm_mean", glrlm.mean().item(), ep)
                writer.add_scalar("phase2/policy_history_mean", policy.history_buffer.mean().item(), ep)
                writer.add_scalar("phase2/brightness_mean", brightness.mean().item(), ep)
                writer.add_scalar("phase2/contrast_mean", contrast.mean().item(), ep)
                writer.add_scalar("phase2/dull_penalty_mean", dull_penalty.mean().item(), ep)

            seg_loss_tot += seg_loss.item(); pol_loss_tot += pol_loss.item()
        print(f"[P2 Ep{ep}] SegLoss={seg_loss_tot/len(tr_loader):.4f} PolLoss={pol_loss_tot/len(tr_loader):.4f}")

    torch.save(seg.state_dict(), os.path.join(args.outdir,"seg_joint.pth"))
    torch.save(policy.state_dict(), os.path.join(args.outdir,"policy_joint.pth"))
    torch.save(critic.state_dict(), os.path.join(args.outdir,"critic_joint.pth"))
    writer.close()

# =========================================================
# Main
# =========================================================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=str,required=True)
    p.add_argument("--outdir",type=str,default="checkpoints-rl")
    p.add_argument("--logdir",type=str,default="runs/aca_rl")
    p.add_argument("--img-size",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--policy-lr",type=float,default=1e-4)
    p.add_argument("--policy-epochs",type=int,default=100)
    p.add_argument("--seg-lr",type=float,default=1e-5)
    p.add_argument("--joint-epochs",type=int,default=35)
    p.add_argument("--history-len",type=int,default=4)
    p.add_argument("--lbp-p",type=int,default=8)
    p.add_argument("--clahe-bins",type=int,default=32)
    p.add_argument("--glcm-levels",type=int,default=16)
    p.add_argument("--phase1-clahe-only", action='store_true', help='Train Phase1 with CLAHE only (gamma fixed)')
    # new dullness reward args:
    p.add_argument("--brightness-threshold", type=float, default=0.25, help="Brightness threshold (0-1) below which penalty applies")
    p.add_argument("--contrast-threshold", type=float, default=0.05, help="Contrast (std) threshold below which penalty applies")
    p.add_argument("--brightness-penalty-weight", type=float, default=0.2, help="Weight of brightness deficit penalty")
    p.add_argument("--contrast-penalty-weight", type=float, default=0.5, help="Weight of contrast deficit penalty")
    p.add_argument("--auto-dull-thresholds", action='store_true', help="Auto compute brightness threshold from batch original images to avoid manual tuning")
    p.add_argument("--brightness-target-factor", type=float, default=0.7, help="When auto thresholds enabled, multiply original mean brightness by this factor for target")
    args=p.parse_args(); os.makedirs(args.outdir,exist_ok=True); train(args)
