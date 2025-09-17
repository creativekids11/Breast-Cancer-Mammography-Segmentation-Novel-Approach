#!/usr/bin/env python3
# segmentation_model.py
"""
Segmentation with ACA-Atrous UNet + Actor-Critic Policy
-------------------------------------------------------
Two-phase training:
  Phase 1 → Policy only (segmentation frozen as oracle)
  Phase 2 → Policy + segmentation model jointly
"""

import argparse, os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern  # requires scikit-image installed

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
# ACA block, UNet, ASPP, etc. (UNCHANGED)
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
# -------------------------
def compute_lbp(x_tensor, P=8, R=1):
    """
    x_tensor: B x 1 x H x W, values in [0,1]
    Returns: B x (P+2) LBP uniform histogram (normalized)
    """
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

def compute_contrast(x_tensor):
    """
    Simple contrast = std-dev of intensity per image.
    x_tensor: B x 1 x H x W (0..1)
    Returns: B x 1 tensor
    """
    B = x_tensor.size(0)
    device = x_tensor.device
    contrasts = []
    for i in range(B):
        img = x_tensor[i].squeeze().cpu().numpy()
        c = np.std(img).astype(np.float32)
        contrasts.append(torch.tensor([c], dtype=torch.float32))
    return torch.stack(contrasts, dim=0).to(device)  # B x 1

def compute_clahe_hist(x_tensor, bins=32, clipLimit=2.0, tileGridSize=(8,8)):
    """
    Compute histogram of CLAHE-applied image (uint8).
    x_tensor: B x 1 x H x W (0..1)
    Returns: B x bins normalized histograms (float)
    """
    B = x_tensor.size(0)
    device = x_tensor.device
    hist_list = []
    for i in range(B):
        img = (x_tensor[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        clahe_img = clahe.apply(img)
        hist = np.histogram(clahe_img, bins=bins, range=(0, 256))[0].astype(np.float32)
        hist = torch.from_numpy(hist)
        hist = hist / (hist.sum() + 1e-8)
        hist_list.append(hist)
    return torch.stack(hist_list, dim=0).to(device)  # B x bins

# =========================================================
# Policy + Value Networks
# =========================================================
class PolicyNetwork(nn.Module):
    def __init__(self, hidden=128, history_len=4, lbp_P=8, clahe_hist_bins=32):
        """
        history_len: how many previous actions to keep (each action = 2 dims)
        lbp_P: LBP P parameter (used only to compute feature size)
        clahe_hist_bins: number of CLAHE histogram bins used in stats
        """
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
        clahe_bins = clahe_hist_bins
        # base stats: hist(32) + entropy(1) + edge(1) + lbp(lbp_bins) + contrast(1) + clahe_hist(clahe_bins)
        base_stats_size = hist_bins + 1 + 1 + lbp_bins + 1 + clahe_bins

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
        self.mean_head = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        # history buffer registered as buffer so it moves to device with model
        # shape: history_len x 2 (stored on model device)
        self.register_buffer('history_buffer', torch.zeros(self.history_len, 2, dtype=torch.float32))

    def forward(self, obs, stats):
        """
        obs: B x 2 x 64 x 64
        stats: B x base_stats_size (without history)
        We append the history buffer (flattened) expanded across batch.
        """
        B = obs.size(0)
        feat = self.encoder(obs).view(B, -1)

        # flatten and expand history across batch dimension
        history = self.history_buffer.view(1, -1).expand(B, -1)  # B x (history_len*2)
        stats_combined = torch.cat([stats, history.to(stats.device)], dim=1)

        feat_stats = self.stats_fc(stats_combined)
        h = self.fusion(torch.cat([feat, feat_stats], dim=1))
        mean = self.mean_head(h)

        # clamp log_std for stable exploration
        log_std_clamped = torch.clamp(self.log_std, -6.0, 1.0)
        std = torch.exp(log_std_clamped).unsqueeze(0).expand_as(mean)
        return mean, std

    def update_history(self, actions):
        """
        Update history buffer with new action(s).
        actions: tensor of shape (B,2) or (2,) — we take the mean across batch so that
                 history buffer stores a single recent action vector per timestep.
        """
        if actions is None:
            return
        # ensure we work on CPU for manipulation, then copy to buffer device
        if actions.dim() == 2:
            a = actions.mean(dim=0).detach().cpu()
        else:
            a = actions.detach().cpu()

        hb_cpu = self.history_buffer.detach().cpu()
        hb_cpu = torch.cat([hb_cpu[1:], a.view(1, 2)], dim=0)
        # copy back to buffer's device
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
# Training (Phase 1 + Phase 2)
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

    policy = PolicyNetwork(hidden=128, history_len=args.history_len, lbp_P=args.lbp_p, clahe_hist_bins=args.clahe_bins).to(device)
    critic = ValueNetwork().to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.policy_lr)
    seg_opt = torch.optim.SGD(seg.parameters(), lr=args.seg_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    crit = DiceBCELoss()
    clahe_min, clahe_max, gamma_min, gamma_max = 0.1, 10.0, 0.1, 5.0

    # helper to compute per-sample histogram (bins) on tensor of shape Bx1xHxW
    def compute_histograms(x_tensor, bins=32, vmin=0.0, vmax=1.0):
        B = x_tensor.size(0)
        hist_list = []
        # compute per-sample histogram
        for i in range(B):
            # torch.histc may operate on CPU/GPU depending on tensor device and version.
            h = torch.histc(x_tensor[i].view(-1), bins=bins, min=vmin, max=vmax)
            hist_list.append(h)
        hist = torch.stack(hist_list, dim=0)  # B x bins
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

            # --- Extended handcrafted stats ---
            hist = compute_histograms(x_denorm, bins=32, vmin=0.0, vmax=1.0).to(x.device)  # B x 32
            hist = hist / (hist.sum(1, keepdim=True) + 1e-8)

            entropy = -(hist * (hist + 1e-8).log()).sum(1, keepdim=True)  # B x 1

            edge = torch.mean(torch.abs(x_denorm[:, :, :, 1:] - x_denorm[:, :, :, :-1]), dim=(1, 2, 3)).view(B, 1)  # B x 1

            lbp = compute_lbp(x_denorm, P=args.lbp_p, R=1)  # B x (P+2)

            contrast = compute_contrast(x_denorm)  # B x 1

            clahe_hist = compute_clahe_hist(x_denorm, bins=args.clahe_bins)  # B x clahe_bins

            stats = torch.cat([hist, entropy, edge, lbp, contrast, clahe_hist], dim=1).float().to(device)  # B x (base_stats_size)
            # policy.forward will append history internally

            # Policy + value
            mean, std = policy(obs, stats)
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
            logp = dist.log_prob(z).sum(1)
            entropy_bonus = dist.entropy().sum(1).mean()

            z_sig = torch.sigmoid(z)
            # update policy history with the action (batch-mean inside update_history)
            policy.update_history(z_sig.detach())

            clahe = clahe_min + z_sig[:,0] * (clahe_max - clahe_min)
            gamma = gamma_min + z_sig[:,1] * (gamma_max - gamma_min)
            adj = apply_clahe_gamma_np(x_uint8, clahe.detach().cpu().numpy(), gamma.detach().cpu().numpy())
            adj_t = torch.tensor(adj, dtype=torch.float32).unsqueeze(1).to(device)
            adj_t = (adj_t - 0.5) / 0.5

            with torch.no_grad():
                seg_p = torch.sigmoid(seg(adj_t))

            # shaped reward: weighted dice + (1 - normalized BCE) previously — here we keep the shaped reward
            bce_loss = F.binary_cross_entropy(seg_p, y)
            rewards = dice_per_sample(seg_p, y) * 2 - bce_loss.detach()

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

            # logging
            if step == 0:
                writer.add_images("phase1/input_images", x_denorm, ep)
                writer.add_images("phase1/adjusted_images", (adj_t * 0.5 + 0.5), ep)
                writer.add_images("phase1/policy_output_seg", seg_p, ep)
                # log action stats
                writer.add_scalar("phase1/mean_clahe", mean[:,0].mean().item(), ep)
                writer.add_scalar("phase1/mean_gamma", mean[:,1].mean().item(), ep)
                writer.add_scalar("phase1/std_clahe", std[:,0].mean().item(), ep)
                writer.add_scalar("phase1/std_gamma", std[:,1].mean().item(), ep)
                writer.add_scalar("phase1/entropy_bonus", entropy_bonus.item(), ep)
                # log additional stats
                writer.add_scalar("phase1/hist_entropy_mean", entropy.mean().item(), ep)
                writer.add_scalar("phase1/contrast_mean", contrast.mean().item(), ep)
                writer.add_scalar("phase1/edge_mean", edge.mean().item(), ep)
                # CLAHE histogram — log mean of bins (summary)
                writer.add_scalar("phase1/clahe_hist_mean", clahe_hist.mean().item(), ep)
                # log policy history mean
                hist_buffer_mean = policy.history_buffer.mean().item()
                writer.add_scalar("phase1/policy_history_mean", hist_buffer_mean, ep)

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

            # seg feedback (current seg on original input)
            with torch.no_grad():
                seg_probs = torch.sigmoid(seg(x))

            obs = torch.cat([
                F.interpolate(x_denorm, size=(64,64)),
                F.interpolate(seg_probs, size=(64,64))
            ], dim=1)

            # --- Extended handcrafted stats (same as Phase 1) ---
            hist = compute_histograms(x_denorm, bins=32, vmin=0.0, vmax=1.0).to(x.device)
            hist = hist / (hist.sum(1, keepdim=True) + 1e-8)
            entropy = -(hist * (hist + 1e-8).log()).sum(1, keepdim=True)
            edge = torch.mean(torch.abs(x_denorm[:, :, :, 1:] - x_denorm[:, :, :, :-1]), dim=(1, 2, 3)).view(B, 1)
            lbp = compute_lbp(x_denorm, P=args.lbp_p, R=1)
            contrast = compute_contrast(x_denorm)
            clahe_hist = compute_clahe_hist(x_denorm, bins=args.clahe_bins)
            stats = torch.cat([hist, entropy, edge, lbp, contrast, clahe_hist], dim=1).float().to(device)  # B x base_stats_size

            mean, std = policy(obs, stats)
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
            logp = dist.log_prob(z).sum(1)
            entropy_bonus = dist.entropy().sum(1).mean()

            z_sig = torch.sigmoid(z)
            policy.update_history(z_sig.detach())

            clahe = clahe_min + z_sig[:,0] * (clahe_max - clahe_min)
            gamma = gamma_min + z_sig[:,1] * (gamma_max - gamma_min)
            adj = apply_clahe_gamma_np(x_uint8, clahe.cpu().numpy(), gamma.cpu().numpy())
            adj_t = torch.tensor(adj).unsqueeze(1).to(device)
            adj_t = (adj_t - 0.5) / 0.5

            seg_logits = seg(adj_t)
            seg_loss = crit(seg_logits, y)
            seg_p = torch.sigmoid(seg_logits)

            bce_loss = F.binary_cross_entropy(seg_p, y)
            rewards = dice_per_sample(seg_p, y) * 2 - bce_loss.detach()
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
                # more logging
                writer.add_scalar("phase2/hist_entropy_mean", entropy.mean().item(), ep)
                writer.add_scalar("phase2/contrast_mean", contrast.mean().item(), ep)
                writer.add_scalar("phase2/edge_mean", edge.mean().item(), ep)
                writer.add_scalar("phase2/clahe_hist_mean", clahe_hist.mean().item(), ep)
                writer.add_scalar("phase2/policy_history_mean", policy.history_buffer.mean().item(), ep)

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
    p.add_argument("--policy-epochs",type=int,default=40)
    p.add_argument("--seg-lr",type=float,default=1e-5)
    p.add_argument("--joint-epochs",type=int,default=10)
    p.add_argument("--history-len",type=int,default=4)
    p.add_argument("--lbp-p",type=int,default=8)
    p.add_argument("--clahe-bins",type=int,default=32)
    args=p.parse_args(); os.makedirs(args.outdir,exist_ok=True); train(args)
