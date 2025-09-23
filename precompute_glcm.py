#!/usr/bin/env python3
"""
precompute_glcm.py (optimized)

Precompute GLCM contrast heatmaps (or fast texture proxies) for mammograms.

Usage examples:
  # Fast proxy (recommended for quick experimentation)
  python precompute_glcm.py --csv dataset.csv --outdir glcm_npy --method sobel --num-workers 8

  # Exact GLCM (slow) with fewer workers and larger step to save time
  python precompute_glcm.py --csv dataset.csv --outdir glcm_npy --method glcm --num-workers 4 --step 12

Outputs:
 - per-image .npy heatmaps (float32, 0..1)
 - new CSV with `glcm_file_path` column (default: dataset_with_glcm.csv)
"""
from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops
import concurrent.futures
import traceback
from typing import Tuple

# ----------------------------
# Pectoral removal (component-based)
# ----------------------------
def remove_pectoral_by_cc(img_uint8: np.ndarray, min_area_frac: float = 0.001) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Detect and remove pectoral muscle if present.
    Returns:
        clean (np.ndarray): image with pectoral removed (if detected)
        pectoral_mask (np.ndarray): binary mask (255 = muscle)
        removed (bool): True if muscle was removed, False otherwise
    """
    H, W = img_uint8.shape
    img = cv2.equalizeHist(img_uint8)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    num_labels, labels = cv2.connectedComponents(bw)
    top_strip = img[:max(1, H // 8), :]
    left_sum = int(top_strip[:, :max(1, W // 3)].sum())
    right_sum = int(top_strip[:, -max(1, W // 3):].sum())
    corner_side = "left" if left_sum > right_sum else "right"

    best_label, best_area = None, 0
    for lab in range(1, num_labels):
        mask = (labels == lab).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area_frac * H * W:
            continue
        if corner_side == "left":
            touches = mask[: H // 20, : W // 20].sum() > 0
        else:
            touches = mask[: H // 20, -W // 20 :].sum() > 0
        if touches and area > best_area:
            best_area, best_label = area, lab

    clean = img_uint8.copy()
    pectoral_mask = np.zeros_like(img_uint8, dtype=np.uint8)
    if best_label is not None:
        mask = (labels == best_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(hull_mask, hull, 255)
            pectoral_mask = hull_mask
            clean[hull_mask > 0] = 0
            return clean, pectoral_mask, True  # ✅ muscle removed

    return img_uint8, pectoral_mask, False  # ✅ unchanged if no muscle

# ----------------------------
# Exact (slow) GLCM contrast map (keeps original semantics)
# ----------------------------
def glcm_contrast_map_exact(img_uint8: np.ndarray, win: int = 31, levels: int = 8, step: int = 8) -> np.ndarray:
    H, W = img_uint8.shape
    half = win // 2
    if win < 3:
        win = 3
        half = 1
    q = (img_uint8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
    heat = np.zeros((H, W), dtype=np.float32)
    ys = range(half, H - half, step)
    xs = range(half, W - half, step)
    for y in ys:
        for x in xs:
            patch = q[y - half : y + half + 1, x - half : x + half + 1]
            try:
                glcm = graycomatrix(patch, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
                contrast = graycoprops(glcm, "contrast")[0, 0]
            except Exception:
                contrast = 0.0
            heat[y, x] = float(contrast)
    heat = gaussian_filter(heat, sigma=max(1.0, step / 2.0))
    mn, mx = float(heat.min()), float(heat.max())
    if mx - mn < 1e-8:
        return np.zeros((H, W), dtype=np.float32)
    heat = (heat - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

# ----------------------------
# Fast proxies for texture (much faster)
# ----------------------------
def glcm_contrast_map_sobel(img_uint8: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Fast edge-based proxy using Sobel magnitude + Gaussian smoothing."""
    # Sobel produces float images; convert to float32
    gx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = gaussian_filter(mag, sigma=sigma)
    mn, mx = float(mag.min()), float(mag.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    heat = (mag - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

def glcm_contrast_map_local_var(img_uint8: np.ndarray, win: int = 31) -> np.ndarray:
    """Fast local variance proxy using box filters (integral filter style)."""
    img_f = img_uint8.astype(np.float32)
    k = max(3, win // 2 * 2 + 1)
    # mean and mean of squares via boxFilter (fast)
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(k, k), normalize=True)
    mean_sq = cv2.boxFilter(img_f * img_f, ddepth=-1, ksize=(k, k), normalize=True)
    var = mean_sq - mean * mean
    var = np.clip(var, a_min=0.0, a_max=None)
    var = gaussian_filter(var, sigma=max(1.0, k / 6.0))
    mn, mx = float(var.min()), float(var.max())
    if mx - mn < 1e-8:
        return np.zeros_like(var, dtype=np.float32)
    heat = (var - mn) / (mx - mn + 1e-8)
    return heat.astype(np.float32)

# ----------------------------
# Worker: process one image (called in a process pool)
# ----------------------------
def process_one(args_tuple) -> Tuple[int, str]:
    """
    args_tuple: (i, img_path, outdir, cfg)
    returns (i, saved_path) or (i, "")
    """
    i, img_path, outdir, cfg = args_tuple
    try:
        if not os.path.exists(img_path):
            print(f"[WARN] missing image: {img_path}")
            return i, ""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] failed to read: {img_path}")
            return i, ""
        # optional pectoral removal
        if cfg["remove_pectoral"]:
            img_clean, _, removed = remove_pectoral_by_cc(img, min_area_frac=cfg["min_area_frac"])
            if not removed:
                # No muscle detected → keep original
                img_clean = img
        else:
            img_clean = img

        method = cfg["method"]
        if method == "glcm":
            heat = glcm_contrast_map_exact(img_clean, win=cfg["win"], levels=cfg["levels"], step=cfg["step"])
        elif method == "sobel":
            heat = glcm_contrast_map_sobel(img_clean, sigma=cfg.get("sobel_sigma", 3.0))
        elif method == "local_var":
            heat = glcm_contrast_map_local_var(img_clean, win=cfg.get("local_var_win", cfg["win"]))
        else:
            raise ValueError(f"Unknown method: {method}")
        fname = os.path.join(outdir, f"glcm_{i}.npy")
        np.save(fname, heat.astype(np.float32))
        return i, fname
    except Exception:
        traceback.print_exc()
        return i, ""

# ----------------------------
# Main entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input dataset CSV")
    parser.add_argument("--outdir", required=True, help="Output folder for npy heatmaps")
    parser.add_argument("--outcsv", default="dataset_with_glcm.csv", help="Path to new CSV with GLCM column")
    parser.add_argument("--win", type=int, default=31)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--method", choices=["glcm", "sobel", "local_var"], default="sobel",
                        help="Computation method. use 'sobel' or 'local_var' for fast proxies.")
    parser.add_argument("--remove-pectoral", action="store_true")
    parser.add_argument("--min-area-frac", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sobel-sigma", type=float, default=3.0)
    parser.add_argument("--local-var-win", type=int, default=31)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    n = len(df)
    out_paths = [""] * n

    # prepare list of tasks (skip existing files)
    tasks = []
    cfg = dict(
        win=args.win,
        levels=args.levels,
        step=args.step,
        method=args.method,
        remove_pectoral=args.remove_pectoral,
        min_area_frac=args.min_area_frac,
        sobel_sigma=args.sobel_sigma,
        local_var_win=args.local_var_win,
    )

    for i, row in df.iterrows():
        out_path = os.path.join(args.outdir, f"glcm_{i}.npy")
        if os.path.exists(out_path):
            out_paths[i] = out_path
            continue
        tasks.append((i, row["image_file_path"], args.outdir, cfg))

    if len(tasks) == 0:
        print("All heatmaps already exist. Writing CSV and exiting.")
        df["glcm_file_path"] = out_paths
        df.to_csv(args.outcsv, index=False)
        return

    # run in parallel
    print(f"Computing {len(tasks)} heatmaps with method={args.method} using {args.num_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        futures = {exe.submit(process_one, t): t[0] for t in tasks}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            i = futures[fut]
            try:
                idx, saved = fut.result()
                if saved:
                    out_paths[idx] = saved
                else:
                    out_paths[idx] = ""
            except Exception:
                out_paths[i] = ""
                traceback.print_exc()

    df["glcm_file_path"] = out_paths
    df.to_csv(args.outcsv, index=False)
    print(f"Done. Saved new CSV to {args.outcsv}")

if __name__ == "__main__":
    main()
