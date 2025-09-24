#!/usr/bin/env python3
"""
Unified dataset prep with texture overlay enhancement.

Adds options to enhance images by overlaying a texture map computed
by one of: 'local_var' (fast), 'sobel' (fast), or 'glcm' (exact, slow).

Example:
  python prepare_dataset.py --cbis-csv cbis.csv --mias-info-csv mias.csv \
      --mias-images-dir MIAS_IMAGES --normal-dir NORMAL_IMAGES \
      --inbreast-base-dir roboflow_inbreast --inbreast-yaml data.yaml \
      --output-csv unified.csv --outdir DATASET \
      --texture-method local_var --texture-strength 0.25 --texture-alpha 0.6
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import yaml
from typing import Tuple

# Optional imports for GLCM / smoothing
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_class_map(yaml_path: str):
    """Load class_id → name mapping from data.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data.get("names", []))}

# ---------------- Preprocessing ---------------- #
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Base preprocessing: normalize → median blur → CLAHE → normalize.
    Returns uint8 grayscale image.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

# ---------------- Texture computations ---------------- #
def glcm_contrast_map_exact(img_uint8: np.ndarray, win: int = 31, levels: int = 8, step: int = 8) -> np.ndarray:
    """
    Exact GLCM contrast heatmap (slow). Returns float32 0..1 same size as img.
    Use only for small experiments or small dataset.
    """
    H, W = img_uint8.shape
    half = win // 2
    if win < 3:
        win = 3; half = 1
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

def glcm_contrast_map_sobel(img_uint8: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Fast Sobel-based texture proxy (0..1)."""
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
    """Fast local variance proxy using box filters (0..1)."""
    img_f = img_uint8.astype(np.float32)
    k = max(3, win // 2 * 2 + 1)
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

def compute_texture_map(img_uint8: np.ndarray, method: str = "local_var",
                        win: int = 31, levels: int = 8, step: int = 8,
                        sobel_sigma: float = 3.0, local_var_win: int = 31) -> np.ndarray:
    """
    Returns float32 texture map in range [0,1] same size as img.
    method: 'local_var', 'sobel', 'glcm'
    """
    if method == "local_var":
        return glcm_contrast_map_local_var(img_uint8, win=local_var_win)
    elif method == "sobel":
        return glcm_contrast_map_sobel(img_uint8, sigma=sobel_sigma)
    elif method == "glcm":
        # slow
        return glcm_contrast_map_exact(img_uint8, win=win, levels=levels, step=step)
    else:
        raise ValueError(f"Unknown texture method: {method}")

# ---------------- Enhancement overlay ---------------- #
def enhance_with_texture(img_uint8: np.ndarray, texture: np.ndarray,
                         strength: float = 0.25, alpha: float = 0.6) -> np.ndarray:
    """
    Overlay texture onto image while avoiding overpowering.
    img_uint8: HxW grayscale uint8
    texture: HxW float32 in [0,1]
    strength: how much texture modifies pixel values (added amount)
    alpha: blending factor between original and textured overlay
    Returns uint8 image.
    Formula:
       overlay = clip(img + texture*strength*255)
       out = (1-alpha)*img + alpha*overlay
    This produces a modest, controllable enhancement.
    """
    if img_uint8.dtype != np.uint8:
        img = img_uint8.astype(np.uint8)
    else:
        img = img_uint8
    h, w = img.shape
    if texture.shape != (h, w):
        texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    # texture in 0..1
    overlay = img_f + (texture * strength)
    overlay = np.clip(overlay, 0.0, 1.0)
    out = (1.0 - alpha) * img_f + alpha * overlay
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

# ---------------- CBIS ---------------- #
def process_cbis(input_csv, mask_outdir, image_outdir, texture_cfg):
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    rows = []
    grouped = df.groupby(["patient_id", "image_file_path"], dropna=False)

    for (pid, img_path), group in grouped:
        base_row = group.iloc[0].to_dict()
        abnormality_ids = group["abnormality_id"].astype(str).unique().tolist()

        mask_paths = [mp for mp in group["roi_mask_file_path"].dropna().unique().tolist() if isinstance(mp, str)]
        merged_mask = None
        for mp in mask_paths:
            if not os.path.exists(mp):
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8) * 255
            merged_mask = mask if merged_mask is None else cv2.bitwise_or(merged_mask, mask)

        if os.path.exists(img_path):
            full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                continue
            processed_img = preprocess_image(full_img)
        else:
            continue

        if merged_mask is None:
            merged_mask = np.zeros_like(processed_img, dtype=np.uint8)

        # texture enhancement
        texture = compute_texture_map(processed_img, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(processed_img, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])

        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"

        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")

        cv2.imwrite(proc_img_path, enhanced)
        cv2.imwrite(mask_path, merged_mask)

        base_row["dataset"] = "CBIS"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)

    return pd.DataFrame(rows)

# ---------------- MIAS ---------------- #
def process_mias_from_csv(info_csv_path, images_dir, mask_outdir, image_outdir, texture_cfg):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    df_info = pd.read_csv(info_csv_path)
    rows = []
    for idx, row in df_info.iterrows():
        refnum = row["REFNUM"]
        img_filename = f"{refnum}.png"
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_proc = preprocess_image(img)
        is_normal = (row["CLASS"] == "NORM")
        mask = np.zeros_like(img_proc, dtype=np.uint8)
        if not is_normal and not pd.isna(row["X"]) and not pd.isna(row["Y"]) and not pd.isna(row["RADIUS"]):
            x, y, r = int(row["X"]), int(row["Y"]), int(row["RADIUS"])
            cv2.circle(mask, (x, y), r, 255, -1)

        # texture enhance
        texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(img_proc, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])

        pid = f"MIAS_{refnum}"
        proc_img_path = os.path.join(image_outdir, f"{pid}.png")
        mask_path = os.path.join(mask_outdir, f"{pid}_mask.png")
        cv2.imwrite(proc_img_path, enhanced)
        cv2.imwrite(mask_path, mask)

        pathology = "N" if is_normal else row["SEVERITY"]
        abnormality_id = "None" if is_normal else row["CLASS"]
        row_data = {
            "dataset": "MIAS",
            "patient_id": pid,
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path,
            "pathology": pathology,
            "abnormality_id": abnormality_id,
        }
        rows.append(row_data)
    return pd.DataFrame(rows)

# ---------------- NORMAL ---------------- #
def process_normal_images(normal_img_dir, mask_outdir, image_outdir, texture_cfg):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    rows = []
    normal_img_files = [f for f in os.listdir(normal_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in normal_img_files:
        img_path = os.path.join(normal_img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_proc = preprocess_image(img)
        # texture enhance
        texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                      win=texture_cfg["win"], levels=texture_cfg["levels"],
                                      step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                      local_var_win=texture_cfg["local_var_win"])
        enhanced = enhance_with_texture(img_proc, texture,
                                        strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])
        pid = os.path.splitext(img_file)[0]
        proc_img_path = os.path.join(image_outdir, f"NORMAL_{pid}.png")
        mask_path = os.path.join(mask_outdir, f"NORMAL_{pid}_mask.png")
        cv2.imwrite(proc_img_path, enhanced)
        mask = np.zeros_like(img_proc, dtype=np.uint8)
        cv2.imwrite(mask_path, mask)
        row = {
            "dataset": "NORMAL",
            "patient_id": f"NORMAL_{pid}",
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path,
            "pathology": "N",
            "abnormality_id": "None",
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- INBREAST (Roboflow YOLOv11) ---------------- #
def yolo_to_mask_with_classes(label_path: str, img_shape: Tuple[int,int], class_map):
    """Convert YOLOv11 labels to binary mask + class list."""
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    classes = []
    if not os.path.exists(label_path):
        return mask, classes
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id, x_c, y_c, w, h = map(float, parts[:5])
        cls_id = int(cls_id)
        cls_name = class_map.get(cls_id, f"class_{cls_id}")
        classes.append(cls_name)
        x_c, y_c, w, h = x_c * W, y_c * H, w * W, h * H
        x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
        x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
        # clip to image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask, list(dict.fromkeys(classes))  # preserve order & unique

def process_inbreast_roboflow_v11(base_dir, yaml_path, mask_outdir, image_outdir, texture_cfg):
    ensure_dir(mask_outdir); ensure_dir(image_outdir)
    class_map = load_class_map(yaml_path)
    rows = []
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Cannot read {img_path}")
                continue
            img_proc = preprocess_image(img)
            mask, class_list = yolo_to_mask_with_classes(label_path, img_proc.shape, class_map)
            # texture enhance
            texture = compute_texture_map(img_proc, method=texture_cfg["method"],
                                          win=texture_cfg["win"], levels=texture_cfg["levels"],
                                          step=texture_cfg["step"], sobel_sigma=texture_cfg["sobel_sigma"],
                                          local_var_win=texture_cfg["local_var_win"])
            enhanced = enhance_with_texture(img_proc, texture,
                                            strength=texture_cfg["strength"], alpha=texture_cfg["alpha"])
            pid = f"{split}_{os.path.splitext(img_file)[0]}"
            proc_img_path = os.path.join(image_outdir, f"INBREAST_{pid}.png")
            mask_path = os.path.join(mask_outdir, f"INBREAST_{pid}_mask.png")
            cv2.imwrite(proc_img_path, enhanced)
            cv2.imwrite(mask_path, mask)
            row = {
                "dataset": "INBREAST",
                "patient_id": f"INBREAST_{pid}",
                "image_file_path": proc_img_path,
                "roi_mask_file_path": mask_path,
                "pathology": "Abnormality" if mask.sum() > 0 else "N",
                "abnormality_id": ";".join(class_list) if len(class_list) > 0 else "None",
            }
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------- UNIFIED ---------------- #
def prepare_unified(cbis_csv, mias_info_csv, mias_images_dir, normal_dir,
                    inbreast_base_dir, inbreast_yaml, output_csv, outdir,
                    texture_cfg):
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES"); cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mias_img_dir = os.path.join(outdir, "MIAS_IMAGES"); mias_mask_dir = os.path.join(outdir, "MIAS_MASKS")
    normal_img_dir = os.path.join(outdir, "NORMAL_IMAGES"); normal_mask_dir = os.path.join(outdir, "NORMAL_MASKS")
    inbreast_img_outdir = os.path.join(outdir, "INBREAST_IMAGES"); inbreast_mask_outdir = os.path.join(outdir, "INBREAST_MASKS")

    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir, texture_cfg)
    mias_df = process_mias_from_csv(mias_info_csv, mias_images_dir, mias_mask_dir, mias_img_dir, texture_cfg)
    normal_df = process_normal_images(normal_dir, normal_mask_dir, normal_img_dir, texture_cfg)
    inbreast_df = process_inbreast_roboflow_v11(inbreast_base_dir, inbreast_yaml, inbreast_mask_outdir, inbreast_img_outdir, texture_cfg)

    merged = pd.concat([cbis_df, mias_df, normal_df, inbreast_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS={len(cbis_df)}, MIAS={len(mias_df)}, NORMAL={len(normal_df)}, INBREAST={len(inbreast_df)})")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + MIAS + INBREAST (Roboflow YOLOv11) unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True)
    p.add_argument("--mias-info-csv", type=str, required=True)
    p.add_argument("--mias-images-dir", type=str, required=True)
    p.add_argument("--normal-dir", type=str, required=True)
    p.add_argument("--inbreast-base-dir", type=str, required=True, help="Roboflow INBreast dataset root dir")
    p.add_argument("--inbreast-yaml", type=str, required=True, help="Roboflow data.yaml file")
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DATASET")

    # Texture options
    p.add_argument("--texture-method", choices=["local_var", "sobel", "glcm"], default="local_var",
                   help="Texture map method: local_var (fast), sobel (fast), glcm (slow)")
    p.add_argument("--texture-strength", type=float, default=0.25,
                   help="How much texture is added to overlay (0..1). Smaller = subtler.")
    p.add_argument("--texture-alpha", type=float, default=0.6,
                   help="Blend factor between original and textured overlay (0..1).")
    p.add_argument("--texture-win", type=int, default=31, help="Window size for GLCM/local_var")
    p.add_argument("--texture-levels", type=int, default=8, help="Levels for exact glcm")
    p.add_argument("--texture-step", type=int, default=8, help="Step for exact glcm")
    p.add_argument("--sobel-sigma", type=float, default=3.0)
    p.add_argument("--local-var-win", type=int, default=31)

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    texture_cfg = {
        "method": args.texture_method,
        "strength": float(args.texture_strength),
        "alpha": float(args.texture_alpha),
        "win": int(args.texture_win),
        "levels": int(args.texture_levels),
        "step": int(args.texture_step),
        "sobel_sigma": float(args.sobel_sigma),
        "local_var_win": int(args.local_var_win),
    }
    prepare_unified(
        args.cbis_csv, args.mias_info_csv, args.mias_images_dir,
        args.normal_dir, args.inbreast_base_dir, args.inbreast_yaml,
        args.output_csv, args.outdir, texture_cfg
    )
