#!/usr/bin/env python3
"""
precompute_rois.py

Compute candidate ROIs per image using multiple classical methods and ensemble them.

Usage:
  python precompute_rois.py --input-csv unified_segmentation_dataset.csv \
      --out-csv precomputed_rois.csv --outdir PRECOMPUTED --pad 1.2 \
      --use-gt-for-selection

Output:
  CSV with extra columns:
    roi_x1, roi_y1, roi_x2, roi_y2, roi_method, roi_score, roi_candidates_json

Notes:
  - If masks exist and --use-gt-for-selection is used, script selects the candidate
    that maximizes IoU with the mask (good for preparing training crops).
  - Otherwise it picks the highest ensemble score.
"""
import os
import argparse
import json
import math
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- utilities ----------------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def safe_read_gray(path: str) -> Optional[np.ndarray]:
    if not isinstance(path, str) or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def iou_box(boxA, boxB):
    # boxes as (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1); interH = max(0, yB - yA + 1)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    union = areaA + areaB - inter
    if union <= 0: return 0.0
    return inter/union

def clamp_box(box, W, H):
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, int(round(x1))))
    y1 = max(0, min(H-1, int(round(y1))))
    x2 = max(0, min(W-1, int(round(x2))))
    y2 = max(0, min(H-1, int(round(y2))))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return (x1,y1,x2,y2)

def pad_box(box, pad_mul, W, H):
    x1,y1,x2,y2 = box
    w = x2 - x1; h = y2 - y1
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    nw = w * pad_mul; nh = h * pad_mul
    nx1 = cx - nw/2.0; ny1 = cy - nh/2.0
    nx2 = cx + nw/2.0; ny2 = cy + nh/2.0
    return clamp_box((nx1, ny1, nx2, ny2), W, H)

# ---------------- methods ----------------
def saliency_map(img_gray: np.ndarray) -> np.ndarray:
    # Try OpenCV saliency; fallback to gradient magnitude
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, salmap = saliency.computeSaliency(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR))
        if success:
            salmap = (salmap - salmap.min()) / (salmap.max() - salmap.min() + 1e-8)
            return salmap.astype(np.float32)
    except Exception:
        pass
    # fallback: Sobel magnitude + gaussian blur
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = cv2.GaussianBlur(mag, (7,7), 0)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32)

def saliency_box_from_map(salmap: np.ndarray, thr_frac: float = 0.6) -> Optional[Tuple[int,int,int,int,float]]:
    # threshold top fraction
    flat = salmap.flatten()
    if flat.size == 0: return None
    thr = np.quantile(flat, thr_frac)
    mask = (salmap >= thr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(cnt)
    box = (x, y, x+w-1, y+h-1)
    score = salmap[y:y+h, x:x+w].sum()
    return box + (float(score),)

def local_variance_map(img_gray: np.ndarray, win: int = 64) -> np.ndarray:
    imgf = img_gray.astype(np.float32)
    k = max(3, (win//2)*2+1)
    mean = cv2.boxFilter(imgf, ddepth=-1, ksize=(k,k), normalize=True)
    mean_sq = cv2.boxFilter(imgf*imgf, ddepth=-1, ksize=(k,k), normalize=True)
    var = np.maximum(0.0, mean_sq - mean*mean)
    var = cv2.GaussianBlur(var, (7,7), 0)
    var = (var - var.min()) / (var.max() - var.min() + 1e-8)
    return var.astype(np.float32)

def sliding_window_best_box(score_map: np.ndarray, win_w: int = 128, win_h: int = 128, stride:int = 32) -> Optional[Tuple[int,int,int,int,float]]:
    H,W = score_map.shape
    if H < 1 or W < 1:
        return None
    best = None
    best_score = -1.0
    for y in range(0, max(1,H-win_h+1), max(1,stride)):
        for x in range(0, max(1,W-win_w+1), max(1,stride)):
            region = score_map[y:y+win_h, x:x+win_w]
            if region.size == 0: continue
            s = float(region.sum())
            if s > best_score:
                best_score = s
                best = (x,y, min(W-1, x+win_w-1), min(H-1, y+win_h-1), s)
    # if best None, try whole image
    if best is None:
        return (0,0,W-1,H-1,float(score_map.sum()))
    return best

def threshold_morph_box(img_gray: np.ndarray) -> Optional[Tuple[int,int,int,int,float]]:
    # Otsu threshold then morphology
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # depending on lesion darkness, we may want inverted mask too; try both and pick larger object
    th_inv = cv2.bitwise_not(th)
    boxes = []
    for mask in (th, th_inv):
        # morphological open to remove noise; close to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=lambda c: cv2.contourArea(c))
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append(((x,y,x+w-1,y+h-1), float(area)))
    if not boxes:
        return None
    # pick largest area
    boxes.sort(key=lambda x: x[1], reverse=True)
    b, score = boxes[0]
    return b + (score,)

def edge_contour_convex_box(img_gray: np.ndarray) -> Optional[Tuple[int,int,int,int,float]]:
    edges = cv2.Canny(img_gray, 50, 150)
    # dilate to make contours thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    # convex hull
    hull = cv2.convexHull(cnt)
    x,y,w,h = cv2.boundingRect(hull)
    area = cv2.contourArea(hull)
    return (x,y,x+w-1,y+h-1, float(area))

# ---------------- ensemble ----------------
def ensemble_candidates(cands: List[Tuple[int,int,int,int,float]], W:int, H:int,
                        weights: Optional[Dict[str,float]]=None) -> List[Dict[str,Any]]:
    """
    cands: list of candidates as tuples (x1,y1,x2,y2,score)
    returns list of dicts with normalized score.
    """
    out = []
    if not cands:
        # fallback whole image
        out.append({"box": (0,0,W-1,H-1), "score": 1.0, "method": "fallback"})
        return out
    scores = np.array([c[4] for c in cands], dtype=np.float32)
    # normalize to 0..1
    if scores.max() - scores.min() < 1e-8:
        norm = np.ones_like(scores)
    else:
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    for i,c in enumerate(cands):
        out.append({"box": clamp_box((c[0],c[1],c[2],c[3]), W,H), "score": float(norm[i]), "method": f"method_{i}"})
    # sort descending
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ---------------- main pipeline per image ----------------
def compute_rois_for_image(img_path: str, mask_path: Optional[str], pad: float = 1.2,
                           use_gt_for_selection: bool = True) -> Dict[str,Any]:
    img = safe_read_gray(img_path)
    if img is None:
        return {"error": "img_not_found"}
    H,W = img.shape[:2]
    mask = safe_read_gray(mask_path) if mask_path and os.path.exists(mask_path) else None
    # ensure mask same shape if present
    if mask is not None and mask.shape != img.shape:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    cands = []

    # 1) saliency
    sal = saliency_map(img)
    sb = saliency_box_from_map(sal, thr_frac=0.80)
    if sb is not None:
        cands.append(sb)  # (x1,y1,x2,y2,score)

    # 2) local var sliding window (two window sizes)
    lv = local_variance_map(img, win=64)
    sw1 = sliding_window_best_box(lv, win_w=min(256,W), win_h=min(256,H), stride=max(16, min(W,H)//32))
    if sw1 is not None:
        cands.append(sw1)
    sw2 = sliding_window_best_box(lv, win_w=min(128,W), win_h=min(128,H), stride=max(8, min(W,H)//64))
    if sw2 is not None:
        cands.append(sw2)

    # 3) threshold + morphology
    thb = threshold_morph_box(img)
    if thb is not None:
        cands.append(thb)

    # 4) edge + contour + convex hull
    ecb = edge_contour_convex_box(img)
    if ecb is not None:
        cands.append(ecb)

    # add whole image as fallback candidate with low score
    cands.append((0,0,W-1,H-1, 0.1))

    # ensemble-normalize
    ensemble_list = ensemble_candidates(cands, W, H)

    # If using GT mask for selection and mask present, pick candidate with highest IoU with mask bbox
    selected = None
    if use_gt_for_selection and mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            gt_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            best_iou = -1.0
            best_idx = 0
            for i, cand in enumerate(ensemble_list):
                iou = iou_box(cand["box"], gt_box)
                if iou > best_iou:
                    best_iou = iou; best_idx = i
            # expand selected with padding multiplier
            sel = ensemble_list[best_idx]
            sel_box = pad_box(sel["box"], pad, W, H)
            sel_score = sel["score"]
            sel_method = sel["method"]
            selected = {"box": sel_box, "score": float(sel_score), "method": sel_method, "iou_with_gt": float(best_iou)}
        else:
            # no gt mask area -> fallback to top scored
            top = ensemble_list[0]
            sel_box = pad_box(top["box"], pad, W, H)
            selected = {"box": sel_box, "score": float(top["score"]), "method": top["method"], "iou_with_gt": 0.0}
    else:
        # pick top ensemble-scored candidate
        top = ensemble_list[0]
        sel_box = pad_box(top["box"], pad, W, H)
        selected = {"box": sel_box, "score": float(top["score"]), "method": top["method"], "iou_with_gt": None}

    # produce serializable candidate list (first up to 6)
    candidates_serial = [{"box": cand["box"], "score": float(cand["score"]), "method": cand["method"]} for cand in ensemble_list[:6]]

    return {"selected": selected, "candidates": candidates_serial}

# ---------------- CLI main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=str, required=True, help="CSV with image_file_path and roi_mask_file_path columns")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="precomputed_rois")
    p.add_argument("--pad", type=float, default=1.2, help="Padding multiplier for chosen bbox")
    p.add_argument("--use-gt-for-selection", action="store_true", help="If mask exists, select candidate with best IoU to mask")
    p.add_argument("--preview", action="store_true", help="Save preview overlay images into outdir/previews/")
    p.add_argument("--max-samples", type=int, default=None, help="Process only first N samples")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    preview_dir = os.path.join(args.outdir, "previews")
    if args.preview:
        ensure_dir(preview_dir)

    df = pd.read_csv(args.input_csv)
    rows = []
    n = len(df) if args.max_samples is None else min(len(df), args.max_samples)
    for i in tqdm(range(n), desc="Precomputing ROIs"):
        row = df.iloc[i].to_dict()
        img_path = row.get("image_file_path")
        mask_path = row.get("roi_mask_file_path")
        res = compute_rois_for_image(img_path, mask_path, pad=args.pad, use_gt_for_selection=args.use_gt_for_selection)
        if "error" in res:
            row.update({"roi_x1": None, "roi_y1": None, "roi_x2": None, "roi_y2": None,
                        "roi_method": None, "roi_score": None, "roi_candidates_json": None})
            rows.append(row)
            continue
        sel = res["selected"]
        bx = sel["box"]
        row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"] = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        row["roi_method"] = sel["method"]
        row["roi_score"] = sel["score"]
        row["roi_iou_with_gt"] = res["selected"].get("iou_with_gt", None)
        row["roi_candidates_json"] = json.dumps(res["candidates"])
        rows.append(row)

        if args.preview:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                H,W = img.shape[:2]
                overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                x1,y1,x2,y2 = row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"]
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                if mask_path and os.path.exists(mask_path):
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None and m.shape[:2] != (H,W):
                        m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
                    if m is not None:
                        contours, _ = cv2.findContours((m>0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, (0,0,255), 1)
                fname = os.path.join(preview_dir, f"preview_{i:05d}.png")
                cv2.imwrite(fname, overlay)
            except Exception:
                pass

    outdf = pd.DataFrame(rows)
    outdf.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved {len(outdf)} rows to {args.out_csv}")
    if args.preview:
        print(f"[INFO] Saved preview images to {preview_dir}")

if __name__ == "__main__":
    main()
