import os
import argparse
import pandas as pd
import numpy as np
import cv2
import h5py
import re
import struct


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 3)
    # Removed CLAHE here to let model learn optimal CLAHE dynamically
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def process_cbis(input_csv, mask_outdir, image_outdir):
    """Prepare CBIS dataset"""
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)

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

        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"

        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")

        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path, merged_mask)

        base_row["dataset"] = "CBIS"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)

    return pd.DataFrame(rows)

def process_mias_from_csv(info_csv_path, images_dir, mask_outdir, image_outdir):
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)

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

        img = preprocess_image(img)

        is_normal = (row["CLASS"] == "NORM")
        mask = np.zeros_like(img, dtype=np.uint8)

        if not is_normal and not pd.isna(row["X"]) and not pd.isna(row["Y"]) and not pd.isna(row["RADIUS"]):
            x, y, r = int(row["X"]), int(row["Y"]), int(row["RADIUS"])
            cv2.circle(mask, (x, y), r, 255, -1)

        pid = f"MIAS_{refnum}"
        proc_img_path = os.path.join(image_outdir, f"{pid}.png")
        mask_path = os.path.join(mask_outdir, f"{pid}_mask.png")

        cv2.imwrite(proc_img_path, img)
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


def process_normal_images(normal_img_dir, mask_outdir, image_outdir):
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)

    rows = []
    normal_img_files = [f for f in os.listdir(normal_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in normal_img_files:
        img_path = os.path.join(normal_img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = preprocess_image(img)

        pid = os.path.splitext(img_file)[0]
        proc_img_path = os.path.join(image_outdir, f"NORMAL_{pid}.png")
        mask_path = os.path.join(mask_outdir, f"NORMAL_{pid}_mask.png")

        cv2.imwrite(proc_img_path, img)

        mask = np.zeros_like(img, dtype=np.uint8)
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

def parse_inbreast_roi(roi_file_path):
    try:
        with open(roi_file_path, 'rb') as f:
            data = f.read()

        if len(data) < 512:
            print(f"[WARNING] ROI file too small: {roi_file_path}")
            return []

        roi_data = data[512:]  # Skip the header

        rois = []
        record_size = 64  # Each ROI record is 64 bytes
        num_records = len(roi_data) // record_size

        for i in range(num_records):
            record = roi_data[i * record_size : (i + 1) * record_size]

            if len(record) != record_size:
                continue

            shape_type = struct.unpack('<H', record[0:2])[0]
            n_points = struct.unpack('<H', record[2:4])[0]

            if shape_type != 5:  # 5 indicates a Circle
                continue

            x = struct.unpack('<f', record[4:8])[0]
            y = struct.unpack('<f', record[8:12])[0]
            radius = struct.unpack('<f', record[12:16])[0]

            rois.append((int(x), int(y), int(radius)))

        return rois

    except Exception as e:
        print(f"[ERROR] Failed to parse ROI file {roi_file_path}: {e}")
        return []


def process_inbreast(images_dir, roi_dir, mask_outdir, image_outdir):
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)

    rows = []
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]

    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        roi_filename = os.path.splitext(img_file)[0] + ".roi"
        roi_path = os.path.join(roi_dir, roi_filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Cannot read image: {img_path}")
            continue

        img = preprocess_image(img)
        mask = np.zeros_like(img, dtype=np.uint8)

        rois = parse_inbreast_roi(roi_path)
        for (x, y, r) in rois:
            cv2.circle(mask, (x, y), r, 255, -1)

        pid = os.path.splitext(img_file)[0]
        proc_img_path = os.path.join(image_outdir, f"INBREAST_{pid}.png")
        mask_path = os.path.join(mask_outdir, f"INBREAST_{pid}_mask.png")

        cv2.imwrite(proc_img_path, img)
        cv2.imwrite(mask_path, mask)

        row = {
            "dataset": "INBREAST",
            "patient_id": f"INBREAST_{pid}",
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path,
            "pathology": "Calcification" if len(rois) > 0 else "N",
            "abnormality_id": "Calcification" if len(rois) > 0 else "None",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_unified(cbis_csv, mias_info_csv, mias_images_dir, normal_dir, inbreast_img_dir, inbreast_roi_dir, output_csv, outdir):
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES")
    cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mias_img_dir = os.path.join(outdir, "MIAS_IMAGES")
    mias_mask_dir = os.path.join(outdir, "MIAS_MASKS")
    normal_img_dir = os.path.join(outdir, "NORMAL_IMAGES")
    normal_mask_dir = os.path.join(outdir, "NORMAL_MASKS")
    inbreast_img_outdir = os.path.join(outdir, "INBREAST_IMAGES")
    inbreast_mask_outdir = os.path.join(outdir, "INBREAST_MASKS")

    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir)
    mias_df = process_mias_from_csv(mias_info_csv, mias_images_dir, mias_mask_dir, mias_img_dir)
    normal_df = process_normal_images(normal_dir, normal_mask_dir, normal_img_dir)
    inbreast_df = process_inbreast(inbreast_img_dir, inbreast_roi_dir, inbreast_mask_outdir, inbreast_img_outdir)

    merged = pd.concat([cbis_df, mias_df, normal_df, inbreast_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)

    print(f"[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} (CBIS={len(cbis_df)}, MIAS={len(mias_df)}, NORMAL={len(normal_df)}, INBREAST={len(inbreast_df)})")


def get_args():
    p = argparse.ArgumentParser(description="Prepare CBIS-DDSM + MIAS + INBREAST unified dataset")
    p.add_argument("--cbis-csv", type=str, required=True)
    p.add_argument("--mias-info-csv", type=str, required=True)
    p.add_argument("--mias-images-dir", type=str, required=True)
    p.add_argument("--normal-dir", type=str, required=True)
    p.add_argument("--inbreast-img-dir", type=str, required=True)
    p.add_argument("--inbreast-roi-dir", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DATASET")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare_unified(
        args.cbis_csv, args.mias_info_csv, args.mias_images_dir,
        args.normal_dir, args.inbreast_img_dir, args.inbreast_roi_dir,
        args.output_csv, args.outdir
    )
