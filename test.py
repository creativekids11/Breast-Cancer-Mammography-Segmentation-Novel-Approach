# test.py (problem-cases + problems-only viewer)
import sys
import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# try to import robust loader from your model file; fallback to plain model if missing
try:
    from segmentation_model import load_model_from_checkpoint, create_model, ACAAtrousUNet
except Exception:
    # If segmentation_model doesn't expose loader, import ACAAtrousUNet class fallback
    try:
        from segmentation_model import load_model_from_checkpoint, create_model, ACAAtrousUNet
    except Exception:
        # As a last resort, try to import ACAAtrousUNet only (user likely has it defined)
        from segmentation_model import ACAAtrousUNet  # may raise if not present

class SegmentationViewer(QWidget):
    def __init__(self, dataset_csv, model_checkpoint, img_size=512):
        super().__init__()
        self.setWindowTitle("Segmentation Model Viewer")

        # config thresholds for problem detection (tweakable)
        self.iou_threshold = 0.05
        self.max_prob_threshold = 0.6
        self.mean_prob_threshold = 0.3
        self.pred_thr = 0.5  # threshold to binarize pred map
        self.img_size = img_size

        self.df = pd.read_csv(dataset_csv)
        self.index = 0  # index into the active index list (all_indices or problem_indices)
        self.active_is_problems = False

        self.all_indices = list(range(len(self.df)))
        self.problem_indices = []   # indices of dataset rows flagged as problem
        self.problem_log = []       # records to write CSV
        self.problem_set = set()    # to avoid duplicates
        self.problem_dir = "problem_cases"
        os.makedirs(self.problem_dir, exist_ok=True)

        self.clahe_clip = 2.0  # Default CLAHE clip limit

        # Load model robustly (auto-detect architecture from checkpoint where possible)
        device = torch.device("cpu")
        self.device = device
        self.model = None
        try:
            # prefer ACA-Atrous UNet but allow auto-detection
            if "load_model_from_checkpoint" in globals():
                self.model, info, chosen = load_model_from_checkpoint(model_checkpoint, preferred_model_name="aca-atrous-unet", device=device, img_size=img_size)
                print(f"[MODEL] chosen={chosen}\n{info}")
            else:
                # fallback to explicit create_model if available
                self.model = create_model("aca-atrous-unet", device=device, img_size=img_size)
                print("[MODEL] created model without checkpoint loader (no weights loaded).")
        except Exception as e:
            print(f"[ERROR] loading model: {e}")
            try:
                self.model = ACAAtrousUNet(in_ch=1, out_ch=1).to(device)
                print("[INFO] instantiated fresh ACAAtrousUNet without checkpoint.")
            except Exception as ee:
                raise RuntimeError(f"Failed to instantiate fallback model: {ee}")

        self.model.eval()

        # UI Elements
        self.img_label = QLabel(); self.gt_label = QLabel(); self.pred_label = QLabel()
        self.next_btn = QPushButton("Next")
        self.prev_btn = QPushButton("Previous")
        self.toggle_mode_btn = QPushButton("Mode: ALL")
        self.next_btn.clicked.connect(self.next_image)
        self.prev_btn.clicked.connect(self.prev_image)
        self.toggle_mode_btn.clicked.connect(self.toggle_mode)

        # CLAHE slider
        self.clahe_slider = QSlider(Qt.Horizontal)
        self.clahe_slider.setMinimum(10); self.clahe_slider.setMaximum(100); self.clahe_slider.setValue(20)
        self.clahe_slider.setTickInterval(10); self.clahe_slider.setTickPosition(QSlider.TicksBelow)
        self.clahe_slider.valueChanged.connect(self.on_clahe_slider_changed)

        # Layout
        layout = QVBoxLayout()
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.img_label); img_layout.addWidget(self.gt_label); img_layout.addWidget(self.pred_label)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn); btn_layout.addWidget(self.next_btn); btn_layout.addWidget(self.toggle_mode_btn)
        layout.addLayout(img_layout)
        layout.addWidget(QLabel("CLAHE Clip Limit (adjust to see effect)"))
        layout.addWidget(self.clahe_slider)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # run full scan to collect problem cases (synchronous)
        print("[INFO] Scanning dataset for problematic cases. This may take a while...")
        t0 = time.time()
        self.find_all_problems()
        t1 = time.time()
        print(f"[INFO] Scan finished in {t1-t0:.1f}s. Found {len(self.problem_indices)} problematic cases.")

        # show first image (either in all mode or problems if none)
        if self.active_is_problems and len(self.problem_indices) == 0:
            self.active_is_problems = False
            self.toggle_mode_btn.setText("Mode: ALL")
        self.show_image()

    # ---------------- utilities ----------------
    def to_pixmap(self, img_array):
        mn = float(img_array.min()); mx = float(img_array.max())
        if mx - mn < 1e-8:
            img = (np.clip(img_array, 0, 255)).astype(np.uint8)
        else:
            img = ((img_array - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)

    # inside test.py - replace apply_clahe
    def extract_breast_mask_local(self, img_uint8: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(img_uint8, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean() < 128: th = 255 - th
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((th>0).astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return np.ones_like(img_uint8, dtype=np.uint8) * 255
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8) * 255
        mask = cv2.blur(mask, (7,7))
        mask = (mask > 127).astype(np.uint8) * 255
        return mask

    def adaptive_clahe_viewer(self, img_uint8: np.ndarray, base_clip: float = 2.0):
        if img_uint8.dtype != np.uint8:
            img = img_uint8.astype(np.uint8)
        else:
            img = img_uint8.copy()
        img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        mask = self.extract_breast_mask_local(img_filtered)
        breast_pixels = img_filtered[mask>0] if mask.sum()>0 else img_filtered.flatten()
        std = float(np.std(breast_pixels)) if breast_pixels.size>0 else 0.0
        std_small, std_large = 6.0, 40.0
        min_clip, max_clip = 0.5, 6.0
        clip = float(np.clip(np.interp(std, [std_small, std_large], [max_clip, min_clip]), min_clip, max_clip))
        clip = max(0.1, clip * (base_clip / 2.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        clahe_img = clahe.apply(img_filtered)
        out = img.copy()
        if mask.sum() > 0:
            feather = cv2.GaussianBlur((mask>0).astype(np.float32), (31,31), 0)
            feather = np.clip(feather, 0.0, 1.0)
            out_f = (img.astype(np.float32) * (1.0 - feather) + clahe_img.astype(np.float32) * feather)
            out = np.clip(out_f, 0, 255).astype(np.uint8)
        else:
            out = clahe_img
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = cv2.medianBlur(out, 3)
        return out

    # then replace in your class:
    def apply_clahe(self, img):
        return self.adaptive_clahe_viewer(img, base_clip=self.clahe_clip)

    def get_active_index_list(self):
        return self.problem_indices if self.active_is_problems else self.all_indices

    # ---------------- prediction / problem logic ----------------
    def predict_single(self, idx):
        """Returns a dict with img_clahe, gt_mask_bin, pred_prob_map, pred_mask_bin, iou, max_prob, mean_prob, n_components"""
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_file_path"], cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(row["roi_mask_file_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {row['image_file_path']}")
        if gt_mask is None:
            gt_mask = np.zeros_like(img, dtype=np.uint8)

        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_resized > 0).astype(np.uint8)

        img_clahe = self.apply_clahe(img_resized)
        img_tensor = torch.tensor(img_clahe / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        result = {
            "idx": idx, "row": row, "img": img_resized, "img_clahe": img_clahe,
            "gt_mask": gt_resized, "gt_bin": gt_bin,
            "pred_prob": None, "pred_mask": None,
            "iou": None, "max_prob": 0.0, "mean_prob": 0.0, "n_components": 0, "error": None
        }

        try:
            with torch.no_grad():
                out = self.model(img_tensor)
                if isinstance(out, tuple):
                    out = out[0]
                pred = torch.sigmoid(out)
                pred_map = pred.squeeze(0).squeeze(0).cpu().numpy()
                # resize if necessary
                if pred_map.shape != (self.img_size, self.img_size):
                    pred_map = cv2.resize(pred_map, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                pred_mask = (pred_map > self.pred_thr).astype(np.uint8)
                # components
                num_labels, labels = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)
                n_components = max(0, num_labels - 1)
                inter = np.logical_and(gt_bin > 0, pred_mask > 0).sum()
                union = np.logical_or(gt_bin > 0, pred_mask > 0).sum()
                iou = float(inter) / (union + 1e-8) if union > 0 else 0.0
                max_prob = float(pred_map.max())
                mean_prob = float(pred_map[pred_mask > 0].mean()) if pred_mask.sum() > 0 else 0.0

                result.update({
                    "pred_prob": pred_map, "pred_mask": pred_mask,
                    "iou": iou, "max_prob": max_prob, "mean_prob": mean_prob,
                    "n_components": n_components
                })
        except Exception as e:
            result["error"] = str(e)

        return result

    def is_problem_case(self, res):
        """Return (bool, list_of_problem_types)"""
        problems = []
        # prediction error
        if res["error"] is not None:
            problems.append("prediction_error")
            return True, problems

        gt_has = res["gt_bin"].sum() > 0
        pred_has = res["pred_mask"].sum() > 0

        # missed lesion
        if gt_has and not pred_has:
            problems.append("missed_lesion")

        # low iou
        if (res["iou"] is not None) and (res["iou"] < self.iou_threshold) and (res["pred_mask"].sum() + res["gt_bin"].sum() > 0):
            problems.append(f"low_iou_{res['iou']:.4f}")

        # low confidence: <=1 components and low max/mean
        if pred_has:
            if res["n_components"] <= 1 and (res["max_prob"] < self.max_prob_threshold or res["mean_prob"] < self.mean_prob_threshold):
                problems.append(f"low_confidence_max{res['max_prob']:.3f}_mean{res['mean_prob']:.3f}_comp{res['n_components']}")

        return (len(problems) > 0), problems

    def save_problem_case(self, res, problem_types):
        """Saves img, gt overlay and pred overlay; logs metadata. Avoid duplicates via problem_set."""
        idx = res["idx"]
        row = res["row"]
        base = os.path.splitext(os.path.basename(row["image_file_path"]))[0]
        fname_base = f"{idx:04d}_{base}"
        if fname_base in self.problem_set:
            return
        self.problem_set.add(fname_base)

        # images to save: raw CLAHE image, GT overlay, pred overlay
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_img.png"), res["img_clahe"])

        vis_gt = cv2.cvtColor(res["img_clahe"], cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours((res["gt_bin"] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_gt, contours, -1, (0,0,255), 1)
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_gt.png"), vis_gt)

        vis_pred = cv2.cvtColor(res["img_clahe"], cv2.COLOR_GRAY2BGR)
        if res["pred_mask"] is not None:
            contours_p, _ = cv2.findContours((res["pred_mask"] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_pred, contours_p, -1, (0,255,0), 1)
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_pred.png"), vis_pred)

        # log entry
        log_entry = {
            "index": idx,
            "image_file": row["image_file_path"],
            "mask_file": row["roi_mask_file_path"],
            "fname_base": fname_base,
            "problems": ";".join(problem_types),
            "iou": res["iou"],
            "max_prob": res["max_prob"],
            "mean_prob": res["mean_prob"],
            "n_components": res["n_components"],
            "error": res["error"]
        }
        self.problem_log.append(log_entry)
        # append to problem_indices so viewer can switch immediately
        self.problem_indices.append(idx)

    def find_all_problems(self):
        """Scan entire dataset, save problem images and build problem log. Synchronous."""
        for idx in range(len(self.df)):
            try:
                res = self.predict_single(idx)
            except Exception as e:
                print(f"[SCAN ERROR] index={idx} error={e}")
                res = {"idx": idx, "row": self.df.iloc[idx], "img_clahe": None, "img": None, "gt_bin": np.zeros((self.img_size,self.img_size),dtype=np.uint8),
                       "pred_mask": np.zeros((self.img_size,self.img_size),dtype=np.uint8), "iou": None, "max_prob": 0.0, "mean_prob": 0.0, "n_components": 0, "error": str(e)}
            is_prob, problems = self.is_problem_case(res)
            if is_prob:
                self.save_problem_case(res, problems)
            # progress print every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"[SCAN] processed {idx+1}/{len(self.df)}")
        # write problem_log to csv now
        if self.problem_log:
            log_df = pd.DataFrame(self.problem_log)
            log_df.to_csv(os.path.join(self.problem_dir, "problem_log.csv"), index=False)
            print(f"[INFO] Saved problem_log.csv with {len(self.problem_log)} entries to {self.problem_dir}")
        else:
            print("[INFO] No problems detected; problem_log is empty.")

    # ---------------- UI actions ----------------
    def load_current_data(self):
        active_list = self.get_active_index_list()
        if len(active_list) == 0:
            raise RuntimeError("No items in active list to display.")
        dataset_idx = active_list[self.index]
        row = self.df.iloc[dataset_idx]
        res = self.predict_single(dataset_idx)
        # If this index wasn't saved in problem scan (could happen if later toggled thresholds), optionally save
        is_prob, problems = self.is_problem_case(res)
        if is_prob and (dataset_idx not in self.problem_indices):
            self.save_problem_case(res, problems)
        # convert masks for display
        gt_display = (res["gt_bin"] * 255).astype(np.uint8)
        pred_display = (res["pred_mask"] * 255).astype(np.uint8) if res["pred_mask"] is not None else np.zeros_like(gt_display)
        return res["img_clahe"], gt_display, pred_display

    def show_image(self):
        try:
            img, gt_mask, pred_mask = self.load_current_data()
            self.img_label.setPixmap(self.to_pixmap(img))
            self.gt_label.setPixmap(self.to_pixmap(gt_mask))
            self.pred_label.setPixmap(self.to_pixmap(pred_mask))
            # update window title with progress
            active_list = self.get_active_index_list()
            self.setWindowTitle(f"Segmentation Viewer - {'PROBLEMS' if self.active_is_problems else 'ALL'} "
                                f"[{self.index+1}/{len(active_list)}] (total_problems={len(self.problem_indices)})")
        except Exception as e:
            print(f"[SHOW ERROR] {e}")

    def next_image(self):
        active_list = self.get_active_index_list()
        if self.index < len(active_list) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def toggle_mode(self):
        # toggle between all and problems-only
        self.active_is_problems = not self.active_is_problems
        if self.active_is_problems:
            self.toggle_mode_btn.setText("Mode: PROBLEMS")
            if len(self.problem_indices) == 0:
                print("[INFO] No problem cases detected; staying in ALL mode.")
                self.active_is_problems = False
                self.toggle_mode_btn.setText("Mode: ALL")
            else:
                self.index = 0
        else:
            self.toggle_mode_btn.setText("Mode: ALL")
            self.index = 0
        self.show_image()

    def on_clahe_slider_changed(self, value):
        self.clahe_clip = max(value / 10.0, 0.1)
        # re-render current image with new CLAHE
        self.show_image()

    def closeEvent(self, event):
        # save problem log again to ensure latest entries
        if self.problem_log:
            pd.DataFrame(self.problem_log).to_csv(os.path.join(self.problem_dir, "problem_log.csv"), index=False)
            print(f"[INFO] Saved problem log to {os.path.join(self.problem_dir, 'problem_log.csv')}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SegmentationViewer(
        dataset_csv="unified_segmentation_dataset.csv",
        model_checkpoint="checkpoints_directimg/best.pth",
        img_size=512
    )
    viewer.show()
    sys.exit(app.exec())
