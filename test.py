# test.py (updated)
import sys
import os
import torch
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# Import the robust loader from your model file. Adjust the module name if you saved file under a different name.
from segmentation_model import load_model_from_checkpoint, create_model  # either helper will be available

class SegmentationViewer(QWidget):
    def __init__(self, dataset_csv, model_checkpoint, img_size=256):
        super().__init__()
        self.setWindowTitle("Segmentation Model Viewer")
        self.df = pd.read_csv(dataset_csv)
        self.img_size = img_size
        self.index = 0
        self.clahe_clip = 2.0  # Default CLAHE clip limit

        # Load model robustly (auto-detect architecture from checkpoint where possible)
        device = torch.device("cpu")
        try:
            # prefer ACA-Atrous UNet but allow auto-detection
            self.model, info, chosen = load_model_from_checkpoint(model_checkpoint, preferred_model_name="aca-atrous-unet", device=device, img_size=img_size)
            print(f"[MODEL] chosen={chosen}\n{info}")
        except Exception as e:
            print(f"[ERROR] loading model: {e}")
            # fallback to instantiating plain ACAAtrousUNet without weights
            from segmentation_model import ACAAtrousUNet
            self.model = ACAAtrousUNet(in_ch=1, out_ch=1).to(device)
            print("[INFO] instantiated fresh ACAAtrousUNet without checkpoint.")

        self.model.eval()

        # UI Elements
        self.img_label = QLabel()
        self.gt_label = QLabel()
        self.pred_label = QLabel()

        self.next_btn = QPushButton("Next")
        self.prev_btn = QPushButton("Previous")

        self.next_btn.clicked.connect(self.next_image)
        self.prev_btn.clicked.connect(self.prev_image)

        # CLAHE clip limit slider
        self.clahe_slider = QSlider(Qt.Horizontal)
        self.clahe_slider.setMinimum(10)    # Represents clip limit 0.1
        self.clahe_slider.setMaximum(100)   # Represents clip limit 10.0
        self.clahe_slider.setValue(20)      # Starts at 2.0
        self.clahe_slider.setTickInterval(10)
        self.clahe_slider.setTickPosition(QSlider.TicksBelow)
        self.clahe_slider.valueChanged.connect(self.on_clahe_slider_changed)

        layout = QVBoxLayout()
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.img_label)
        img_layout.addWidget(self.gt_label)
        img_layout.addWidget(self.pred_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(img_layout)
        layout.addWidget(QLabel("CLAHE Clip Limit (adjust to see effect)"))
        layout.addWidget(self.clahe_slider)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # problem cases output
        self.problem_dir = "problem_cases"
        os.makedirs(self.problem_dir, exist_ok=True)
        self.problem_log = []

        self.show_image()

    def to_pixmap(self, img_array):
        mn = img_array.min(); mx = img_array.max()
        if mx - mn < 1e-8:
            img = (np.clip(img_array, 0, 255)).astype(np.uint8)
        else:
            img = ((img_array - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)

    def apply_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        return clahe.apply(img)

    def save_problem_case(self, row, img, gt_mask, pred_mask, note):
        base = os.path.splitext(os.path.basename(row["image_file_path"]))[0]
        fname_base = f"{self.index:04d}_{base}"
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_img.png"), img)
        vis_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours((gt_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_gt, contours, -1, (0,0,255), 1)
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_gt.png"), vis_gt)
        vis_pred = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours_p, _ = cv2.findContours((pred_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_pred, contours_p, -1, (0,255,0), 1)
        cv2.imwrite(os.path.join(self.problem_dir, fname_base + "_pred.png"), vis_pred)
        self.problem_log.append({"index": self.index, "image": row["image_file_path"], "mask": row["roi_mask_file_path"], "note": note})

    def load_current_data(self):
        row = self.df.iloc[self.index]

        img = cv2.imread(row["image_file_path"], cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(row["roi_mask_file_path"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Cannot read image: {row['image_file_path']}")
        if gt_mask is None:
            gt_mask = np.zeros_like(img, dtype=np.uint8)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        gt_mask_bin = (gt_mask > 0).astype(np.uint8)

        img_clahe = self.apply_clahe(img)

        img_tensor = torch.tensor(img_clahe / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pred_mask_bin = np.zeros_like(gt_mask_bin)
        try:
            with torch.no_grad():
                out = self.model(img_tensor)
                if isinstance(out, tuple):
                    out = out[0]
                pred = torch.sigmoid(out)
                pred_mask = pred.squeeze(0).squeeze(0).cpu().numpy()
                if pred_mask.shape != (self.img_size, self.img_size):
                    pred_mask = cv2.resize(pred_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"[PRED ERROR] index={self.index}, error={e}")
            self.save_problem_case(row, img_clahe, gt_mask_bin, pred_mask_bin, note=f"prediction_error: {e}")
            return img_clahe, gt_mask_bin, pred_mask_bin

        # problem detection
        if gt_mask_bin.sum() > 0 and pred_mask_bin.sum() == 0:
            print(f"[PROBLEM] Missed lesion at index {self.index}.")
            self.save_problem_case(row, img_clahe, gt_mask_bin, pred_mask_bin, note="missed_lesion")

        inter = np.logical_and(gt_mask_bin > 0, pred_mask_bin > 0).sum()
        union = np.logical_or(gt_mask_bin > 0, pred_mask_bin > 0).sum()
        iou = float(inter) / (union + 1e-8)
        if union > 0 and iou < 0.05:
            print(f"[LOW IOU] index={self.index}, IoU={iou:.4f}")
            self.save_problem_case(row, img_clahe, gt_mask_bin, pred_mask_bin, note=f"low_iou_{iou:.4f}")

        return img_clahe, gt_mask_bin, pred_mask_bin

    def show_image(self):
        img, gt_mask, pred_mask = self.load_current_data()
        self.img_label.setPixmap(self.to_pixmap(img))
        self.gt_label.setPixmap(self.to_pixmap(gt_mask * 255))
        self.pred_label.setPixmap(self.to_pixmap(pred_mask * 255))

    def next_image(self):
        if self.index < len(self.df) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def on_clahe_slider_changed(self, value):
        self.clahe_clip = max(value / 10.0, 0.1)  # Maps slider to [0.1, 10.0]
        self.show_image()

    def closeEvent(self, event):
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
