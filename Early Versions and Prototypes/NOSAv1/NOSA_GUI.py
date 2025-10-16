from NOSA_v1 import predict_tumor 
import sys
import numpy as np
import mat73
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QFileDialog, QCheckBox, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

heatmap_path = resource_path("tumor_heatmap_mask.npy")

if os.path.exists(heatmap_path):
    heatmap = np.load(heatmap_path)
else:
    print("Warning: tumor_heatmap_mask.npy not found!")
    heatmap = None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(200, 200, 800, 500)

        self.drag_pos = None # to track mouse drag position

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Title Bar
        self.title_bar = QFrame()
        self.title_bar.setStyleSheet("background-color: #2c3e50;")
        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(10, 5, 10, 5)

        self.title_label = QLabel("NOSA v1.0")
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        left_spacer = QWidget(); left_spacer.setFixedWidth(50)
        right_spacer = QWidget(); right_spacer.setFixedWidth(50)

        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setStyleSheet("background: #4d7094; color: white; font-size: 16px;")
        self.fullscreen_btn.setFixedSize(30, 30)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        self.close_btn = QPushButton("✕")
        self.close_btn.setStyleSheet("background: #ff0000; color: white; font-size: 16px;")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.clicked.connect(self.close)

        self.title_layout.addWidget(left_spacer)
        self.title_layout.addWidget(self.title_label, stretch=1, alignment=Qt.AlignCenter)
        self.title_layout.addWidget(self.fullscreen_btn)
        self.title_layout.addWidget(self.close_btn)
        self.main_layout.addWidget(self.title_bar)

        self.title_bar.mousePressEvent = self.start_window_drag
        self.title_bar.mouseMoveEvent = self.move_window
        self.title_bar.mouseReleaseEvent = self.stop_window_drag

        # Content Area
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        self.main_layout.addWidget(self.image_label, stretch=1)

        self.path_label = QLabel("")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.path_label)

        # Load Image Button
        self.load_img_btn = QPushButton("Load Image (.mat, .nii, .nii.gz)")
        self.load_img_btn.setStyleSheet("background: #4d7094; color: white; font-size: 16px;")
        self.load_img_btn.setFixedSize(300, 30)
        self.load_img_btn.clicked.connect(self.load_file)
        self.main_layout.addWidget(self.load_img_btn, alignment=Qt.AlignCenter)

        # Show Ground Truth Mask Button (hidden until image is loaded)
        self.show_gt_btn = QPushButton("Show Solution")
        self.show_gt_btn.setStyleSheet("background: #16a085; color: white; font-size: 16px;")
        self.show_gt_btn.setFixedSize(250, 30)
        self.show_gt_btn.clicked.connect(self.show_ground_truth_mask)
        self.show_gt_btn.setVisible(False)  # only visible if available
        self.main_layout.addWidget(self.show_gt_btn, alignment=Qt.AlignCenter)

        self.ground_truth_mask = None  # store tumorMask if found

        # Find Tumors Button (hidden until image is loaded)
        self.find_tumors_btn = QPushButton("Find Tumors")
        self.find_tumors_btn.setStyleSheet("background: #27ae60; color: white; font-size: 16px;")
        self.find_tumors_btn.setFixedSize(200, 30)
        self.find_tumors_btn.clicked.connect(self.find_tumors)
        self.find_tumors_btn.setVisible(False)
        self.main_layout.addWidget(self.find_tumors_btn, alignment=Qt.AlignCenter)

        # Toggle for Common False Positive Prevention
        self.false_positive_toggle = QCheckBox("Common False Positive Prevention")
        self.false_positive_toggle.setChecked(True)
        self.false_positive_toggle.setVisible(False)
        self.main_layout.addWidget(self.false_positive_toggle, alignment=Qt.AlignRight)

        self.is_fullscreen = False
        self.original_image = None
        self.loaded_file_path = None

        #confidence slider
        self.confidence_slider_label = QLabel("Confidence: 70%")
        self.confidence_slider_label.setAlignment(Qt.AlignRight)
        self.main_layout.addWidget(self.confidence_slider_label)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(70)  # default 70%
        self.confidence_slider.setTickInterval(5)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        self.main_layout.addWidget(self.confidence_slider)

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True

    def start_window_drag(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def move_window(self, event):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def stop_window_drag(self, event):
        self._drag_pos = None

    def update_confidence_label(self, value):
        self.confidence_slider_label.setText(f"Confidence: {value}%")

    def numpy_to_qpixmap(self, array: np.ndarray) -> QPixmap:
        norm_array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        norm_array = (norm_array * 255).astype(np.uint8)
        h, w = norm_array.shape
        # Convert to bytes
        qimage = QImage(norm_array.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "MAT files (*.mat);;NIfTI files (*.nii *.nii.gz)"
        )
        if file_path:
            self.loaded_file_path = file_path
            self.path_label.setText(file_path)
            self.ground_truth_mask = None
            self.show_gt_btn.setVisible(False)

            if file_path.endswith(".mat"):
                # Load the .mat file using mat73
                mat = mat73.loadmat(file_path)
                cjdata = mat['cjdata']
                # Convert image to float32
                preview_img = np.array(cjdata['image'], dtype=np.float32)
                self.original_image = preview_img

                # Check if tumorMask exists
                if 'tumorMask' in cjdata:
                    self.ground_truth_mask = np.array(cjdata['tumorMask'], dtype=np.float32)
                    self.show_gt_btn.setVisible(True)
            else:
                # For .nii or .nii.gz, you can still use predict_tumor
                pred_mask, _ = predict_tumor(file_path)
                self.original_image = pred_mask  # placeholder preview for NIfTI files

            # Convert to QPixmap and show
            pixmap = self.numpy_to_qpixmap(self.original_image)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

            self.find_tumors_btn.setVisible(True)
            self.false_positive_toggle.setVisible(True)

            # Show the "Find Tumors" button and toggle
            self.find_tumors_btn.setVisible(True)
            self.false_positive_toggle.setVisible(True)

    def find_tumors(self):
        if self.loaded_file_path is None:
            return

        # 1. Get predicted probabilities (0-1), not binary
        pred_mask, _ = predict_tumor(self.loaded_file_path)

        # 2. Apply heatmap if checkbox is checked
        if self.false_positive_toggle.isChecked() and heatmap is not None:
            pred_mask = pred_mask * heatmap
            # Normalize to [0,1] again after multiplication
            pred_mask = pred_mask / (pred_mask.max() + 1e-8)

        # 3. Get threshold from slider (0-100 -> 0.0-1.0)
        threshold = self.confidence_slider.value() / 100.0

        # 4. Reset the image to original first
        base_pixmap = self.numpy_to_qpixmap(self.original_image)
        self.image_label.setPixmap(base_pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 5. Overlay the mask using threshold (mask is still float 0-1)
        overlay_pixmap = self.overlay_mask_on_image(self.original_image, pred_mask, alpha=0.4, threshold=threshold)
        scaled_pixmap = overlay_pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def overlay_mask_on_image(self, image: np.ndarray, mask: np.ndarray, alpha=0.4, threshold=0.7, color=(255,0,0)) -> QPixmap:
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        h, w = img_uint8.shape
        qimage = QImage(img_uint8.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        mask_display = np.where(mask >= threshold, mask, 0)
        mask_uint8 = (mask_display * 255 * alpha).astype(np.uint8)

        mask_rgb = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgb[..., 0] = color[0]
        mask_rgb[..., 1] = color[1]
        mask_rgb[..., 2] = color[2]
        mask_rgb[..., 3] = mask_uint8

        overlay = QImage(mask_rgb.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
        painter = QPainter(pixmap)
        painter.drawImage(0, 0, overlay)
        painter.end()

        return pixmap    

    def show_ground_truth_mask(self):
        if self.ground_truth_mask is not None and self.original_image is not None:
            overlay_pixmap = self.overlay_mask_on_image(
                self.original_image, self.ground_truth_mask, alpha=0.4, threshold=0.5, color=(0, 255, 0)  # Green color for ground truth mask
            )
            scaled_pixmap = overlay_pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            QMessageBox.warning(self, "No Ground Truth", "No ground truth mask available for this image.")


#run dat bih
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
