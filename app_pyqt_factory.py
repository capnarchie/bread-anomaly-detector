"""
Bread Anomaly Detector - Factory Touchscreen Interface
Simplified touch-friendly GUI for factory line deployment
"""
import sys
import os
import cv2
import numpy as np
import torch
from collections import deque
import queue
import threading
import time

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QGroupBox, QGridLayout, QTextEdit,
                             QFileDialog, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

from pypylon import pylon
from ultralytics import FastSAM

# Import custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam import centroid_tracker as ct
from sam import validator as v


class CameraThread(QThread):
    """Thread for camera/video processing"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    relay_activated = pyqtSignal(int, str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.use_camera = True
        self.video_path = None
        self.video_capture = None
        
        # Model paths
        self.sam_model_path = None
        self.crack_model_path = None
        self.template_path = None
        
        # Detection parameters (hardcoded)
        self.shape_threshold = 0.15
        self.crack_threshold = 5
        self.confidence = 0.50
        
        # Reject zone parameters (hardcoded)
        self.reject_zone_x = 900
        self.reject_zone_y = 175
        self.reject_zone_width = 280
        self.reject_zone_height = 725
        
        # Models
        self.sam_model = None
        self.crack_model = None
        self.template_contour = None
        
        # Frame processing
        self.frame_count = 0
        self.processing_enabled = True
        
    def run(self):
        """Main thread execution"""
        self.running = True
        
        # Load models
        try:
            self.load_models()
        except Exception as e:
            self.error_occurred.emit(f"Error loading models: {str(e)}")
            return
        
        # Initialize camera or video
        if self.use_camera:
            try:
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                camera.Open()
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                
                while self.running:
                    if camera.IsGrabbing():
                        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            image = converter.Convert(grab_result)
                            frame = image.GetArray()
                            self.process_frame(frame)
                        grab_result.Release()
                    
                camera.StopGrabbing()
                camera.Close()
            except Exception as e:
                self.error_occurred.emit(f"Camera error: {str(e)}")
        else:
            # Video file processing
            if not self.video_path:
                self.error_occurred.emit("No video file selected")
                return
                
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                self.error_occurred.emit(f"Cannot open video: {self.video_path}")
                return
            
            while self.running:
                ret, frame = self.video_capture.read()
                if not ret:
                    # Loop video
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                self.process_frame(frame)
                time.sleep(0.033)  # ~30 fps
            
            self.video_capture.release()
    
    def load_models(self):
        """Load AI models and template"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load SAM model
        if self.sam_model_path and os.path.exists(self.sam_model_path):
            self.sam_model = FastSAM(self.sam_model_path)
            self.sam_model.to(device)
        
        # Load crack detection model
        if self.crack_model_path and os.path.exists(self.crack_model_path):
            self.crack_model = FastSAM(self.crack_model_path)
            self.crack_model.to(device)
        
        # Load template
        if self.template_path and os.path.exists(self.template_path):
            template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                _, binary = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    self.template_contour = max(contours, key=cv2.contourArea)
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        stats = {
            'total_detected': 0,
            'normal': 0,
            'defects': 0,
            'rejected': 0
        }
        
        try:
            # SAM segmentation
            if self.sam_model:
                results = self.sam_model(frame, device='cuda' if torch.cuda.is_available() else 'cpu', 
                                       retina_masks=True, imgsz=1024, conf=self.confidence, iou=0.9)
                
                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        
                        for mask in masks:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if not contours:
                                continue
                            
                            contour = max(contours, key=cv2.contourArea)
                            stats['total_detected'] += 1
                            
                            # Shape matching
                            shape_defect = False
                            if self.template_contour is not None:
                                match_value = cv2.matchShapes(contour, self.template_contour, 
                                                             cv2.CONTOURS_MATCH_I1, 0)
                                if match_value > self.shape_threshold:
                                    shape_defect = True
                            
                            # Crack detection
                            crack_defect = False
                            if self.crack_model:
                                x, y, w, h = cv2.boundingRect(contour)
                                roi = frame[y:y+h, x:x+w]
                                if roi.size > 0:
                                    crack_results = self.crack_model(roi, device='cuda' if torch.cuda.is_available() else 'cpu',
                                                                    conf=self.confidence)
                                    for crack_result in crack_results:
                                        if crack_result.boxes is not None and len(crack_result.boxes) > 0:
                                            crack_defect = True
                            
                            # Check if in reject zone
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                in_reject_zone = (self.reject_zone_x <= cx <= self.reject_zone_x + self.reject_zone_width and
                                                self.reject_zone_y <= cy <= self.reject_zone_y + self.reject_zone_height)
                                
                                if shape_defect or crack_defect:
                                    stats['defects'] += 1
                                    if in_reject_zone:
                                        stats['rejected'] += 1
                                        # Activate relay (relay 1 for defects)
                                        self.relay_activated.emit(1, "Defect detected in reject zone")
                                else:
                                    stats['normal'] += 1
            
            # Emit frame with stats (no video display needed)
            self.frame_ready.emit(frame, stats)
            
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        if self.video_capture:
            self.video_capture.release()


class MainWindow(QMainWindow):
    """Main application window - Factory touchscreen interface"""
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Kvaliteedikontroll")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Model Configuration
        model_group = QGroupBox()
        model_main_layout = QVBoxLayout()
        
        # SAM Model selection (crack model auto-loaded based on SAM model name)
        # NAMING SCHEME: If SAM model is "model_name.pt", crack model should be "model_name_crack.pt" in same folder
        # Example: FastSAM-s.pt -> FastSAM-s_crack.pt
        self.sam_button = QPushButton("VALI LEIVA TOOTE MUDEL")
        self.sam_button.setMinimumHeight(100)
        self.sam_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.sam_button.clicked.connect(self.select_sam_model)
        model_main_layout.addWidget(self.sam_button)
        
        self.sam_label = QLabel("Mudelit pole valitud")
        self.sam_label.setStyleSheet("color: gray; font-size: 14px;")
        self.sam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_main_layout.addWidget(self.sam_label)
        
        self.crack_model_label = QLabel("Lõhe mudelit pole laetud")
        self.crack_model_label.setStyleSheet("color: gray; font-size: 12px; font-style: italic;")
        self.crack_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_main_layout.addWidget(self.crack_model_label)
        

        model_group.setLayout(model_main_layout)
        main_layout.addWidget(model_group)
        
        # Defect Parameters
        params_group = QGroupBox()
        params_layout = QVBoxLayout()
        params_layout.setSpacing(15)
        
        # Shape threshold
        self.shape_label = QLabel("Kuju erinevus: 0.15")
        self.shape_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        params_layout.addWidget(self.shape_label)
        
        self.shape_slider = QSlider(Qt.Orientation.Horizontal)
        self.shape_slider.setMinimum(1)
        self.shape_slider.setMaximum(100)
        self.shape_slider.setValue(15)
        self.shape_slider.setMinimumHeight(60)
        self.shape_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 15px;
                background: #bdc3c7;
                border-radius: 8px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 35px;
                height: 35px;
                margin: -15px 0;
                border-radius: 18px;
            }
        """)
        self.shape_slider.valueChanged.connect(self.update_shape_param)
        params_layout.addWidget(self.shape_slider)
        
        # Crack threshold
        self.crack_label = QLabel("Lõhe pindala leivast: 5%")
        self.crack_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        params_layout.addWidget(self.crack_label)
        
        self.crack_slider = QSlider(Qt.Orientation.Horizontal)
        self.crack_slider.setMinimum(0)
        self.crack_slider.setMaximum(50)
        self.crack_slider.setValue(5)
        self.crack_slider.setMinimumHeight(60)
        self.crack_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 15px;
                background: #bdc3c7;
                border-radius: 8px;
            }
            QSlider::handle:horizontal {
                background: #e74c3c;
                width: 35px;
                height: 35px;
                margin: -15px 0;
                border-radius: 18px;
            }
        """)
        self.crack_slider.valueChanged.connect(self.update_crack_param)
        params_layout.addWidget(self.crack_slider)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        self.start_button = QPushButton("START")
        self.start_button.setMinimumHeight(100)
        self.start_button.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                font-weight: bold;
                background-color: #27ae60;
                color: white;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #27ae60;
                opacity: 0.6;
            }
        """)
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("STOP")
        self.stop_button.setMinimumHeight(100)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                font-weight: bold;
                background-color: #e74c3c;
                color: white;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #e74c3c;
                opacity: 0.6;
            }
        """)
        self.stop_button.clicked.connect(self.stop_detection)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.statusBar().showMessage("Valmis")
        self.statusBar().setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
    
    def select_sam_model(self):
        """Select SAM model file and auto-load corresponding crack model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM Model", "", "Model Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.sam_model_path = file_path
            model_name = os.path.basename(file_path)
            self.sam_label.setText(f"Valitud: {model_name}")
            self.sam_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
            
            # Auto-load crack model based on SAM model name
            # Naming scheme: model_name.pt -> model_name_crack.pt
            base_name = os.path.splitext(file_path)[0]  # Remove extension
            crack_path = f"{base_name}_crack.pt"
            
            if os.path.exists(crack_path):
                self.crack_model_path = crack_path
                crack_name = os.path.basename(crack_path)
                self.crack_model_label.setText(f"Automaatselt valitud: {crack_name}")
                self.crack_model_label.setStyleSheet("color: green; font-size: 12px; font-style: italic;")
                self.statusBar().showMessage(f"SAM: {model_name} | Crack: {crack_name}")
            else:
                self.crack_model_path = None
                self.crack_model_label.setText(f"Lõhe mudelit ei leitud ({os.path.basename(crack_path)})")
                self.crack_model_label.setStyleSheet("color: orange; font-size: 12px; font-style: italic;")
                self.statusBar().showMessage(f"Leiva mudel laetud: {model_name} (lõhe mudelit ei leitud)")

    def update_shape_param(self, value):
        """Update shape parameter value"""
        self.shape_label.setText(f"Kuju erinevus: {value/100:.2f}")
        if self.camera_thread:
            self.camera_thread.shape_threshold = value / 100.0
    
    def update_crack_param(self, value):
        """Update crack parameter value"""
        self.crack_label.setText(f"Lõhe pindala leivast: {value}%")
        if self.camera_thread:
            self.camera_thread.crack_threshold = value
    
    def start_detection(self):
        """Start detection system"""
        # Validate that at least one model is loaded
        if not hasattr(self, 'sam_model_path') and not hasattr(self, 'crack_model_path'):
            self.handle_error("Please load at least one model before starting detection")
            return
        
        if not getattr(self, 'sam_model_path', None) and not getattr(self, 'crack_model_path', None):
            self.handle_error("Please load at least one model before starting detection")
            return
        
        self.statusBar().showMessage("Starting detection system...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Create and start camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_stats)
        self.camera_thread.error_occurred.connect(self.handle_error)
        self.camera_thread.relay_activated.connect(self.log_relay_activation)
        
        # Set to use camera (factory deployment)
        self.camera_thread.use_camera = True
        
        # Set model paths
        self.camera_thread.sam_model_path = getattr(self, 'sam_model_path', None)
        self.camera_thread.crack_model_path = getattr(self, 'crack_model_path', None)
        self.camera_thread.template_path = None  # No template support
        
        # Set parameters from UI
        self.camera_thread.shape_threshold = self.shape_slider.value() / 100.0
        self.camera_thread.crack_threshold = self.crack_slider.value()
        
        self.camera_thread.start()
        
        self.statusBar().showMessage("Detection system running on Basler Camera")
    
    def stop_detection(self):
        """Stop detection system"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Detection stopped")
    
    def update_stats(self, frame, stats):
        """Update statistics display"""
        # Just update status bar with stats
        status = f"Detected: {stats['total_detected']} | Normal: {stats['normal']} | Defects: {stats['defects']} | Rejected: {stats['rejected']}"
        self.statusBar().showMessage(status)
    
    def log_relay_activation(self, relay_num, reason):
        """Log relay activation event"""
        # Just show in status bar
        self.statusBar().showMessage(f"🔴 RELAY {relay_num} ACTIVATED - {reason}")
    
    def handle_error(self, error_message):
        """Handle error messages"""
        self.statusBar().showMessage(f"❌ ERROR: {error_message}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
