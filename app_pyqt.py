"""
Bread Anomaly Detector - PyQt6 Desktop Application
Professional desktop GUI for bread quality inspection system
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
                             QGroupBox, QGridLayout, QLCDNumber, QTextEdit,
                             QTabWidget, QCheckBox, QDoubleSpinBox, QFileDialog,
                             QComboBox, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

from pypylon import pylon
from ultralytics import FastSAM

# Import custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam import centroid_tracker as ct
from sam import validator as v
#from relay import on_relay, off_relay, off_all, close_device


class CameraThread(QThread):
    """Thread for camera capture and processing"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    relay_activated = pyqtSignal(int, str)  # Signal for relay activation (bread_id, reason)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.video_capture = None
        self.converter = None
        self.sam_model = None
        self.sam_model_cracks = None
        self.template_contour = None
        self.tracker = None
        self.validator = None
        
        # Input source settings
        self.use_camera = True  # True for Basler camera, False for video file
        self.video_path = None
        
        # Model paths
        self.sam_model_path = None
        self.crack_model_path = None
        
        # Reject zone parameters
        self.reject_zone_x = 1060
        self.reject_zone_y = 175
        self.reject_zone_width = 140
        self.reject_zone_height = 725
        
        # Parameters
        self.shape_threshold = 0.15
        self.crack_threshold = 5
        self.confidence = 0.8
        self.relay_enabled = True
        self.relay_duration = 0.1
        
        # Statistics
        self.stats = {
            'fps': 0.0,
            'total_breads': 0,
            'defective': 0,
            'shape_defects': 0,
            'crack_defects': 0,
            'relay_activations': 0,
            'cracks_detected': 0
        }
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        
        # Relay control
        self.relay_queue = queue.Queue()
        self.relay_active = False
        self.relay_lock = threading.Lock()
        self.activated_bread_ids = set()
        
    def initialize_camera(self):
        """Initialize Basler camera or video file"""
        try:
            if self.use_camera:
                # Use Basler camera
                self.camera = pylon.InstantCamera(
                    pylon.TlFactory.GetInstance().CreateFirstDevice()
                )
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                
                self.converter = pylon.ImageFormatConverter()
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            else:
                # Use video file
                if not self.video_path:
                    self.error_occurred.emit("No video file selected")
                    return False
                self.video_capture = cv2.VideoCapture(self.video_path)
                if not self.video_capture.isOpened():
                    self.error_occurred.emit(f"Failed to open video: {self.video_path}")
                    return False
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Initialization error: {str(e)}")
            return False
    
    def load_models(self):
        """Load AI models based on selection"""
        try:
            # Load FastSAM model if path provided
            if self.sam_model_path:
                self.sam_model = FastSAM(self.sam_model_path)
                self.sam_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load crack detection model if path provided
            if self.crack_model_path:
                self.sam_model_cracks = FastSAM(self.crack_model_path)
                self.sam_model_cracks.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load template
            template_img = cv2.imread("./output_frames/wide_template.jpg")
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.template_contour = max(contours[0], key=cv2.contourArea)
            
            # Initialize tracker and validator
            self.tracker = ct.SimpleBreadTracker()
            self.validator = v.SimpleValidator()
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Model loading error: {str(e)}")
            return False
    
    def relay_worker(self):
        """Worker for relay control"""
        while self.running:
            try:
                relay_number, duration, bread_id = self.relay_queue.get(timeout=1)
                with self.relay_lock:
                    if self.relay_active:
                        self.relay_queue.task_done()
                        continue
                    self.relay_active = True
                
                try:
                    time.sleep(0.1)
                    if self.relay_enabled:
                        pass
                        #on_relay(relay_number)
                    time.sleep(duration)
                    if self.relay_enabled:
                        pass
                        #off_relay(relay_number)
                    self.stats['relay_activations'] += 1
                    
                    # Emit signal for logging
                    self.relay_activated.emit(bread_id, f"Defective bread #{bread_id} rejected")
                finally:
                    with self.relay_lock:
                        self.relay_active = False
                    self.relay_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Relay error: {e}")
    
    def activate_relay(self, bread_id=None):
        """Queue relay activation"""
        if bread_id is not None:
            if bread_id in self.activated_bread_ids:
                return
            self.activated_bread_ids.add(bread_id)
        
        with self.relay_lock:
            if self.relay_active:
                return
        self.relay_queue.put((1, self.relay_duration, bread_id if bread_id is not None else 0))
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_crack_in_bread(self, crack_box, bread_box):
        """Check if crack overlaps with bread"""
        iou = self.calculate_iou(crack_box, bread_box)
        
        crack_center_x = (crack_box[0] + crack_box[2]) / 2
        crack_center_y = (crack_box[1] + crack_box[3]) / 2
        
        x1, y1, x2, y2 = bread_box
        center_in_bread = (x1 <= crack_center_x <= x2) and (y1 <= crack_center_y <= y2)
        
        return iou > 0.3 or center_in_bread
    
    def run(self):
        """Main processing loop"""
        self.running = True
        
        # Start relay worker thread
        relay_thread = threading.Thread(target=self.relay_worker, daemon=True)
        relay_thread.start()
        
        if not self.initialize_camera():
            return
        
        if not self.load_models():
            if self.camera:
                self.camera.StopGrabbing()
            if self.video_capture:
                self.video_capture.release()
            return
        
        try:
            while self.running:
                frame = None
                
                # Get frame from camera or video
                if self.use_camera and self.camera and self.camera.IsGrabbing():
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded():
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray()
                    grabResult.Release()
                elif not self.use_camera and self.video_capture:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        # Loop video
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                
                if frame is None:
                    continue
                
                start_time = time.time()
                
                height, width, _ = frame.shape
                
                # Define zones
                center_x, center_y = width // 2, height // 2
                x1_val = int(center_x - 0.1 * width)
                x2_val = int(center_x + 0.1 * width)
                y1_val = int(center_y - 0.35 * height)
                y2_val = int(center_y + 0.35 * height)
                
                # Run detection based on model selection
                results = None
                crack_results = None
                
                if self.sam_model:
                    results = self.sam_model.predict(frame, imgsz=640, verbose=False,
                                                     conf=self.confidence, retina_masks=True)
                
                if self.sam_model_cracks:
                    crack_results = self.sam_model_cracks.predict(frame, imgsz=640, verbose=False,
                                                                   conf=0.3, retina_masks=True)
                    
                    # Extract bread detections
                    detections = []
                    bread_masks = []
                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.astype(int)
                            detections.append((x1, y1, x2, y2))
                            bread_masks.append(masks[idx] if idx < len(masks) else None)
                    
                    # Extract crack detections
                    crack_detections = []
                    crack_masks = []
                    if crack_results and crack_results[0].boxes is not None:
                        crack_boxes = crack_results[0].boxes.xyxy.cpu().numpy()
                        crack_mask_data = crack_results[0].masks.data.cpu().numpy() if crack_results[0].masks is not None else []
                        for idx, box in enumerate(crack_boxes):
                            x1, y1, x2, y2 = box.astype(int)
                            crack_detections.append((x1, y1, x2, y2))
                            crack_masks.append(crack_mask_data[idx] if idx < len(crack_mask_data) else None)
                    
                    # Update tracker
                    tracked_breads = self.tracker.update(detections)
                    
                    # Analyze each bread
                    for bread_id, bread_data in tracked_breads.items():
                        bread_box = bread_data['box']
                        bread_mask = None
                        
                        # Find corresponding mask
                        for idx, det_box in enumerate(detections):
                            if det_box == bread_box and idx < len(bread_masks):
                                bread_mask = bread_masks[idx]
                                break
                        
                        if bread_mask is None:
                            bread_data['crack_percentage'] = 0.0
                            bread_data['shape_defective'] = False
                            bread_data['crack_defective'] = False
                            bread_data['is_defective'] = False
                            bread_data['match_score'] = 0.0
                            continue
                        
                        # Calculate bread area and contour
                        bread_binary_mask = (bread_mask > 0.5).astype(np.uint8)
                        bread_area = np.sum(bread_binary_mask)
                        
                        bread_binary_mask_255 = (bread_binary_mask * 255).astype(np.uint8)
                        bread_contours, _ = cv2.findContours(bread_binary_mask_255, 
                                                             cv2.RETR_EXTERNAL, 
                                                             cv2.CHAIN_APPROX_SIMPLE)
                        bread_contour = max(bread_contours, key=cv2.contourArea) if bread_contours else None
                        
                        # Shape matching
                        shape_defective = False
                        match_score = 0.0
                        if bread_contour is not None and len(bread_contour) > 5:
                            match_score = cv2.matchShapes(self.template_contour, bread_contour, 
                                                          cv2.CONTOURS_MATCH_I1, 0.0)
                            shape_defective = match_score > self.shape_threshold
                        
                        # Find cracks in this bread
                        total_crack_area = 0
                        for crack_idx, crack_box in enumerate(crack_detections):
                            if self.is_crack_in_bread(crack_box, bread_box):
                                if crack_idx < len(crack_masks) and crack_masks[crack_idx] is not None:
                                    crack_binary_mask = (crack_masks[crack_idx] > 0.5).astype(np.uint8)
                                    crack_area = np.sum(crack_binary_mask)
                                    total_crack_area += crack_area
                                    
                                    # Draw crack on frame
                                    x1_c, y1_c, x2_c, y2_c = crack_box
                                    cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 255), 1)
                        
                        # Calculate crack percentage
                        crack_percentage = (total_crack_area / bread_area) * 100.0 if bread_area > 0 else 0.0
                        crack_defective = crack_percentage > self.crack_threshold
                        
                        # Update bread data
                        bread_data['crack_percentage'] = crack_percentage
                        bread_data['match_score'] = match_score
                        bread_data['shape_defective'] = shape_defective
                        bread_data['crack_defective'] = crack_defective
                        bread_data['is_defective'] = shape_defective or crack_defective
                    
                    # Draw tracking results
                    for bread_id, bread_data in tracked_breads.items():
                        x1, y1, x2, y2 = bread_data['box']
                        center = bread_data['center']
                        is_defective = bread_data.get('is_defective', False)
                        shape_defective = bread_data.get('shape_defective', False)
                        crack_defective = bread_data.get('crack_defective', False)
                        crack_percentage = bread_data.get('crack_percentage', 0.0)
                        match_score = bread_data.get('match_score', 0.0)
                        
                        # Color coding
                        color = (0, 0, 255) if is_defective else (0, 255, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, center, 5, color, -1)
                        
                        # Label
                        label = f"Bread_{bread_id}"
                        if shape_defective:
                            label += f" | Shape:{match_score:.2f}"
                        if crack_defective:
                            label += f" | Crack:{crack_percentage:.1f}%"
                        if is_defective:
                            label += " [REJECT]"
                        
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Validation zone logic
                        in_validation_zone = (x1_val <= center[0] <= x2_val) and \
                                            (y1_val <= center[1] <= y2_val)
                        
                        # Calculate reject zone boundaries
                        reject_x1 = self.reject_zone_x
                        reject_y1 = self.reject_zone_y
                        reject_x2 = self.reject_zone_x + self.reject_zone_width
                        reject_y2 = self.reject_zone_y + self.reject_zone_height
                        
                        in_exit_zone = (reject_x1 <= center[0] <= reject_x2) and (reject_y1 <= center[1] <= reject_y2)
                        
                        if in_validation_zone:
                            self.validator.add_detection(is_defective, bread_id)
                        
                        if in_exit_zone and self.validator.is_valid(bread_id):
                            self.validator.add_exit_detection(bread_id)
                            if self.validator.can_exit_detection(bread_id):
                                self.activate_relay(bread_id)
                            cv2.putText(frame, "REJECT", (x1, y2 + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    
                    # Draw zones
                    cv2.rectangle(frame, (x1_val, y1_val), (x2_val, y2_val), (255, 0, 0), 2)
                    cv2.putText(frame, "Validation", (x1_val, y1_val - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Draw reject zone
                    reject_x1 = self.reject_zone_x
                    reject_y1 = self.reject_zone_y
                    reject_x2 = self.reject_zone_x + self.reject_zone_width
                    reject_y2 = self.reject_zone_y + self.reject_zone_height
                    cv2.rectangle(frame, (reject_x1, reject_y1), (reject_x2, reject_y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Reject", (reject_x1, reject_y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Update statistics
                    fps = 1.0 / (time.time() - start_time)
                    self.fps_counter.append(fps)
                    self.stats['fps'] = sum(self.fps_counter) / len(self.fps_counter)
                    self.stats['total_breads'] = len(tracked_breads)
                    self.stats['defective'] = sum(1 for b in tracked_breads.values() 
                                                  if b.get('is_defective', False))
                    self.stats['shape_defects'] = sum(1 for b in tracked_breads.values() 
                                                      if b.get('shape_defective', False))
                    self.stats['crack_defects'] = sum(1 for b in tracked_breads.values() 
                                                      if b.get('crack_defective', False))
                    self.stats['cracks_detected'] = len(crack_detections)
                    
                    # Emit frame and stats
                    self.frame_ready.emit(frame, self.stats)
                
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
        finally:
            if self.camera:
                self.camera.StopGrabbing()
            if self.video_capture:
                self.video_capture.release()
            #off_all()
            #close_device()
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        if self.video_capture:
            self.video_capture.release()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("🍞 Bread Anomaly Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Video feed
        video_group = QGroupBox("Live Camera Feed")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #555;")
        video_layout.addWidget(self.video_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶️ Start Detection")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ Stop Detection")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-size: 14px; padding: 10px;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        button_layout.addWidget(self.stop_button)
        
        video_layout.addLayout(button_layout)
        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group, stretch=3)
        
        # Right side - Controls and statistics
        right_layout = QVBoxLayout()
        
        # Input Source Selection
        source_group = QGroupBox("📹 Input Source")
        source_layout = QVBoxLayout()
        
        self.source_button_group = QButtonGroup()
        self.camera_radio = QRadioButton("Basler Camera")
        self.video_radio = QRadioButton("Video File")
        self.camera_radio.setChecked(True)
        self.source_button_group.addButton(self.camera_radio)
        self.source_button_group.addButton(self.video_radio)
        
        source_layout.addWidget(self.camera_radio)
        source_layout.addWidget(self.video_radio)
        
        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.select_video_button = QPushButton("Select Video File")
        self.select_video_button.clicked.connect(self.select_video_file)
        self.select_video_button.setEnabled(False)
        
        self.video_radio.toggled.connect(lambda checked: self.select_video_button.setEnabled(checked))
        
        source_layout.addWidget(self.select_video_button)
        source_layout.addWidget(self.video_path_label)
        
        source_group.setLayout(source_layout)
        right_layout.addWidget(source_group)
        
        # Model Selection
        model_group = QGroupBox("🤖 Model Selection")
        model_layout = QVBoxLayout()
        
        # SAM Model
        sam_layout = QHBoxLayout()
        self.sam_model_button = QPushButton("Load SAM Model")
        self.sam_model_button.clicked.connect(self.select_sam_model)
        sam_layout.addWidget(QLabel("SAM Model:"))
        sam_layout.addWidget(self.sam_model_button)
        model_layout.addLayout(sam_layout)
        
        self.sam_model_label = QLabel("No model loaded")
        self.sam_model_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.sam_model_label)
        
        # Crack Model
        crack_layout = QHBoxLayout()
        self.crack_model_button = QPushButton("Load Crack Model")
        self.crack_model_button.clicked.connect(self.select_crack_model)
        crack_layout.addWidget(QLabel("Crack Model:"))
        crack_layout.addWidget(self.crack_model_button)
        model_layout.addLayout(crack_layout)
        
        self.crack_model_label = QLabel("No model loaded")
        self.crack_model_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.crack_model_label)
        
        # GPU info
        gpu_info = "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Device: CPU"
        self.gpu_label = QLabel(gpu_info)
        self.gpu_label.setStyleSheet("color: green; font-weight: bold;" if torch.cuda.is_available() else "color: orange;")
        model_layout.addWidget(self.gpu_label)
        
        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)
        
        # Zone Configuration
        zone_group = QGroupBox("� Reject Zone Configuration")
        zone_layout = QVBoxLayout()
        
        # Reject zone X position
        reject_x_label = QLabel("Reject Zone X: 1060")
        self.reject_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.reject_x_slider.setMinimum(0)
        self.reject_x_slider.setMaximum(1440)
        self.reject_x_slider.setValue(1060)
        self.reject_x_slider.valueChanged.connect(
            lambda v: self.update_zone_param('reject_x', v, reject_x_label)
        )
        zone_layout.addWidget(reject_x_label)
        zone_layout.addWidget(self.reject_x_slider)
        
        # Reject zone Y position
        reject_y_label = QLabel("Reject Zone Y: 175")
        self.reject_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.reject_y_slider.setMinimum(0)
        self.reject_y_slider.setMaximum(1080)
        self.reject_y_slider.setValue(175)
        self.reject_y_slider.valueChanged.connect(
            lambda v: self.update_zone_param('reject_y', v, reject_y_label)
        )
        zone_layout.addWidget(reject_y_label)
        zone_layout.addWidget(self.reject_y_slider)
        
        # Reject zone width
        reject_width_label = QLabel("Reject Zone Width: 140")
        self.reject_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.reject_width_slider.setMinimum(50)
        self.reject_width_slider.setMaximum(500)
        self.reject_width_slider.setValue(140)
        self.reject_width_slider.valueChanged.connect(
            lambda v: self.update_zone_param('reject_width', v, reject_width_label)
        )
        zone_layout.addWidget(reject_width_label)
        zone_layout.addWidget(self.reject_width_slider)
        
        # Reject zone height
        reject_height_label = QLabel("Reject Zone Height: 725")
        self.reject_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.reject_height_slider.setMinimum(100)
        self.reject_height_slider.setMaximum(1080)
        self.reject_height_slider.setValue(725)
        self.reject_height_slider.valueChanged.connect(
            lambda v: self.update_zone_param('reject_height', v, reject_height_label)
        )
        zone_layout.addWidget(reject_height_label)
        zone_layout.addWidget(self.reject_height_slider)
        
        zone_group.setLayout(zone_layout)
        right_layout.addWidget(zone_group)
        
        # Detection parameters
        params_group = QGroupBox("🎯 Detection Parameters")
        params_layout = QVBoxLayout()
        
        # Shape threshold
        shape_label = QLabel("Shape Match Threshold: 0.15")
        self.shape_slider = QSlider(Qt.Orientation.Horizontal)
        self.shape_slider.setMinimum(1)
        self.shape_slider.setMaximum(100)
        self.shape_slider.setValue(15)
        self.shape_slider.valueChanged.connect(
            lambda v: self.update_param('shape', v, shape_label)
        )
        params_layout.addWidget(shape_label)
        params_layout.addWidget(self.shape_slider)
        
        # Crack threshold
        crack_label = QLabel("Crack Threshold: 5%")
        self.crack_slider = QSlider(Qt.Orientation.Horizontal)
        self.crack_slider.setMinimum(0)
        self.crack_slider.setMaximum(50)
        self.crack_slider.setValue(5)
        self.crack_slider.valueChanged.connect(
            lambda v: self.update_param('crack', v, crack_label)
        )
        params_layout.addWidget(crack_label)
        params_layout.addWidget(self.crack_slider)
        
        # Confidence threshold
        conf_label = QLabel("Detection Confidence: 0.80")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(80)
        self.conf_slider.valueChanged.connect(
            lambda v: self.update_param('confidence', v, conf_label)
        )
        params_layout.addWidget(conf_label)
        params_layout.addWidget(self.conf_slider)
        
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)
        
        # Relay control
        relay_group = QGroupBox("🔌 Relay Control")
        relay_layout = QVBoxLayout()
        
        self.relay_enabled_check = QCheckBox("Enable Relay")
        self.relay_enabled_check.setChecked(True)
        relay_layout.addWidget(self.relay_enabled_check)
        
        relay_duration_layout = QHBoxLayout()
        relay_duration_layout.addWidget(QLabel("Duration (s):"))
        self.relay_duration_spin = QDoubleSpinBox()
        self.relay_duration_spin.setMinimum(0.05)
        self.relay_duration_spin.setMaximum(1.0)
        self.relay_duration_spin.setSingleStep(0.05)
        self.relay_duration_spin.setValue(0.1)
        relay_duration_layout.addWidget(self.relay_duration_spin)
        relay_layout.addLayout(relay_duration_layout)
        
        relay_group.setLayout(relay_layout)
        right_layout.addWidget(relay_group)
        
        # Log panel
        log_group = QGroupBox("📝 Event Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Add stretch to push everything to top
        right_layout.addStretch()
        
        main_layout.addLayout(right_layout, stretch=1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click 'Start Detection' to begin")
        
        # Log startup
        self.log_message("Application initialized")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.log_message(f"Using device: {device}")
    
    def update_param(self, param_type, value, label):
        """Update parameter value"""
        if self.camera_thread:
            if param_type == 'shape':
                self.camera_thread.shape_threshold = value / 100.0
                label.setText(f"Shape Match Threshold: {value/100:.2f}")
            elif param_type == 'crack':
                self.camera_thread.crack_threshold = value
                label.setText(f"Crack Threshold: {value}%")
            elif param_type == 'confidence':
                self.camera_thread.confidence = value / 100.0
                label.setText(f"Detection Confidence: {value/100:.2f}")
    
    def update_zone_param(self, param_type, value, label):
        """Update zone parameter value"""
        if self.camera_thread:
            if param_type == 'reject_x':
                self.camera_thread.reject_zone_x = value
                label.setText(f"Reject Zone X: {value}")
            elif param_type == 'reject_y':
                self.camera_thread.reject_zone_y = value
                label.setText(f"Reject Zone Y: {value}")
            elif param_type == 'reject_width':
                self.camera_thread.reject_zone_width = value
                label.setText(f"Reject Zone Width: {value}")
            elif param_type == 'reject_height':
                self.camera_thread.reject_zone_height = value
                label.setText(f"Reject Zone Height: {value}")
    
    def start_detection(self):
        """Start detection system"""
        # Validate that at least one model is loaded
        if not hasattr(self, 'sam_model_path') and not hasattr(self, 'crack_model_path'):
            self.handle_error("Please load at least one model before starting detection")
            return
        
        if not getattr(self, 'sam_model_path', None) and not getattr(self, 'crack_model_path', None):
            self.handle_error("Please load at least one model before starting detection")
            return
        
        self.log_message("Starting detection system...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Create and start camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.error_occurred.connect(self.handle_error)
        self.camera_thread.relay_activated.connect(self.log_relay_activation)
        
        # Set input source
        self.camera_thread.use_camera = self.camera_radio.isChecked()
        if not self.camera_thread.use_camera:
            if not hasattr(self, 'selected_video_path') or not self.selected_video_path:
                self.handle_error("Please select a video file first")
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
            self.camera_thread.video_path = self.selected_video_path
        
        # Set model paths
        self.camera_thread.sam_model_path = getattr(self, 'sam_model_path', None)
        self.camera_thread.crack_model_path = getattr(self, 'crack_model_path', None)
        
        # Set zone parameters
        self.camera_thread.reject_zone_x = self.reject_x_slider.value()
        self.camera_thread.reject_zone_y = self.reject_y_slider.value()
        self.camera_thread.reject_zone_width = self.reject_width_slider.value()
        self.camera_thread.reject_zone_height = self.reject_height_slider.value()
        
        # Set parameters from UI
        self.camera_thread.shape_threshold = self.shape_slider.value() / 100.0
        self.camera_thread.crack_threshold = self.crack_slider.value()
        self.camera_thread.confidence = self.conf_slider.value() / 100.0
        self.camera_thread.relay_enabled = self.relay_enabled_check.isChecked()
        self.camera_thread.relay_duration = self.relay_duration_spin.value()
        
        self.camera_thread.start()
        
        source_type = "Basler Camera" if self.camera_thread.use_camera else "Video File"
        self.statusBar().showMessage(f"Detection system running on {source_type}...")
        self.log_message(f"System started successfully using {source_type}")
        models = []
        if self.camera_thread.sam_model_path:
            models.append("SAM")
        if self.camera_thread.crack_model_path:
            models.append("Crack Detection")
        self.log_message(f"Active models: {', '.join(models) if models else 'None'}")
    
    def select_sam_model(self):
        """Open file dialog to select SAM model weights"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM Model Weights",
            "",
            "Model Files (*.pt *.pth);;All Files (*.*)"
        )
        if file_path:
            self.sam_model_path = file_path
            filename = file_path.split('/')[-1].split('\\')[-1]
            self.sam_model_label.setText(f"Loaded: {filename}")
            self.sam_model_label.setStyleSheet("color: green; font-size: 10px;")
            self.log_message(f"SAM model loaded: {filename}")
    
    def select_crack_model(self):
        """Open file dialog to select crack detection model weights"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Crack Detection Model Weights",
            "",
            "Model Files (*.pt *.pth);;All Files (*.*)"
        )
        if file_path:
            self.crack_model_path = file_path
            filename = file_path.split('/')[-1].split('\\')[-1]
            self.crack_model_label.setText(f"Loaded: {filename}")
            self.crack_model_label.setStyleSheet("color: green; font-size: 10px;")
            self.log_message(f"Crack model loaded: {filename}")
    
    def select_video_file(self):
        """Open file dialog to select video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.avi *.mp4 *.mkv *.mov);;All Files (*.*)"
        )
        if file_path:
            self.selected_video_path = file_path
            filename = file_path.split('/')[-1].split('\\')[-1]
            self.video_path_label.setText(f"Selected: {filename}")
            self.video_path_label.setStyleSheet("color: green; font-size: 10px;")
            self.log_message(f"Video file selected: {filename}")
    
    def stop_detection(self):
        """Stop detection system"""
        if self.camera_thread:
            self.log_message("Stopping detection system...")
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Detection stopped")
        self.log_message("System stopped")
    
    def update_frame(self, frame, stats):
        """Update video frame and statistics"""
        # Convert frame to QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)
        
        # Update status bar with statistics
        self.statusBar().showMessage(
            f"FPS: {stats['fps']:.1f} | Tracking: {stats['total_breads']} breads | "
            f"Defective: {stats['defective']} | Cracks: {stats['cracks_detected']}"
        )
    
    def handle_error(self, error_msg):
        """Handle errors from camera thread"""
        self.log_message(f"ERROR: {error_msg}")
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.stop_detection()
    
    def log_relay_activation(self, bread_id, reason):
        """Log relay activation events"""
        self.log_message(f"🔴 RELAY ACTIVATED - {reason}")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.camera_thread:
            self.stop_detection()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
