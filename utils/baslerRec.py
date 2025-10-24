import sys
import time
import os
import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from pypylon import pylon

# ======================= CONFIG ===============================
CAMERA_TIMEOUT_MS = 5000
FRAME_DELAY_SEC = 0.005
VIDEO_CODEC = 'XVID'
VIDEO_FPS = 30.0

BASE_OUTPUT_DIR = "output"
SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "snapshots")
RECORDING_DIR = os.path.join(BASE_OUTPUT_DIR, "recordings")

SNAPSHOT_PREFIX = "snapshot_"
RECORDING_PREFIX = "recording_"

WINDOW_TITLE = "Basler video salvestamine"
WINDOW_GEOMETRY = (100, 100, 1400, 600)
FEED_SIZE = (650, 400)
# ==============================================================

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

class CameraThread(QThread):
    frame_received = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.recording = False
        self.take_pic = False
        self.camera = None
        self.converter = None
        self.video_writer = None

    def run(self):
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.is_running = True

            while self.is_running and self.camera.IsGrabbing():
                grab = self.camera.RetrieveResult(CAMERA_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    img = self.converter.Convert(grab).GetArray()
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    q_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
                    self.frame_received.emit(q_img)

                    if self.recording:
                        if self.video_writer is None:
                            filename = os.path.join(
                                RECORDING_DIR,
                                f"{RECORDING_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}.avi"
                            )
                            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                            self.video_writer = cv2.VideoWriter(
                                filename, fourcc, VIDEO_FPS,
                                (rgb.shape[1], rgb.shape[0])
                            )
                        self.video_writer.write(img)

                    if self.take_pic:
                        filename = os.path.join(
                            SNAPSHOT_DIR,
                            f"{SNAPSHOT_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        )
                        cv2.imwrite(filename, img)
                        print(f"Saved snapshot: {filename}")
                        self.take_pic = False

                grab.Release()
                time.sleep(FRAME_DELAY_SEC)

        except Exception as e:
            print("Camera thread error:", e)
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.video_writer:
                self.video_writer.release()
            if self.camera:
                self.camera.Close()
        except Exception as e:
            print("Cleanup error:", e)

    def stop(self):
        self.is_running = False
        self.wait()

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def take_snapshot(self):
        self.take_pic = True

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(*WINDOW_GEOMETRY)
        self.camera_thread = None
        self.camera_active = False
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout(self.central_widget)
        feed_layout = QHBoxLayout()

        self.label1 = QLabel()
        self.label1.setFixedSize(*FEED_SIZE)
        feed_layout.addWidget(self.label1)

        self.label2 = QLabel()
        self.label2.setFixedSize(*FEED_SIZE)
        feed_layout.addWidget(self.label2)
        main_layout.addLayout(feed_layout)

        self.start_feed_btn = QPushButton("Start Feed")
        self.start_feed_btn.clicked.connect(self.start_feed)
        main_layout.addWidget(self.start_feed_btn)

        self.stop_feed_btn = QPushButton("Stop Feed", enabled=False)
        self.stop_feed_btn.clicked.connect(self.stop_feed)
        main_layout.addWidget(self.stop_feed_btn)

        self.start_record_btn = QPushButton("Start Recording", enabled=False)
        self.start_record_btn.clicked.connect(self.start_recording)
        main_layout.addWidget(self.start_record_btn)

        self.stop_record_btn = QPushButton("Stop Recording", enabled=False)
        self.stop_record_btn.clicked.connect(self.stop_recording)
        main_layout.addWidget(self.stop_record_btn)

        self.snapshot_btn = QPushButton("Take Snapshot", enabled=False)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        main_layout.addWidget(self.snapshot_btn)

    def start_feed(self):
        if not self.camera_active:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_received.connect(self.update_frame)
            self.camera_thread.start()
            self.camera_active = True
            self._toggle_buttons(True)

    def stop_feed(self):
        if self.camera_active and self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            self.camera_active = False
            self._toggle_buttons(False)

    def start_recording(self):
        if self.camera_thread and not self.camera_thread.recording:
            self.camera_thread.start_recording()
            self.start_record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)
            print("Recording started...")

    def stop_recording(self):
        if self.camera_thread and self.camera_thread.recording:
            self.camera_thread.stop_recording()
            self.start_record_btn.setEnabled(True)
            self.stop_record_btn.setEnabled(False)
            print("Recording stopped.")

    def take_snapshot(self):
        if self.camera_thread:
            self.camera_thread.take_snapshot()

    def update_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.label1.width(), self.label1.height(), Qt.KeepAspectRatio
        )
        self.label1.setPixmap(pixmap)

        gray = q_img.convertToFormat(QImage.Format_Grayscale8)
        pixmap_gray = QPixmap.fromImage(gray).scaled(
            self.label2.width(), self.label2.height(), Qt.KeepAspectRatio
        )
        self.label2.setPixmap(pixmap_gray)

    def _toggle_buttons(self, active):
        self.start_feed_btn.setEnabled(not active)
        self.stop_feed_btn.setEnabled(active)
        self.start_record_btn.setEnabled(active)
        self.stop_record_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(active)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
