import cv2
import threading
import time
import os
import numpy as np
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import psutil
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics not available. Using HSV detection only.")

# ==============================
# Configuration and Data Classes
# ==============================
@dataclass
class CameraConfig:
    name: str
    url: str
    enabled: bool = True
    confidence_threshold: float = 0.7
    fire_pixel_threshold: int = 6000
    max_fps: int = 8
    resolution: Tuple[int, int] = (640, 480)

@dataclass
class FireDetection:
    timestamp: datetime
    camera_name: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]
    method: str
    frame_saved: bool = False
    filename: Optional[str] = None

class Config:
    """Configuration management"""
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.captures_dir = self.base_dir / "captures"
        self.logs_dir = self.base_dir / "logs"
        self.model_path = self.base_dir / "fire100epochs.pt"

        # Create directories
        self.captures_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Performance settings
        self.max_queue_size = 10
        self.frame_skip = 2
        self.detection_cooldown = 5.0
        self.max_memory_usage = 80

    def get_camera_configs(self):
        return {
            "office": CameraConfig(
                name="office",
                url="rtsp://adminnew:",
                confidence_threshold=0.8,
                max_fps=8,
                resolution=(640, 480)
            ),
            "cabin": CameraConfig(
                name="cabin", 
                url="rtsp://adminnew:",
                confidence_threshold=0.5,
                max_fps=8,
                resolution=(640, 480)
            ),
            "cabin": CameraConfig(
                name="gate", 
                url="rtsp://",
                confidence_threshold=0.5,
                max_fps=8,
                resolution=(640, 480)
            )
        }

# ==============================
# Logging Setup
# ==============================
def setup_logging():
    log_file = Path("fire_detection.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    cv2.setLogLevel(0)
    return logging.getLogger(__name__)

# ==============================
# Fire Detection Engine
# ==============================
class FireDetectionEngine:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.load_model()

    def load_model(self):
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO not available, using HSV only")
            return
        if self.config.model_path.exists():
            try:
                self.model = YOLO(str(self.config.model_path))
                self.logger.info("YOLO model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            self.logger.warning(f"YOLO model file not found: {self.config.model_path}")
            self.logger.warning("Proceeding with HSV detection only")
            self.model = None

    def detect_fire_hsv(self, frame: np.ndarray, threshold: int):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = [
            ([18, 50, 50], [35, 255, 255]),
            ([0, 50, 50], [18, 255, 255]),
            ([160, 50, 50], [179, 255, 255])
        ]
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        fire_pixels = cv2.countNonZero(combined_mask)
        fire_detected = fire_pixels > threshold
        confidence = min(fire_pixels / (threshold * 2), 1.0)

        bbox = None
        if fire_detected:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                bbox = (x, y, x + w, y + h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"HSV Fire {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return fire_detected, confidence, frame, bbox

    def detect_fire(self, frame: np.ndarray, camera_config: CameraConfig):
        if self.model is not None:
            try:
                results = self.model(frame, conf=camera_config.confidence_threshold, verbose=False)
                fire_detected = False
                max_confidence = 0.0
                bbox = None

                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            if conf >= camera_config.confidence_threshold:
                                fire_detected = True
                                if conf > max_confidence:
                                    max_confidence = conf
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    bbox = (x1, y1, x2, y2)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"YOLO Fire {conf:.2f}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                confidence = max_confidence
                method = "YOLO"
            except Exception as e:
                self.logger.error(f"YOLO detection failed: {e}")
                fire_detected, confidence, frame, bbox = self.detect_fire_hsv(frame, camera_config.fire_pixel_threshold)
                method = "HSV"
        else:
            fire_detected, confidence, frame, bbox = self.detect_fire_hsv(frame, camera_config.fire_pixel_threshold)
            method = "HSV"

        detection = None
        if fire_detected:
            detection = FireDetection(
                timestamp=datetime.now(),
                camera_name=camera_config.name,
                confidence=confidence,
                bbox=bbox,
                method=method
            )
        return fire_detected, detection, frame

# ==============================
# Camera Worker
# ==============================
class CameraWorker:
    def __init__(self, camera_config: CameraConfig, config: Config, detection_engine: FireDetectionEngine, logger: logging.Logger):
        self.camera_config = camera_config
        self.config = config
        self.detection_engine = detection_engine
        self.logger = logger

        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_detection_time = 0
        self.fps_counter = deque(maxlen=30)
        self.frame_queue = deque(maxlen=10)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.camera_config.url)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open camera {self.camera_config.name}")
            return False
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.max_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
        self.logger.info(f"Camera {self.camera_config.name} initialized")
        return True

    def capture_frames(self):
        skip_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            skip_count += 1
            if skip_count < self.config.frame_skip:
                continue
            skip_count = 0
            self.frame_queue.append(frame.copy())

    def process_frames(self):
        while self.running:
            if not self.frame_queue:
                time.sleep(0.01)
                continue
            frame = self.frame_queue.popleft()
            self.frame_count += 1
            fire_detected, detection, processed_frame = self.detection_engine.detect_fire(frame, self.camera_config)

            if fire_detected and detection and time.time() - self.last_detection_time > self.config.detection_cooldown:
                self.save_frame(detection, processed_frame)
                self.last_detection_time = time.time()

            fps = self.calculate_fps()
            info_text = [f"Camera: {self.camera_config.name}", f"FPS: {fps:.1f}", f"Frame: {self.frame_count}", f"Status: {'FIRE!' if fire_detected else 'OK'}"]
            for i, text in enumerate(info_text):
                color = (0, 0, 255) if fire_detected and i == 3 else (0, 255, 0)
                cv2.putText(processed_frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(self.camera_config.name, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def save_frame(self, detection: FireDetection, frame):
        camera_dir = self.config.captures_dir / self.camera_config.name
        camera_dir.mkdir(exist_ok=True)
        timestamp_str = detection.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.camera_config.name}_{timestamp_str}.jpg"
        filepath = camera_dir / filename
        if cv2.imwrite(str(filepath), frame):
            detection.frame_saved = True
            detection.filename = str(filepath)
            self.logger.info(f"[FIRE DETECTED] {self.camera_config.name} - Saved: {filename}")

    def calculate_fps(self):
        now = time.time()
        self.fps_counter.append(now)
        if len(self.fps_counter) < 2:
            return 0.0
        return len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])

    def start(self):
        if not self.setup_camera():
            return False
        self.running = True
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyWindow(self.camera_config.name)
        self.logger.info(f"Camera {self.camera_config.name} stopped")

# ==============================
# Fire Detection System
# ==============================
class FireDetectionSystem:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.detection_engine = FireDetectionEngine(self.config, self.logger)
        self.workers = []

    def start(self):
        self.logger.info("Starting Fire Detection System")
        for name, cam_cfg in self.config.get_camera_configs().items():
            if not cam_cfg.enabled:
                continue
            worker = CameraWorker(cam_cfg, self.config, self.detection_engine, self.logger)
            if worker.start():
                self.workers.append(worker)
                self.logger.info(f"Started camera worker: {name}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.logger.info("Stopping Fire Detection System")
        for worker in self.workers:
            worker.stop()
        cv2.destroyAllWindows()
        self.logger.info("Fire Detection System stopped")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    system = FireDetectionSystem()
    system.start()
