import cv2
import threading
import time
import os
import numpy as np
import logging
import json
import queue
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import sqlite3
from pathlib import Path
import psutil

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
    confidence_threshold: float = 0.6
    fire_pixel_threshold: int = 5000
    max_fps: int = 10
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
        self.db_path = self.base_dir / "fire_detection.db"
        self.model_path = self.base_dir / "fire100epochs.pt"
        
        # Create directories
        self.captures_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Performance settings
        self.max_queue_size = 10
        self.frame_skip = 2  # Process every nth frame
        self.detection_cooldown = 5.0  # seconds between detections per camera
        self.max_memory_usage = 80  # percentage
        
        # OpenCV optimizations
        self.cv_backend = cv2.CAP_FFMPEG
        self.cv_buffer_size = 1
        
    def get_camera_configs(self) -> Dict[str, CameraConfig]:
        """Load camera configurations"""
        return {
            "office": CameraConfig(
                name="office",
                url="",
                confidence_threshold=0.5,
                max_fps=8,
                resolution=(640, 480)
            ),
            "cabin": CameraConfig(
                name="cabin", 
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
    """Setup comprehensive logging"""
    config = Config()
    log_file = config.logs_dir / f"fire_detection_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Suppress verbose OpenCV and HEVC logs
    cv2.setLogLevel(0)  # Suppress OpenCV logs
    
    return logging.getLogger(__name__)

# ==============================
# Database Management
# ==============================
class DatabaseManager:
    """Manage fire detection database"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fire_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER, 
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    filename TEXT,
                    frame_saved BOOLEAN DEFAULT 0
                )
            """)
            conn.commit()
    
    def log_detection(self, detection: FireDetection):
        """Log fire detection to database"""
        with sqlite3.connect(self.db_path) as conn:
            bbox_values = detection.bbox if detection.bbox else (None, None, None, None)
            conn.execute("""
                INSERT INTO fire_detections 
                (timestamp, camera_name, confidence, method, bbox_x1, bbox_y1, bbox_x2, bbox_y2, filename, frame_saved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detection.timestamp.isoformat(),
                detection.camera_name,
                detection.confidence,
                detection.method,
                *bbox_values,
                detection.filename,
                detection.frame_saved
            ))
            conn.commit()

# ==============================
# Fire Detection Engine
# ==============================
class FireDetectionEngine:
    """Advanced fire detection with multiple methods"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model with error handling"""
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO not available, using HSV only")
            return
            
        try:
            if self.config.model_path.exists():
                self.model = YOLO(str(self.config.model_path))
                self.logger.info("YOLO model loaded successfully")
            else:
                self.logger.error(f"Model file not found: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_fire_hsv(self, frame: np.ndarray, threshold: int) -> Tuple[bool, float, np.ndarray, Optional[Tuple]]:
        """HSV-based fire detection with improvements"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple HSV ranges for better detection
        ranges = [
            ([18, 50, 50], [35, 255, 255]),    # Orange-red
            ([0, 50, 50], [18, 255, 255]),     # Red
            ([160, 50, 50], [179, 255, 255])   # Deep red
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        fire_pixels = cv2.countNonZero(combined_mask)
        fire_detected = fire_pixels > threshold
        confidence = min(fire_pixels / (threshold * 2), 1.0)
        
        # Find bounding box of largest contour
        bbox = None
        if fire_detected:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (x, y, x + w, y + h)
                
                # Draw detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"HSV Fire {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return fire_detected, confidence, frame, bbox
    
    def detect_fire_yolo(self, frame: np.ndarray, confidence_threshold: float) -> Tuple[bool, float, np.ndarray, Optional[Tuple]]:
        """YOLO-based fire detection"""
        if self.model is None:
            return False, 0.0, frame, None
        
        try:
            results = self.model(frame, verbose=False, conf=confidence_threshold)
            fire_detected = False
            max_confidence = 0.0
            best_bbox = None
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf >= confidence_threshold:
                            fire_detected = True
                            if conf > max_confidence:
                                max_confidence = conf
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                best_bbox = (x1, y1, x2, y2)
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"YOLO Fire {conf:.2f}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return fire_detected, max_confidence, frame, best_bbox
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return False, 0.0, frame, None
    
    def detect_fire(self, frame: np.ndarray, camera_config: CameraConfig) -> Tuple[bool, FireDetection, np.ndarray]:
        """Main fire detection method"""
        detection = None
        
        # Try YOLO first, fall back to HSV
        if self.model is not None:
            fire_detected, confidence, processed_frame, bbox = self.detect_fire_yolo(
                frame, camera_config.confidence_threshold
            )
            method = "YOLO"
        else:
            fire_detected, confidence, processed_frame, bbox = self.detect_fire_hsv(
                frame, camera_config.fire_pixel_threshold
            )
            method = "HSV"
        
        if fire_detected:
            detection = FireDetection(
                timestamp=datetime.now(),
                camera_name=camera_config.name,
                confidence=confidence,
                bbox=bbox,
                method=method
            )
        
        return fire_detected, detection, processed_frame

# ==============================
# Advanced Camera Worker
# ==============================
class CameraWorker:
    """Advanced camera processing with optimization"""
    def __init__(self, camera_config: CameraConfig, config: Config, 
                 detection_engine: FireDetectionEngine, db_manager: DatabaseManager, 
                 logger: logging.Logger):
        self.camera_config = camera_config
        self.config = config
        self.detection_engine = detection_engine
        self.db_manager = db_manager
        self.logger = logger
        
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_detection_time = 0
        self.fps_counter = deque(maxlen=30)
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.display_queue = queue.Queue(maxsize=5)
        
    def setup_camera(self) -> bool:
        """Setup camera with optimized settings"""
        try:
            # Try different backends for better compatibility
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_config.url, backend)
                if self.cap.isOpened():
                    break
                self.cap.release()
                
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_config.name}")
                return False
            
            # Optimize capture settings
            #self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, self.config.cv_buffer_size)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.max_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
            
            # Set codec preferences to avoid HEVC issues
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            self.logger.info(f"Camera {self.camera_config.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup camera {self.camera_config.name}: {e}")
            return False
    
    def capture_frames(self):
        """Optimized frame capture thread"""
        frame_skip_count = 0
        consecutive_failures = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 100:  # Reset connection after many failures
                        self.logger.warning(f"Too many consecutive failures for {self.camera_config.name}, attempting reconnection")
                        self.cap.release()
                        if not self.setup_camera():
                            self.logger.error(f"Failed to reconnect {self.camera_config.name}")
                            break
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue
                
                consecutive_failures = 0  # Reset failure counter on success
                
                # Skip frames for performance
                frame_skip_count += 1
                if frame_skip_count < self.config.frame_skip:
                    continue
                frame_skip_count = 0
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((time.time(), frame.copy()))
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((time.time(), frame.copy()))
                    except queue.Empty:
                        pass
                
                # Memory management
                try:
                    if psutil.virtual_memory().percent > self.config.max_memory_usage:
                        time.sleep(0.1)
                except:
                    pass  # Ignore psutil errors
                    
            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"Frame capture error for {self.camera_config.name}: {e}")
                if consecutive_failures > 10:
                    time.sleep(1)  # Longer sleep for persistent errors
    
    def process_frames(self):
        """Process frames for fire detection"""
        while self.running:
            try:
                # Get frame from queue with timeout
                try:
                    timestamp, frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Fire detection
                fire_detected, detection, processed_frame = self.detection_engine.detect_fire(
                    frame, self.camera_config
                )
                
                # Handle fire detection
                if fire_detected and detection:
                    # Check cooldown period
                    if current_time - self.last_detection_time > self.config.detection_cooldown:
                        self.handle_fire_detection(detection, processed_frame)
                        self.last_detection_time = current_time
                
                # Add to display queue
                try:
                    # Add camera info and stats
                    fps = self.calculate_fps()
                    info_text = [
                        f"Camera: {self.camera_config.name}",
                        f"FPS: {fps:.1f}",
                        f"Frame: {self.frame_count}",
                        f"Status: {'FIRE!' if fire_detected else 'OK'}"
                    ]
                    
                    for i, text in enumerate(info_text):
                        color = (0, 0, 255) if fire_detected and i == 3 else (0, 255, 0)
                        cv2.putText(processed_frame, text, (10, 30 + i * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    self.display_queue.put_nowait(processed_frame)
                    
                except queue.Full:
                    # Remove oldest frame
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(processed_frame)
                    except queue.Empty:
                        pass
                
                # Update FPS counter
                self.fps_counter.append(current_time)
                
            except Exception as e:
                self.logger.error(f"Frame processing error for {self.camera_config.name}: {e}")
    
    def display_frames(self):
        """Display processed frames"""
        while self.running:
            try:
                frame = self.display_queue.get(timeout=1.0)
                cv2.imshow(self.camera_config.name, frame)
                
                # Check for quit signal
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Display error for {self.camera_config.name}: {e}")
    
    def handle_fire_detection(self, detection: FireDetection, frame: np.ndarray):
        """Handle fire detection event"""
        try:
            # Save frame
            camera_dir = self.config.captures_dir / self.camera_config.name
            camera_dir.mkdir(exist_ok=True)
            
            timestamp_str = detection.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.camera_config.name}_{timestamp_str}.jpg"
            filepath = camera_dir / filename
            
            if cv2.imwrite(str(filepath), frame):
                detection.frame_saved = True
                detection.filename = str(filepath)
                self.logger.info(f"[FIRE DETECTED] {self.camera_config.name} - Saved: {filename}")
            else:
                self.logger.error(f"Failed to save frame for {self.camera_config.name}")
            
            # Log to database
            self.db_manager.log_detection(detection)
            
        except Exception as e:
            self.logger.error(f"Error handling fire detection for {self.camera_config.name}: {e}")
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.fps_counter) < 2:
            return 0.0
        return len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
    
    def start(self):
        """Start camera processing"""
        if not self.setup_camera():
            return False
        
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        self.display_thread.start()
        
        return True
    
    def stop(self):
        """Stop camera processing"""
        self.running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2)
        if hasattr(self, 'display_thread'):
            self.display_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyWindow(self.camera_config.name)
        self.logger.info(f"Camera {self.camera_config.name} stopped")

# ==============================
# Main Fire Detection System
# ==============================
class FireDetectionSystem:
    """Main system orchestrator"""
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.db_manager = DatabaseManager(self.config.db_path)
        self.detection_engine = FireDetectionEngine(self.config, self.logger)
        self.workers = []
        self.running = False
    
    def start(self):
        """Start the fire detection system"""
        self.logger.info("Starting Fire Detection System")
        
        camera_configs = self.config.get_camera_configs()
        
        for name, camera_config in camera_configs.items():
            if not camera_config.enabled:
                continue
                
            worker = CameraWorker(
                camera_config, self.config, self.detection_engine, 
                self.db_manager, self.logger
            )
            
            if worker.start():
                self.workers.append(worker)
                self.logger.info(f"Started camera worker: {name}")
            else:
                self.logger.error(f"Failed to start camera worker: {name}")
        
        if not self.workers:
            self.logger.error("No cameras started successfully")
            return
        
        self.running = True
        self.logger.info(f"Fire Detection System started with {len(self.workers)} cameras")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
                # Monitor system resources
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > self.config.max_memory_usage:
                    self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the fire detection system"""
        self.logger.info("Stopping Fire Detection System")
        self.running = False
        
        for worker in self.workers:
            worker.stop()
        
        cv2.destroyAllWindows()
        self.logger.info("Fire Detection System stopped")

# ==============================
# Main Entry Point
# ==============================
if __name__ == "__main__":
    try:
        system = FireDetectionSystem()
        system.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise
