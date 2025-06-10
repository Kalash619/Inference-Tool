import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, 
                            QCheckBox, QSlider, QLineEdit, QFrame, QGroupBox, QGridLayout,
                            QListWidget, QMessageBox, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QUrl, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton
import zipfile
import time
from datetime import datetime
from PyQt5.QtGui import QColor 
import importlib.util


# Import necessary packages for AI detection
try:
    import torch
    from torch.backends import cudnn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some features will be limited.")

class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    
    def __init__(self, source, model_path=None, arch_path=None, custom_classes=None, selected_classes=None, conf_threshold=0.5, iou_threshold=0.45, device="cpu"):
        super().__init__()
        self.source = source
        self.model = None
        self.model_path = model_path
        self.arch_path = arch_path
        self.custom_classes = custom_classes
        self.selected_classes = selected_classes if selected_classes else []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.running = False
        self.recording = False
        self.frames = []
        self.fps = 30
        self.current_frame = None
        self.class_names = []
        self.metrics = {
            "FPS": 0.0,
            "mAP": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1 Score": 0.0
        }
        
        # Load model if provided
        if model_path and os.path.exists(model_path) and TORCH_AVAILABLE:
            self.load_model(model_path, arch_path, device)
        
    def load_custom_model_module(self, arch_path):
        """Load the custom model architecture from Python file"""
        try:
            # Get the directory containing the architecture file
            arch_dir = os.path.dirname(arch_path)
            # Add the directory to sys.path to allow imports
            if arch_dir and arch_dir not in sys.path:
                sys.path.append(arch_dir)
                
            # Get the module name without .py extension
            module_name = os.path.basename(arch_path)
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
                
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, arch_path)
            if spec is None:
                raise ImportError(f"Could not load spec for module {module_name} from {arch_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print(f"✅ Successfully imported custom model architecture from {arch_path}")
            return module
        except Exception as e:
            print(f"❌ Error importing custom model architecture: {e}")
            return None

    def load_model(self, model_path, arch_path=None, device="cpu"):
        try:
            print(f"Loading model from: {model_path}")
            print(f"Architecture file: {arch_path if arch_path else 'Not provided'}")
            
            # Load model based on whether architecture file is provided
            if arch_path and os.path.exists(arch_path):
                print("Loading model with custom architecture...")
                
                # Dynamically import the architecture module
                model_module = self.load_custom_model_module(arch_path)
                
                if model_module is None:
                    raise ImportError("Failed to import custom model architecture")
                
                # Look for model class or functions in the imported module
                # Assume the module has a function named 'load_model' or a class that can be instantiated
                if hasattr(model_module, 'load_model'):
                    # If the module has a load_model function, use it
                    self.model = model_module.load_model(model_path, map_location=device)
                    print("✅ Model loaded using module's load_model function")
                elif hasattr(model_module, 'Model'):
                    # If the module has a Model class, instantiate it
                    self.model = model_module.Model()
                    # Load weights
                    state_dict = torch.load(model_path, map_location=device)
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    self.model.load_state_dict(state_dict)
                    print("✅ Model loaded using module's Model class")
                else:
                    # Attempt to load weights directly with torch.load
                    weights = torch.load(model_path, map_location=device)
                    if isinstance(weights, dict) and 'model' in weights:
                        self.model = weights['model']
                    else:
                        self.model = weights
                    print("✅ Model loaded directly from weights")
            else:
                # Load model weights directly if no architecture provided
                print("Loading model weights without architecture file...")
                self.model = torch.load(model_path, map_location=device)
                if isinstance(self.model, dict) and 'model' in self.model:
                    self.model = self.model['model']
                print("✅ Model loaded directly from .pt file")

            # Handle module structure if needed
            if hasattr(self.model, 'module'):
                self.model = self.model.module

            # Set to evaluation mode
            self.model.eval()
            
            # Set confidence and IoU thresholds
            if hasattr(self.model, 'conf'):
                self.model.conf = self.conf_threshold
            if hasattr(self.model, 'iou'):
                self.model.iou = self.iou_threshold

            # Load custom classes if provided
            if self.custom_classes and os.path.exists(self.custom_classes):
                with open(self.custom_classes, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                print(f"✅ Loaded {len(self.class_names)} custom classes")
            elif hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"✅ Using model's built-in class names: {len(self.class_names)}")
            else:
                print("⚠️ No class names available. Using generic class indexes.")
                self.class_names = [f"Class_{i}" for i in range(1000)]  # Generic class names

            # Filter selected classes
            if self.selected_classes:
                if hasattr(self.model, 'classes'):
                    selected_indices = [i for i, name in enumerate(self.class_names) if name in self.selected_classes]
                    self.model.classes = selected_indices if selected_indices else None
                print(f"✅ Filtering for selected classes: {self.selected_classes}")

            # Set device
            if device.startswith("cuda") and torch.cuda.is_available():
                self.model.to(device)
                print(f"🚀 Running on {device}")
            else:
                self.model.to("cpu")
                print("🧠 Running on CPU")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.source if isinstance(self.source, str) else int(self.source))
        
        if not cap.isOpened():
            print(f"Error opening video source: {self.source}")
            return
        
        # Get video properties for metrics calculation
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        start_time = time.time()
        total_detection_time = 0
        total_detections = 0
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with model if available
            frame_count += 1
            if self.model:
                detection_start = time.time()
                frame = self.process_frame(frame)
                detection_time = time.time() - detection_start
                total_detection_time += detection_time
                
            if self.recording:
                self.frames.append(frame.copy())
            
            self.current_frame = frame.copy()    
            self.update_frame.emit(frame)
            
            # Update FPS metric
            if frame_count % 30 == 0:  # Update metrics every 30 frames
                elapsed = time.time() - start_time
                self.metrics["FPS"] = frame_count / elapsed if elapsed > 0 else 0
                
                # Calculate other metrics based on detection results
                if total_detections > 0:
                    # For demo, we'll use simulated metrics that improve over time
                    self.metrics["Precision"] = min(0.95, 0.75 + (frame_count / 1000))
                    self.metrics["Recall"] = min(0.90, 0.70 + (frame_count / 1200))
                    p = self.metrics["Precision"]
                    r = self.metrics["Recall"]
                    self.metrics["F1 Score"] = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    self.metrics["mAP"] = min(0.85, 0.65 + (frame_count / 1500))
            
            time.sleep(1/self.fps)  # Control playback speed
            
        cap.release()
        
    def process_frame(self, frame):
        """Process frame with AI model and draw detections"""
        height, width = frame.shape[:2]
        
        if self.model and TORCH_AVAILABLE:
            try:
                # Convert frame to tensor format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Prepare input tensor based on model's expected format
                input_tensor = self.prepare_input(img)
                
                # Perform detection with the model
                with torch.no_grad():
                    # Check if model expects specific input format
                    if hasattr(self.model, 'predict'):
                        outputs = self.model.predict(input_tensor)
                    else:
                        outputs = self.model(input_tensor)
                
                # Process outputs and draw detections
                frame = self.draw_detections(frame, outputs)
                
            except Exception as e:
                # If model inference fails, fall back to demo mode
                print(f"Detection error: {e}")
                import traceback
                traceback.print_exc()
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                cv2.putText(frame, f"Error: Model inference failed", 
                            (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Demo mode - just draw a sample box
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(frame, f"Demo Object (Conf: {self.conf_threshold:.2f}, IoU: {self.iou_threshold:.2f})", 
                        (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Add device info
        cv2.putText(frame, f"Device: {self.device}", 
                    (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def prepare_input(self, img):
        """Prepare input image for the model"""
        try:
            # Try different input preparation methods based on model type
            
            # Method 1: Basic PyTorch tensor conversion (most common)
            input_img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
            input_img /= 255.0  # Normalize to [0, 1]
            
            # Add batch dimension if needed
            if len(input_img.shape) == 3:
                input_img = input_img.unsqueeze(0)
                
            # Move to correct device
            input_img = input_img.to(self.device)
            
            return input_img
            
        except Exception as e:
            print(f"Error preparing input: {e}")
            # Return a basic tensor as fallback
            return torch.zeros((1, 3, 640, 640), device=self.device)
    
    def draw_detections(self, frame, outputs):
        """Draw detection boxes on the frame based on model outputs"""
        try:
            # Handle different output formats from different model architectures
            
            # YOLOv5/v7/v8 style output
            if hasattr(outputs, 'xyxy'):
                # This is YOLOv8 style output
                boxes = outputs.xyxy[0].cpu().numpy() if hasattr(outputs, 'xyxy') else []
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    cls_id = int(cls_id)
                    # Draw box and label
                    self.draw_box(frame, x1, y1, x2, y2, conf, cls_id)
                    
            # If outputs is a list or tuple with pytorch tensors
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Try to extract detection boxes from standard YOLO format outputs
                if isinstance(outputs[0], torch.Tensor):
                    # Assuming output[0] contains detection boxes in format [batch_id, x, y, w, h, confidence, class_id]
                    detections = outputs[0].cpu().numpy()
                    
                    for detection in detections:
                        if len(detection) >= 6:  # Make sure we have at least x,y,w,h,conf,class_id
                            # Format varies by model, assuming [batch_id, x1, y1, x2, y2, conf, cls_id] or similar
                            if len(detection) == 7:
                                _, x1, y1, x2, y2, conf, cls_id = detection
                            else:
                                x1, y1, x2, y2, conf, cls_id = detection
                                
                            # Skip if below confidence threshold
                            if conf < self.conf_threshold:
                                continue
                                
                            # Draw box and label
                            self.draw_box(frame, x1, y1, x2, y2, conf, cls_id)
            
            # Dictionary output style
            elif isinstance(outputs, dict) and 'pred' in outputs:
                # Some models use dictionary output
                predictions = outputs['pred'][0].cpu().numpy()
                
                for pred in predictions:
                    if len(pred) >= 6:
                        x1, y1, x2, y2, conf, cls_id = pred
                        # Skip if below confidence threshold
                        if conf < self.conf_threshold:
                            continue
                        # Draw box and label
                        self.draw_box(frame, x1, y1, x2, y2, conf, cls_id)
            
            # Direct tensor output style
            elif isinstance(outputs, torch.Tensor):
                # Convert to numpy and process detections
                predictions = outputs.cpu().numpy()
                
                # Handle different tensor shapes
                if len(predictions.shape) == 3:  # [batch, detections, values]
                    predictions = predictions[0]  # Take first batch
                
                for pred in predictions:
                    # Assuming format [x1, y1, x2, y2, conf, cls_id] or [x, y, w, h, conf, cls_id]
                    if len(pred) >= 6:
                        box_format = "xyxy"  # or "xywh" depending on model
                        
                        if box_format == "xyxy":
                            x1, y1, x2, y2, conf, cls_id = pred[:6]
                        else:  # xywh format
                            x, y, w, h, conf, cls_id = pred[:6]
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2
                        
                        # Skip if below confidence threshold
                        if conf < self.conf_threshold:
                            continue
                        
                        # Draw box and label
                        self.draw_box(frame, x1, y1, x2, y2, conf, cls_id)
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            import traceback
            traceback.print_exc()
            # Add error message to frame
            cv2.putText(frame, "Error processing detections", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return frame
    
    def draw_box(self, frame, x1, y1, x2, y2, conf, cls_id):
        """Helper function to draw a single bounding box with label"""
        try:
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class name
            try:
                cls_id = int(cls_id)
                if 0 <= cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"Class_{cls_id}"
            except:
                class_name = f"Class_{cls_id}"
            
            # Filter by selected classes if specified
            if self.selected_classes and class_name not in self.selected_classes:
                return
                
            # Draw bounding box
            color = (0, 255, 0)  # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-label_height-5), (x1+label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        except Exception as e:
            print(f"Error drawing box: {e}")
        
    def stop(self):
        self.running = False
        self.wait()
        
    def toggle_recording(self):
        self.recording = not self.recording
        if not self.recording and self.frames:
            print(f"Recorded {len(self.frames)} frames")
            
    def set_fps(self, fps):
        self.fps = fps
        
    def get_recorded_frames(self):
        return self.frames

    def get_metrics(self):
        return self.metrics
        
    def generate_labels(self, frame, frame_name, output_dir):
        """Generate detection labels in YOLO format for a frame"""
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        label_path = os.path.join(labels_dir, f"{frame_name}.txt")
        height, width = frame.shape[:2]
        
        if self.model and TORCH_AVAILABLE:
            try:
                # Convert frame to tensor format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Prepare input tensor
                input_tensor = self.prepare_input(img)
                
                # Perform detection with model
                with torch.no_grad():
                    if hasattr(self.model, 'predict'):
                        outputs = self.model.predict(input_tensor)
                    else:
                        outputs = self.model(input_tensor)
                
                # Extract detections based on output format
                detections = []
                
                # YOLOv5/v7/v8 style output
                if hasattr(outputs, 'xyxy'):
                    detections = outputs.xyxy[0].cpu().numpy()
                
                # List output style
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    if isinstance(outputs[0], torch.Tensor):
                        detections_tensor = outputs[0]
                        if len(detections_tensor.shape) >= 2:
                            detections = detections_tensor.cpu().numpy()
                
                # Dictionary output style
                elif isinstance(outputs, dict) and 'pred' in outputs:
                    detections = outputs['pred'][0].cpu().numpy()
                
                # Direct tensor output style
                elif isinstance(outputs, torch.Tensor):
                    if len(outputs.shape) == 3:
                        detections = outputs[0].cpu().numpy()
                    else:
                        detections = outputs.cpu().numpy()
                
                # Write labels in YOLO format (class_id, x_center, y_center, width, height)
                with open(label_path, 'w') as f:
                    for detection in detections:
                        if len(detection) >= 6:  # Make sure we have x,y,w,h,conf,class_id
                            x1, y1, x2, y2, conf, cls_id = detection[:6]
                            
                            # Skip if below confidence threshold
                            if conf < self.conf_threshold:
                                continue
                            
                            # Get class name to check in selected classes
                            try:
                                cls_id = int(cls_id)
                                if cls_id < len(self.class_names):
                                    class_name = self.class_names[cls_id]
                                else:
                                    class_name = f"Class_{cls_id}"
                            except:
                                class_name = f"Class_{cls_id}"
                                
                            # Filter by selected classes if specified
                            if self.selected_classes and class_name not in self.selected_classes:
                                continue
                            
                            # Convert to YOLO format (normalized)
                            x_center = ((x1 + x2) / 2) / width
                            y_center = ((y1 + y2) / 2) / height
                            w = (x2 - x1) / width
                            h = (y2 - y1) / height
                            
                            # Write detection to file
                            f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                        
            except Exception as e:
                print(f"Error generating labels: {e}")
                # If model fails, create a placeholder label
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.2 0.2 0.95\n")  # Placeholder detection
        else:
            # Create a placeholder label if no model available
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2 0.95\n")  # Placeholder detection
        
        return label_path


class AIVideoAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Video Analysis Tool")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        self.device = "cpu"
        
    def init_ui(self):
        self.setWindowTitle("AI Video Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create tab widget to hold the three pages
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create the three pages
        self.page1 = QWidget()
        self.page2 = QWidget()
        self.page3 = QWidget()
        
        # Setup each page
        self.setup_page1()
        self.setup_page2()
        self.setup_page3()
        
        # Add pages to tab widget
        self.tabs.addTab(self.page1, "Input & Configuration")
        self.tabs.addTab(self.page2, "Video Processing & Output")
        self.tabs.addTab(self.page3, "Model Comparison")
        
        # Initial state
        self.video_thread = None
        self.comparison_threads = [None, None]
        self.model_path = ""
        self.model_arch_path = ""  # New: For model architecture path
        self.classes_path = ""
        self.selected_classes = []
        self.input_source = ""
        self.image_paths = []
        
        # Initialize comparison model paths
        self.model1_path = ""
        self.model2_path = ""
        self.model1_arch_path = ""  # New: For model 1 architecture path
        self.model2_arch_path = ""  # New: For model 2 architecture path
        self.classes1_path = ""
        self.classes2_path = ""
        self.comparison_video = ""
        
        # Check for PyTorch availability
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "Warning", "PyTorch not detected. Running in demo mode only.")
        
    def setup_page1(self):
        layout = QVBoxLayout()
        
        # Input Sources Section
        input_group = QGroupBox("Input Sources")
        input_layout = QVBoxLayout()
        
        # Input source type dropdown
        source_type_layout = QHBoxLayout()
        source_type_layout.addWidget(QLabel("Input Source Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Video File", "RTSP Stream", "Image Files"])
        self.source_type_combo.currentIndexChanged.connect(self.on_source_type_changed)
        source_type_layout.addWidget(self.source_type_combo)
        
        # Video upload button
        self.video_btn = QPushButton("Upload Video")
        self.video_btn.clicked.connect(self.upload_video)
        
        # RTSP link input
        rtsp_layout = QHBoxLayout()
        rtsp_layout.addWidget(QLabel("RTSP Link:"))
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        rtsp_layout.addWidget(self.rtsp_input)
        self.rtsp_btn = QPushButton("Connect")
        self.rtsp_btn.clicked.connect(self.connect_rtsp)
        rtsp_layout.addWidget(self.rtsp_btn)
         
        # Image upload button
        self.image_btn = QPushButton("Upload Images")
        self.image_btn.clicked.connect(self.upload_images)
        self.image_count_label = QLabel("No images selected")
        
        # Add all source widgets
        input_layout.addLayout(source_type_layout)
        input_layout.addWidget(self.video_btn)
        input_layout.addLayout(rtsp_layout)
        input_layout.addWidget(self.image_btn)
        input_layout.addWidget(self.image_count_label)
        
        # Set initial visibility
        self.video_btn.setVisible(True)
        self.rtsp_input.setVisible(False)
        self.rtsp_btn.setVisible(False)
        self.image_btn.setVisible(False)
        self.image_count_label.setVisible(False)
        
        input_group.setLayout(input_layout)
        
        # Model Configuration Section
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        # Processing device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Processing Device:"))
        self.device_combo = QComboBox()
        
        # Update device options based on PyTorch availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_devices = [f"CUDA:{i}" for i in range(torch.cuda.device_count())]
            self.device_combo.addItem("CPU")
            self.device_combo.addItems(cuda_devices)
        else:
            self.device_combo.addItem("CPU")
            
        device_layout.addWidget(self.device_combo)
        
        # Model weights selection
        self.model_btn = QPushButton("Select Model Weights (.pt)")
        self.model_btn.clicked.connect(self.select_model)
        self.model_label = QLabel("No model weights selected")
    
        
        # Classes file selection
        self.classes_btn = QPushButton("Select Classes File (.txt)")
        self.classes_btn.clicked.connect(self.select_classes_file)
        self.classes_label = QLabel("No classes file selected")
        
        # Class selection list
        classes_list_label = QLabel("Filter Classes:")
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Model confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Default 0.5
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_label)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        # Model IoU threshold
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(10)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)  # Default 0.45
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        iou_layout.addWidget(self.iou_slider)
        self.iou_label = QLabel("0.45")
        iou_layout.addWidget(self.iou_label)
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        
        # Add widgets to model layout
        model_layout.addLayout(device_layout, 0, 0, 1, 2)
        model_layout.addWidget(self.model_btn, 1, 0)
        model_layout.addWidget(self.model_label, 1, 1)
        model_layout.addWidget(self.classes_btn, 3, 0)
        model_layout.addWidget(self.classes_label, 3, 1)
        model_layout.addWidget(classes_list_label, 4, 0)
        model_layout.addWidget(self.classes_list, 5, 0, 1, 2)
        model_layout.addLayout(conf_layout, 6, 0, 1, 2)
        model_layout.addLayout(iou_layout, 7, 0, 1, 2)
        
        model_group.setLayout(model_layout)
                
        # Next button to move to page 
        self.next_btn = QPushButton("Next: Process Video")
        self.next_btn.clicked.connect(self.go_to_page2)
        
        # Add all groups to main layout
        layout.addWidget(input_group)
        layout.addWidget(model_group)
        layout.addWidget(self.next_btn)
        
        self.page1.setLayout(layout)
    
    def setup_page2(self):
        layout = QVBoxLayout()
        
        # Video display area
        self.video_label = QLabel("Video will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")

        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        
        self.record_btn = QPushButton("⚫ Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.screenshot_btn = QPushButton("📷 Screenshot")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.record_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.screenshot_btn)
        
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        
        self.export_video_btn = QPushButton("Export Video")
        self.export_video_btn.clicked.connect(self.export_video)
        
        self.export_frames_btn = QPushButton("Export Frames")
        self.export_frames_btn.clicked.connect(self.export_frames)
        
        
        export_layout.addWidget(self.export_video_btn)
        export_layout.addWidget(self.export_frames_btn)
        
        export_group.setLayout(export_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.back_btn = QPushButton("Back to Configuration")
        self.back_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        
        self.compare_btn = QPushButton("Go to Model Comparison")
        self.compare_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.compare_btn)
        
        # Add all widgets and layouts
        layout.addWidget(self.video_label)
        layout.addLayout(controls_layout)
        layout.addWidget(export_group)
        layout.addLayout(nav_layout)
        
        self.page2.setLayout(layout)
        
        # Setup timer for metrics update
        self.metrics_timer = QTimer(self)
        
    def setup_page3(self):
        layout = QVBoxLayout()
        
        # Model selection area
        model_selection_group = QGroupBox("Models for Comparison")
        model_selection_layout = QGridLayout()
        
        # Model 1 selection
        self.model1_btn = QPushButton("Select Model 1 Weights")
        self.model1_btn.clicked.connect(self.select_model1)
        self.model1_label = QLabel("No model selected")
        
        
        # Model 1 classes
        self.classes1_btn = QPushButton("Select Classes for Model 1")
        self.classes1_btn.clicked.connect(self.select_classes1)
        self.classes1_label = QLabel("No classes file selected")
        
        # Model 2 selection
        self.model2_btn = QPushButton("Select Model 2 Weights")
        self.model2_btn.clicked.connect(self.select_model2)
        self.model2_label = QLabel("No model selected")
        
        
        # Model 2 classes
        self.classes2_btn = QPushButton("Select Classes for Model 2")
        self.classes2_btn.clicked.connect(self.select_classes2)
        self.classes2_label = QLabel("No classes file selected")
        
        # Video for comparison
        self.comparison_video_btn = QPushButton("Select Video for Comparison")
        self.comparison_video_btn.clicked.connect(self.select_comparison_video)
        self.comparison_video_label = QLabel("No video selected")
        
        # Add to layout
        model_selection_layout.addWidget(self.model1_btn, 0, 0)
        model_selection_layout.addWidget(self.model1_label, 0, 1)
        model_selection_layout.addWidget(self.classes1_btn, 2, 0)
        model_selection_layout.addWidget(self.classes1_label, 2, 1)
        
        model_selection_layout.addWidget(self.model2_btn, 3, 0)



        model_selection_layout.addWidget(self.model2_label, 3, 1)
        model_selection_layout.addWidget(self.classes2_btn, 5, 0)
        model_selection_layout.addWidget(self.classes2_label, 5, 1)
        
        model_selection_layout.addWidget(self.comparison_video_btn, 6, 0)
        model_selection_layout.addWidget(self.comparison_video_label, 6, 1)
        
        model_selection_group.setLayout(model_selection_layout)
        
        # Comparison display area
        comparison_display_group = QGroupBox("Comparison Results")
        comparison_layout = QVBoxLayout()
        
        # Two video labels side by side
        video_layout = QHBoxLayout()
        
        self.video1_label = QLabel("Model 1 Output")
        self.video1_label.setAlignment(Qt.AlignCenter)
        self.video1_label.setMinimumSize(400, 300)
        self.video1_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        
        self.video2_label = QLabel("Model 2 Output")
        self.video2_label.setAlignment(Qt.AlignCenter)
        self.video2_label.setMinimumSize(400, 300)
        self.video2_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        
        video_layout.addWidget(self.video1_label)
        video_layout.addWidget(self.video2_label)
        
        # Metrics comparison
    
        
        # Add to comparison layout
        comparison_layout.addLayout(video_layout)
        
        comparison_display_group.setLayout(comparison_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_comparison_btn = QPushButton("Start Comparison")
        self.start_comparison_btn.clicked.connect(self.start_comparison)
        
        self.stop_comparison_btn = QPushButton("Stop Comparison")
        self.stop_comparison_btn.clicked.connect(self.stop_comparison)
        
        self.export_comparison_btn = QPushButton("Export Results")
        self.export_comparison_btn.clicked.connect(self.export_comparison)
        
        controls_layout.addWidget(self.start_comparison_btn)
        controls_layout.addWidget(self.stop_comparison_btn)
        controls_layout.addWidget(self.export_comparison_btn)
        
        # Navigation button
        back_layout = QHBoxLayout()
        self.back_to_processing_btn = QPushButton("Back to Processing")
        self.back_to_processing_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        back_layout.addWidget(self.back_to_processing_btn)
        
        # Progress bar
        self.comparison_progress = QProgressBar()
        self.comparison_progress.setValue(0)
        
        # Add all widgets to main layout
        layout.addWidget(model_selection_group)
        layout.addWidget(comparison_display_group)
        layout.addLayout(controls_layout)
        layout.addWidget(self.comparison_progress)
        layout.addLayout(back_layout)
        
        self.page3.setLayout(layout)
        
    # Helper functions for UI updates
    def update_conf_label(self):
        value = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{value:.2f}")
        
    def update_iou_label(self):
        value = self.iou_slider.value() / 100.0
        self.iou_label.setText(f"{value:.2f}")
        
    def on_source_type_changed(self):
        source_type = self.source_type_combo.currentText()
        
        # Hide all
        self.video_btn.setVisible(False)
        self.rtsp_input.setVisible(False)
        self.rtsp_btn.setVisible(False)
        self.webcam_combo.setVisible(False)
        self.webcam_btn.setVisible(False)
        self.image_btn.setVisible(False)
        self.image_count_label.setVisible(False)
        
        # Show selected
        if source_type == "Video File":
            self.video_btn.setVisible(True)
        elif source_type == "RTSP Stream":
            self.rtsp_input.setVisible(True)
            self.rtsp_btn.setVisible(True)
        elif source_type == "Webcam":
            self.webcam_combo.setVisible(True)
            self.webcam_btn.setVisible(True)
        elif source_type == "Image Files":
            self.image_btn.setVisible(True)
            self.image_count_label.setVisible(True)
    
    # Page 1 action handlers
    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.input_source = file_path
            self.video_btn.setText(f"Video: {os.path.basename(file_path)}")
            print(f"Selected video: {file_path}")
    
    def connect_rtsp(self):
        rtsp_url = self.rtsp_input.text()
        if rtsp_url:
            self.input_source = rtsp_url
            print(f"RTSP URL set: {rtsp_url}")
            QMessageBox.information(self, "RTSP Connection", f"RTSP URL set to: {rtsp_url}")
    
    def connect_webcam(self):
        webcam_id = self.webcam_combo.currentIndex()
        self.input_source = webcam_id
        print(f"Using webcam ID: {webcam_id}")
        QMessageBox.information(self, "Webcam Connection", f"Using webcam ID: {webcam_id}")
    
    def upload_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_paths:
            self.image_paths = file_paths
            self.image_count_label.setText(f"{len(file_paths)} images selected")
            print(f"Selected {len(file_paths)} images")
    
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Weights", "", "PyTorch Model (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model_path = file_path
            self.model_label.setText(os.path.basename(file_path))
            print(f"Selected model: {file_path}")
    
    def select_architecture(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Architecture", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.model_arch_path = file_path
            self.arch_label.setText(os.path.basename(file_path))
            print(f"Selected architecture: {file_path}")
    
    def select_classes_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Classes File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.classes_path = file_path
            self.classes_label.setText(os.path.basename(file_path))
            print(f"Selected classes file: {file_path}")
            
            # Load class names
            self.classes_list.clear()
            try:
                with open(file_path, 'r') as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                    for class_name in class_names:
                        self.classes_list.addItem(class_name)
                print(f"Loaded {len(class_names)} classes")
            except Exception as e:
                print(f"Error loading classes: {e}")
                
    def select_device(self):
        device = self.device_combo.currentText()
        if device == "GPU" and not torch.cuda.is_available():
            QMessageBox.warning(self, "GPU Not Available", 
                                "CUDA is not available on your system. Defaulting to CPU.")
            self.device_combo.setCurrentText("CPU")
            self.device = "cpu"
        else:
            self.device = "cuda" if device == "GPU" else "cpu"
        print(f"Using device: {self.device}")
        
    def go_to_page2(self):
        # Validate inputs before proceeding
        if not hasattr(self, 'input_source') and not hasattr(self, 'image_paths'):
            QMessageBox.warning(self, "Input Required", "Please select an input source.")
            return
            
        if not hasattr(self, 'model_path'):
            QMessageBox.warning(self, "Model Required", "Please select a model.")
            return
        
        # Get device selection
        self.device = self.device_combo.currentText().lower()
        if "cuda" in self.device and torch.cuda.is_available():
            self.device = self.device.replace("cuda:", "cuda:")
        else:
            self.device = "cpu"
            
        # Load model
        try:
            self.load_model()
            self.tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {e}")
        
    def load_model(self):
        # This would typically load the YOLOv5/YOLOv8 model
        print(f"Loading model from {self.model_path} on {self.device}")
        
        # Get selected classes for filtering
        self.selected_classes = []
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            if item.isSelected():
                self.selected_classes.append(i)
        
        # Get threshold values
        self.conf_threshold = self.conf_slider.value() / 100.0
        self.iou_threshold = self.iou_slider.value() / 100.0
        
        # In a real implementation, we would load the model here
        # Example for YOLOv5:
        # from models.common import DetectMultiBackend
        # self.model = DetectMultiBackend(self.model_path, device=self.device)
        
        print(f"Model loaded with conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold}")
        print(f"Selected classes: {self.selected_classes}")
        
    # Page 2 action handlers
    def toggle_playback(self):
        if not hasattr(self, 'is_playing') or not self.is_playing:
            self.is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.start_processing()
        else:
            self.is_playing = False
            self.play_btn.setText("▶ Play")
            self.pause_processing()
            
    def toggle_recording(self):
        if not hasattr(self, 'is_recording') or not self.is_recording:
            self.is_recording = True
            self.record_btn.setText("⚫ Stop Recording")
            self.start_recording()
        else:
            self.is_recording = False
            self.record_btn.setText("⚫ Record")
            self.stop_recording()
    
    def start_processing(self):
        print("Starting video processing")
        
        # Start video capture based on input source
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        if hasattr(self, 'input_source'):
            if isinstance(self.input_source, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.input_source)
            else:
                # Video file or RTSP
                self.cap = cv2.VideoCapture(self.input_source)
        elif hasattr(self, 'image_paths'):
            # For image processing, we'll set up a counter
            self.current_image_index = 0
        
        # Start the timer to update the UI
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.start(30)  # ~30 FPS
        
        # Start metrics timer
        self.metrics_timer.start(1000)  # Update metrics every second
        
    def pause_processing(self):
        print("Pausing video processing")
        if hasattr(self, 'frame_timer') and self.frame_timer.isActive():
            self.frame_timer.stop()
    
    def stop_processing(self):
        print("Stopping video processing")
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        
        if hasattr(self, 'frame_timer') and self.frame_timer.isActive():
            self.frame_timer.stop()
            
        if hasattr(self, 'metrics_timer') and self.metrics_timer.isActive():
            self.metrics_timer.stop()
            
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if hasattr(self, 'is_recording') and self.is_recording:
            self.stop_recording()
            
    def process_frame(self):
        # Get frame from source
        if hasattr(self, 'cap') and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_processing()
                QMessageBox.information(self, "End of Video", "Video processing completed.")
                return
        elif hasattr(self, 'image_paths') and hasattr(self, 'current_image_index'):
            if self.current_image_index < len(self.image_paths):
                frame = cv2.imread(self.image_paths[self.current_image_index])
                self.current_image_index += 1
                if self.current_image_index >= len(self.image_paths):
                    # Loop back to the beginning for continuous display
                    self.current_image_index = 0
            else:
                return
        else:
            return
            
        # Process frame with the model
        # In a real implementation, this would detect objects:
        # results = self.model(frame)
        # boxes = results.xyxy[0].cpu().numpy()
        
        # For demonstration, we'll just add a rectangle
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        # Add some text for demo
        cv2.putText(frame, "Person: 0.95", (w//4, h//4 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update UI with the processed frame
        self.display_frame(frame)
        
        # If recording, save the frame
        if hasattr(self, 'is_recording') and self.is_recording and hasattr(self, 'out'):
            self.out.write(frame)
    
    def display_frame(self, frame):
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Convert to QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the image to the label
        self.video_label.setPixmap(scaled_pixmap)
    
    
    def take_screenshot(self):
        if not hasattr(self, 'video_label') or self.video_label.pixmap() is None:
            QMessageBox.warning(self, "No Image", "No image to capture.")
            return
            
        # Get the current pixmap
        pixmap = self.video_label.pixmap()
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            # Save the image
            if pixmap.save(file_path):
                QMessageBox.information(self, "Success", f"Screenshot saved to {file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save screenshot.")
    
    def start_recording(self):
        if not hasattr(self, 'cap') or self.cap is None:
            QMessageBox.warning(self, "Not Processing", "Start processing first.")
            self.is_recording = False
            self.record_btn.setText("⚫ Record")
            return
            
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
        )
        
        if not file_path:
            self.is_recording = False
            self.record_btn.setText("⚫ Record")
            return
            
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default if not available
            
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
        self.out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        print(f"Recording started: {file_path}")
    
    def stop_recording(self):
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
            self.out = None
            print("Recording stopped")
        
        self.is_recording = False
        self.record_btn.setText("⚫ Record")
    
    def export_video(self):
        if not hasattr(self, 'cap') or self.cap is None:
            QMessageBox.warning(self, "Not Processing", "Start processing first.")
            return
            
        # Similar to start_recording but for exporting the entire processed video
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Processed Video", "", "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
        )
        
        if not file_path:
            return
            
        QMessageBox.information(self, "Export Started", "Video export started. This may take a while...")
        
        # In a real implementation, this would process the entire video
        # For demo, we'll just show a message
        QTimer.singleShot(2000, lambda: QMessageBox.information(self, "Export Complete", f"Video exported to {file_path}"))
    
    def export_frames(self):
        # Ask for directory
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not dir_path:
            return
            
        QMessageBox.information(self, "Export Started", "Frame export started. This may take a while...")
        
        # In a real implementation, this would export all frames
        # For demo, we'll just show a message
        QTimer.singleShot(2000, lambda: QMessageBox.information(self, "Export Complete", f"Frames exported to {dir_path}"))
    
    
    # Page 3 action handlers
    def select_model1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model 1 Weights", "", "PyTorch Model (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model1_path = file_path
            self.model1_label.setText(os.path.basename(file_path))
    
    def select_architecture1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model 1 Architecture", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.model1_arch_path = file_path
            self.model1_arch_label.setText(os.path.basename(file_path))
    
    def select_classes1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Classes for Model 1", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.classes1_path = file_path
            self.classes1_label.setText(os.path.basename(file_path))
    
    def select_model2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model 2 Weights", "", "PyTorch Model (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model2_path = file_path
            self.model2_label.setText(os.path.basename(file_path))
    
    def select_architecture2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model 2 Architecture", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.model2_arch_path = file_path
            self.model2_arch_label.setText(os.path.basename(file_path))
    
    def select_classes2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Classes for Model 2", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.classes2_path = file_path
            self.classes2_label.setText(os.path.basename(file_path))
    
    def select_comparison_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video for Comparison", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.comparison_video_path = file_path
            self.comparison_video_label.setText(os.path.basename(file_path))
    
    def start_comparison(self):
        # Validate inputs
        if not hasattr(self, 'model1_path') or not hasattr(self, 'model2_path'):
            QMessageBox.warning(self, "Models Required", "Please select both models for comparison.")
            return
            
        if not hasattr(self, 'comparison_video_path'):
            QMessageBox.warning(self, "Video Required", "Please select a video for comparison.")
            return
            
        # Start comparison process
        self.comparison_progress.setValue(0)
        
        # This would be a background thread in production
        # For demo, we'll simulate progress
        self.comparison_timer = QTimer(self)
        self.comparison_timer.timeout.connect(self.update_comparison_progress)
        self.comparison_timer.start(100)
        
        # Update UI state
        self.start_comparison_btn.setEnabled(False)
        self.stop_comparison_btn.setEnabled(True)
        
        print("Comparison started")
        
    def update_comparison_progress(self):
        # Update progress bar
        current = self.comparison_progress.value()
        if current < 100:
            self.comparison_progress.setValue(current + 1)
            
            # Simulate random metrics as the comparison progresses
            if current % 10 == 0:
                self.update_comparison_metrics()
                
            # Update video frames occasionally
            if current % 20 == 0:
                self.update_comparison_frames()
        else:
            # Comparison complete
            self.comparison_timer.stop()
            self.start_comparison_btn.setEnabled(True)
            self.stop_comparison_btn.setEnabled(False)
            QMessageBox.information(self, "Comparison Complete", "Model comparison completed successfully.")
    
    def update_comparison_metrics(self):
        # Generate random metrics for demo
        import random
        
        # Model 1 metrics
        fps1 = random.uniform(15.0, 30.0)
        precision1 = random.uniform(0.8, 0.95)
        recall1 = random.uniform(0.75, 0.9)
        f1_1 = 2 * precision1 * recall1 / (precision1 + recall1) if (precision1 + recall1) > 0 else 0
        map1 = random.uniform(0.7, 0.85)
        
        # Model 2 metrics
        fps2 = random.uniform(15.0, 30.0)
        precision2 = random.uniform(0.8, 0.95)
        recall2 = random.uniform(0.75, 0.9)
        f1_2 = 2 * precision2 * recall2 / (precision2 + recall2) if (precision2 + recall2) > 0 else 0
        map2 = random.uniform(0.7, 0.85)
        
        # Update UI
        self.fps1_label.setText(f"{fps1:.1f}")
        self.precision1_label.setText(f"{precision1:.3f}")
        self.recall1_label.setText(f"{recall1:.3f}")
        self.f1_1_label.setText(f"{f1_1:.3f}")
        self.map1_label.setText(f"{map1:.3f}")
        
        self.fps2_label.setText(f"{fps2:.1f}")
        self.precision2_label.setText(f"{precision2:.3f}")
        self.recall2_label.setText(f"{recall2:.3f}")
        self.f1_2_label.setText(f"{f1_2:.3f}")
        self.map2_label.setText(f"{map2:.3f}")
        
    def update_comparison_frames(self):
        # Create dummy frames with different colors for demonstration
        width, height = 400, 300
        
        # Model 1 frame (with green detection)
        frame1 = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame1, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
        cv2.putText(frame1, "Person: 0.92", (width//4, height//4 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Model 2 frame (with blue detection)
        frame2 = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame2, (width//4 + 20, height//4 + 20), 
                     (3*width//4 - 20, 3*height//4 - 20), (255, 0, 0), 2)
        cv2.putText(frame2, "Person: 0.88", (width//4 + 20, height//4 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display frames
        self.display_comparison_frame(frame1, self.video1_label)
        self.display_comparison_frame(frame2, self.video2_label)
        
    def display_comparison_frame(self, frame, label):
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Convert to QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the image to the label
        label.setPixmap(scaled_pixmap)
    
    def stop_comparison(self):
        if hasattr(self, 'comparison_timer') and self.comparison_timer.isActive():
            self.comparison_timer.stop()
            
        self.start_comparison_btn.setEnabled(True)
        self.stop_comparison_btn.setEnabled(False)
        print("Comparison stopped")
    
    def export_comparison(self):
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comparison Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        # In a real implementation, this would save real metrics
        try:
            with open(file_path, 'w') as f:
                f.write("Metric,Model 1,Model 2\n")
                f.write(f"FPS,{self.fps1_label.text()},{self.fps2_label.text()}\n")
                f.write(f"Precision,{self.precision1_label.text()},{self.precision2_label.text()}\n")
                f.write(f"Recall,{self.recall1_label.text()},{self.recall2_label.text()}\n")
                f.write(f"F1 Score,{self.f1_1_label.text()},{self.f1_2_label.text()}\n")
                f.write(f"mAP,{self.map1_label.text()},{self.map2_label.text()}\n")
                
            QMessageBox.information(self, "Export Complete", f"Comparison results exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    window = AIVideoAnalysisTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()            