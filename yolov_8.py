import random
import sys
import os
import cv2
import numpy as np
import torch
import threading
import time
from datetime import datetime
import urllib.request
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QLineEdit, QComboBox, QCheckBox, QSlider, QStackedWidget,
                            QProgressBar, QMessageBox, QGridLayout, QScrollArea, QSplitter,
                            QListWidget, QListWidgetItem, QToolBar, QAction, QFileSystemModel,
                            QTreeView, QDialog, QHeaderView, QGroupBox, QRadioButton,QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl, QDir
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from ultralytics import YOLO

# Default model paths
DEFAULT_MODELS = {
    "YOLOv8": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
    "YOLOv9": ["YOLOv9t", "YOLOv9s", "YOLOv9m", "YOLOv9c", "YOLOv9e"],
    "YOLOv10": ["YOLOv10n", "YOLOv10s", "YOLOv10m", "YOLOv10l", "YOLOv10x"],
    "YOLOv11": ["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"]
}




# Default COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

class YOLOv8Detector:
    def __init__(self, model_path=None, classes_file=None, conf_threshold=0.5, iou_threshold=0.45, device="cuda"):
        """
        Initialize YOLOv8 detector
        Args:
            model_path: Path to YOLOv8 model file or model name
            classes_file: Path to classes.txt file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cuda", "cpu")
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.classes_file = classes_file
        self.class_names = COCO_CLASSES  # Default to COCO classes
        
        # Load custom classes if available
        if self.classes_file and os.path.exists(self.classes_file):
            self.load_classes()
        
        # Set device (use CUDA if available and requested, otherwise CPU)
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using CUDA for inference")
        else:
            self.device = "cpu"
            print(f"Using CPU for inference")
        
        # Get model name or path
        if self.model_path:
            if os.path.exists(self.model_path):
                self.model_name = os.path.basename(self.model_path)
            else:
                self.model_name = self.model_path  # If it's a predefined model name
        else:
            self.model_name = "YOLOv8n"  # Default model
        
        # Load model 
        try:
            print(f"Loading model: {self.model_name}")
            
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model from file
                self.model = YOLO(self.model_path)
                print(f"Custom model loaded from: {self.model_path}")
            else:
                # Use predefined model
                # For predefined models, we convert to the actual model name format
                yolo_model = self.model_name.lower()  # Convert to lowercase for YOLO naming
                self.model = YOLO(yolo_model)
                print(f"Predefined model loaded: {yolo_model}")
            # Set model parameters
            self.model.conf = self.conf_threshold  # Confidence threshold
            self.model.iou = self.iou_threshold    # IoU threshold
            print("Model Details:")
            print(f"- Classes: {self.model.names if hasattr(self.model, 'names') else 'Unknown'}")
            print(f"- Model task: {self.model.task if hasattr(self.model, 'task') else 'Unknown'}")
            print(f"- Model YAML: {self.model.model.yaml if hasattr(self.model.model, 'yaml') else 'Unknown'}")

            print(f"{self.model_name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            raise
   

    def load_classes(self):
        """Load class names from classes.txt file"""
        try:
            with open(self.classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} classes from {self.classes_file}")
        except Exception as e:
            print(f"Error loading classes file: {e}")
            # Fall back to COCO classes
            self.class_names = COCO_CLASSES

    def detect(self, frame, selected_classes=None):
        """
        Perform detection on a frame
        Args:
            frame: Input frame (numpy array)
            selected_classes: List of class names to filter detections
        Returns:
            Processed frame with detections
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        if self.model is None:
            return frame, []
        
        # Create a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Perform inference using YOLOv8
        results = self.model(frame, verbose=False)
        
        # Convert results to a list of detections in the format [x1, y1, x2, y2, conf, cls]
        detections = []
        
        # YOLOv8 provides results differently, extract boxes, confidences and class ids
        for result in results:
            # Get boxes
            boxes = result.boxes
            
            # Extract detection information
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist() if self.device == "cpu" else boxes.xyxy[i].cpu().tolist()
                conf = boxes.conf[i].item() if self.device == "cpu" else boxes.conf[i].cpu().item()
                cls = boxes.cls[i].item() if self.device == "cpu" else boxes.cls[i].cpu().item()
                class_id = int(cls)
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Filter by selected classes if specified
                if selected_classes and class_name not in selected_classes:
                    continue
                
                # Add to detections list
                detections.append([x1, y1, x2, y2, conf, class_id])
        
        # Draw detections on the frame
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(cls)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            label = f"{class_name}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame, detections

class VideoProcessingThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    finished_processing = pyqtSignal(str)
    progress_update = pyqtSignal(int,int)

    def __init__(self, source_path, model_path=None, classes_file=None, 
                 selected_classes=None, conf_threshold=0.5, iou_threshold=0.45, 
                 device="cuda", source_type="video"):
        super().__init__()
        self.source_path = source_path
        self.model_path = model_path
        self.classes_file = classes_file
        self.selected_classes = selected_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.is_running = True
        self.source_type = source_type  # "video", "rtsp", "image"
        self.is_rtsp = False  # Initialize is_rtsp to False

        # For multiple images processing
        if source_type == "images" and isinstance(source_path, list):
            self.source_paths = source_path
            self.total_sources = len(source_path)
        else:
            self.source_paths = [source_path]
            self.total_sources = 1  # Default for video

        # Check if source_path is a string and starts with RTSP/HTTP prefixes
        if isinstance(source_path, str) and source_path.startswith(('rtsp://', 'rtmp://', 'http://')):
            self.is_rtsp = True

        self.output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.detector = None
        self.current_frame_index = 0
        self.total_sources = 1  # Default for video
        
        # For multiple images processing
        if source_type == "images" and isinstance(source_path, list):
            self.source_paths = source_path
            self.total_sources = len(source_path)
        else:
            self.source_paths = [source_path]
        
    def load_model(self):
        try:
            print(f"Loading model: {self.model_path}")
            print(f"Classes file: {self.classes_file}")
            print(f"Selected classes: {self.selected_classes}")
            print(f"Confidence threshold: {self.conf_threshold}")
            print(f"IOU threshold: {self.iou_threshold}")
            print(f"Device: {self.device}")
            
            # Initialize YOLOv8 detector
            self.detector = YOLOv8Detector(
                model_path=self.model_path,
                classes_file=self.classes_file,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                device=self.device
            )
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def run(self):
        # Load model
        model_loaded = self.load_model()
        if not model_loaded:
            self.finished_processing.emit("Error: Failed to load model")
            return
        
        if self.source_type == "images":
            self.process_images()
        else:  # video or rtsp
            self.process_video()
    
    def process_images(self):
        """Process a list of images"""
        os.makedirs(self.output_path, exist_ok=True)
        self.processed_frames = []  # Store tuples of (processed_img, output_path, detections)
        total = len(self.source_paths)

        for i, img_path in enumerate(self.source_paths):
            if not self.is_running:
                break
                
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            
            # Process image
            processed_img, detections = self.detector.detect(img, self.selected_classes)
            
            # Save image with annotations
            img_name = os.path.basename(img_path)
            output_img_path = os.path.join(self.output_path, f"detected_{img_name}")
            cv2.imwrite(output_img_path, processed_img)
            
            # Save detection results in YOLO format
            self.save_yolo_annotations(img_path, detections)
            
        # Store results for navigation
            self.processed_frames.append((processed_img, output_img_path, detections))
            # Update progress
            if i == 0:
                self.current_frame_index = 0
                self.update_frame.emit(processed_img)

            self.progress_update.emit(i + 1, total)
            time.sleep(3)  # Delay between images

        self.finished_processing.emit(f"Processed {len(self.processed_frames)} images. Use Next/Previous to view.")

    def show_next_image(self):
        if self.current_frame_index < len(self.processed_frames) - 1:
            self.current_frame_index += 1
            img = self.processed_frames[self.current_frame_index][0]
            self.update_frame.emit(img)

    def show_previous_image(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            img = self.processed_frames[self.current_frame_index][0]
            self.update_frame.emit(img)
    def next_image(self):
        if hasattr(self, 'thread'):
            self.thread.show_next_image()


    def save_yolo_annotations(self, img_path, detections):
        """Save detections in YOLO format (for training)"""
        try:
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            # Create txt file with same name as image
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.output_path, f"{img_name}.txt")
            
            with open(txt_path, 'w') as f:
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Convert to YOLO format: class_id center_x center_y width height
                    # All values normalized to [0, 1]
                    class_id = int(cls)
                    center_x = ((x1 + x2) / 2) / width
                    center_y = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    
                    # Write to file
                    f.write(f"{class_id} {center_x} {center_y} {box_width} {box_height}\n")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def process_video(self):
        """Process video file or RTSP stream"""
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self.finished_processing.emit(f"Error: Could not open {self.source_type} source")
            return
        
        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_rtsp else -1
        
        # Create output video writer
        self.output_path = f"{self.output_path}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if not self.is_rtsp:  # End of file for regular videos
                    break
                continue  # For RTSP, just try to get the next frame


            
            # Process frame with the model
            processed_frame, detections = self.detector.detect(frame, self.selected_classes)
            
            # Add frame number for training purposes
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, height - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Emit the frame for display
            self.update_frame.emit(processed_frame)
            
            # Write to output video
            out.write(processed_frame)
            
            # Update progress for regular videos
            if not self.is_rtsp and total_frames > 0:
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_update.emit(progress)
            
            # Save frame and annotations periodically for training (every 10 frames)
            if frame_count % 10 == 0:
                os.makedirs(os.path.join(self.output_path.replace(".mp4", ""), "frames"), exist_ok=True)
                frame_path = os.path.join(self.output_path.replace(".mp4", ""), "frames", f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Save annotations
                txt_path = os.path.join(self.output_path.replace(".mp4", ""), "frames", f"frame_{frame_count:06d}.txt")
                with open(txt_path, 'w') as f:
                    for detection in detections:
                        x1, y1, x2, y2, conf, cls = detection
                        
                        # Convert to YOLO format
                        class_id = int(cls)
                        center_x = ((x1 + x2) / 2) / width
                        center_y = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Write to file
                        f.write(f"{class_id} {center_x} {center_y} {box_width} {box_height}\n")
                
        cap.release()
        out.release()
        
        if not self.is_rtsp:
            self.finished_processing.emit(f"Processing completed. Output saved to {self.output_path}")
        else:
            self.finished_processing.emit("RTSP stream recording stopped")
    
    def stop(self):
        self.is_running = False
        self.wait()

class ModelSelectDialog(QDialog):
    """Dialog for selecting model from file system"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.setGeometry(200, 200, 800, 500)
        
        layout = QVBoxLayout()
        
        # Create file system model
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        
        # Set file filters
        self.model.setNameFilters(["*.pt", "*.weights"])
        self.model.setNameFilterDisables(False)
        
        # Create tree view
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.homePath()))
        self.tree.setColumnWidth(0, 300)
        
        # Add predefined models group
        self.predefined_group = QGroupBox("Predefined Models")
        # Scroll area to hold the version groups horizontally
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container_widget = QWidget()
        predefined_layout = QHBoxLayout(container_widget)
        predefined_layout.setSpacing(4)
        predefined_layout.setContentsMargins(2, 2, 2, 2) 
        self.predefined_buttons = []  
        
        for version, models in DEFAULT_MODELS.items():
          version_group = QGroupBox(version)
          version_layout = QVBoxLayout()
          version_layout.setSpacing(0)  
          version_layout.setContentsMargins(1, 1, 1, 1)
    
          for model_name in models:
              radio = QRadioButton(model_name)
              radio.setStyleSheet("margin-left: 0px; margin-right: 0px;") 
              self.predefined_buttons.append(radio)
              version_layout.addWidget(radio)
              version_group.setLayout(version_layout)
              version_group.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
              predefined_layout.addWidget(version_group)
        scroll_area.setWidget(container_widget)
        self.predefined_group.setLayout(predefined_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.predefined_group.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

# Optionally: resize the dialog for a smaller footprint
        self.resize(700, 500)
        # Add to layout
        layout.addWidget(self.predefined_group)
        layout.addWidget(QLabel("OR Select Custom Model:"))
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Paste path or select folder...")
        path_layout.addWidget(self.path_input)
        layout.addLayout(path_layout)
        layout.addWidget(self.tree)
        # Buttons
        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select")
        self.select_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    def browse_model_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Model Folder", "")
        if folder_path:
          self.path_input.setText(folder_path)

    def handle_pasted_path(self, path):
       path = path.strip()
       print(path)



       if os.path.exists(path):
        self.selected_model_path = path
        self.model_label.setText("✅ Selected internally developed model")
        self.check_start_button_status()
       else:
        self.selected_model_path = None
        self.model_label.setText("❌ Invalid path")

    def get_selected_model(self):
        """Return selected model path or name"""
        # Check if predefined model is selected
        for button in self.predefined_buttons:
            if button.isChecked():
                return button.text()
        # CHECK WHETHER VALUE IS IN THIS VARIABLE OR NOT IF YES RETURN IT 
        if self.path_input.text() is not None:
            return  self.path_input.text().strip()
        # Otherwise return selected file path
        indexes = self.tree.selectedIndexes()
        if indexes and indexes[0].column() == 0:
            return self.model.filePath(indexes[0])
        
        return None

class ClassesDialog(QDialog):
    """Dialog for selecting classes file"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Classes File")
        self.setGeometry(200, 200, 800, 500)
        
        layout = QVBoxLayout()
        
        # Create file system model
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        
        # Set file filters
        self.model.setNameFilters(["*.txt", "*.names"])
        self.model.setNameFilterDisables(False)
        
        # Create tree view
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.homePath()))
        self.tree.setColumnWidth(0, 300)
        
        # Default classes option
        self.use_default = QCheckBox("Use default COCO classes")
        self.use_default.setChecked(False)
        
        # Add to layout
        layout.addWidget(self.use_default)
        layout.addWidget(QLabel("OR Select Custom Classes File:"))
       
        layout.addWidget(QLabel("Add Custom Classes:"))

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name")
        self.add_class_button = QPushButton("Add")
        self.add_class_button.clicked.connect(self.add_class)

        class_input_layout = QHBoxLayout()
        class_input_layout.addWidget(self.class_input)
        class_input_layout.addWidget(self.add_class_button)
        layout.addLayout(class_input_layout)

        self.class_list = QListWidget()
        layout.addWidget(self.class_list)
        # Label to show status (e.g., "3 classes selected")
        self.classes_label = QLabel("Using default COCO classes")
        layout.addWidget(self.classes_label)
        layout.addWidget(self.tree)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select")
        self.select_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.custom_classes = []  # Store added custom classes

    def add_class(self):
      class_name = self.class_input.text().strip()
      if class_name and not self.is_class_duplicate(class_name):
        self.class_list.addItem(class_name)  # Add to QListWidget
        self.custom_classes.append(class_name) 
        self.class_input.clear()
        self.update_class_status()
    def update_class_status(self):
     if self.custom_classes:
        self.classes_label.setText(f"{len(self.custom_classes)} classes selected")
     else:
        self.classes_label.setText("Using default COCO classes")


    def is_class_duplicate(self, class_name):
        for i in range(self.class_list.count()):
            if self.class_list.item(i).text().lower() == class_name.lower():
                return True
        return False
    def get_selected_classes_file(self):
        """Return selected classes file path or Non e for default"""
        if self.use_default.isChecked():
            return None
        
        indexes = self.tree.selectedIndexes()
        if indexes and indexes[0].column() == 0:
            return self.model.filePath(indexes[0])
        
        return None
    '''def accept(self):
     if self.custom_classes:
        print(f"Selected custom classes: {self.custom_classes}")
        self.classes_label.setText(f"{len(self.custom_classes)} classes selected")
     else:
        print("Using default COCO classes")
        self.classes_label.setText("Using default COCO classes")
     super().accept()'''
    def accept(self):
     if self.custom_classes:
        class_count = len(self.custom_classes)
        message = f"{class_count} classes selected"
        print(message)
        self.classes_label.setText(message)
     else:
        print("Using default COCO classes")
        self.classes_label.setText("Using default COCO classes")
    
     super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv Video Processing and Detection Tool")
        self.setGeometry(100, 100, 1200, 800)
         # Set overall application stylesheet
        self.video_upload_btn = QPushButton("⬆ Upload Video")
        self.video_upload_btn.setFixedHeight(100)
        self.video_upload_btn.setMinimumWidth(500)
        self.setStyleSheet("""
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #dde6f0, stop:1 #c7d4e8);
    color: #1f2a38;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}

QLabel {
    font-size: 15px;
    color: #2c3e50;
}

QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                     stop:0 #5b7db3, stop:1 #90a8d4);
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    color: #1B1A1D;
    padding: 10px 20px;
    margin: 6px;

}

QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                     stop:0 #4c6ea3, stop:1 #7f9ec6);
   
                           
}

QPushButton:pressed {
    background-color: #395a87;
}

QRadioButton {
    font-size: 14px;
    color: #34495e;
    spacing: 6px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #6c89aa;
    border-radius: 8px;
    background: #ecf0f5;
}

QRadioButton::indicator:checked {
    background-color: #3b5e8d;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #cad6e6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3b5e8d;
    border: 1px solid #2f4d75;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QComboBox {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #9eb3c8;
    border-radius: 6px;
    padding: 6px 12px;
}

QComboBox:hover {
    border-color: #3b5e8d;
}

QProgressBar {
    border: 1px solid #a5b8cc;
    background-color: #f0f3f8;
    border-radius: 8px;
    text-align: center;
    color: #2c3e50;
    height: 22px;
}

QProgressBar::chunk {
    background-color: #3b5e8d;
    border-radius: 8px;
}

QToolBar {
    background-color: #e4ebf3;
    border-bottom: 1px solid #b2c1d5;
    padding: 4px;
}

QToolBar QToolButton {
    background-color: #a8bdda;
    color: #1f2a38;
    border: 1px solid #7d94b3;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 2px;
}

QToolBar QToolButton:hover {
    background-color: #92aad1;
    border-color: #3b5e8d;
}

QStackedWidget {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #cad6e6;

}

QScrollArea {
    background-color: #f7f9fc;
    border-radius: 10px;
}

/* Optional: Tooltip styling */
QToolTip {
    background-color: #3b5e8d;
    color: #ffffff;
    border: none;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 12px;
}
""")

   
        # Create a stacked widget to handle different pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create pages
        self.page1 = self.create_upload_page()
        self.page2 = self.create_detection_page()
        self.page3 = self.create_comparison_page()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        
        # Initialize instance variables
        self.source_paths = []  # Can be single video path or list of image paths
        self.model_path = None
        self.classes_file = None
        self.output_path = ""
        self.processing_thread = None
        self.selected_classes = []
        self.is_rtsp = False
        self.source_type = "video"  # "video", "rtsp", "images"
        self.class_names = COCO_CLASSES  # Default class names
        
        # Set initial page
        self.stacked_widget.setCurrentIndex(0)
        
        # Create toolbar
        self.create_toolbar()

  
    def create_toolbar(self):
        """Create toolbar with common actions"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Home action
        home_action = QAction("Home", self)
        home_action.triggered.connect(self.go_to_page1)
        toolbar.addAction(home_action)
        
        # Detection action
        detection_action = QAction("Detection", self)
        detection_action.triggered.connect(self.go_to_page2)
        toolbar.addAction(detection_action)
        
        # Comparison action
        comparison_action = QAction("Comparison", self)
        comparison_action.triggered.connect(self.go_to_page3)
        toolbar.addAction(comparison_action)
        
        # Separator
        toolbar.addSeparator()
        
        # Help action
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
    
    def create_upload_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("YOLO Upload and Detection Settings")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title_label)
        
        # Source type selection
        source_type_widget = QWidget()
        source_type_layout = QHBoxLayout()
        source_type_layout.addWidget(QLabel("Select Source Type:"))
        
        self.video_radio = QRadioButton("Video")
        self.video_radio.setChecked(True)
        self.video_radio.toggled.connect(lambda: self.set_source_type("video"))
        
        self.images_radio = QRadioButton("Images")
        self.images_radio.toggled.connect(lambda: self.set_source_type("images"))
        
        self.rtsp_radio = QRadioButton("RTSP Stream")
        self.rtsp_radio.toggled.connect(lambda: self.set_source_type("rtsp"))
        
        source_type_layout.addWidget(self.video_radio)
        source_type_layout.addWidget(self.images_radio)
        source_type_layout.addWidget(self.rtsp_radio)
        source_type_widget.setLayout(source_type_layout)
        layout.addWidget(source_type_widget)
        
        # Source input section
        source_section = QWidget()
        source_layout = QVBoxLayout()
        
        # Source selection stack
        self.source_stack = QStackedWidget()
    
        # 1. Video upload widget
        video_widget = QWidget()
        video_layout = QHBoxLayout()
        video_layout.setContentsMargins(10, 10, 10, 10)
        self.video_upload_btn = QPushButton("Upload Video")
        self.video_upload_btn.setFixedSize(180, 100)
        self.video_upload_btn.clicked.connect(self.upload_video)
        video_layout.addWidget(self.video_upload_btn)
        video_widget.setLayout(video_layout)
        self.source_stack.addWidget(video_widget)
        
        '''# 2. Images upload widget
        images_widget = QWidget()
        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(10, 10, 10, 10)
        images_layout.addStretch(1)
        self.images_upload_btn = QPushButton("Select Images")
        self.images_upload_btn.setFixedSize(180, 100)
        self.images_upload_btn.clicked.connect(self.upload_images)
        images_layout.addWidget(self.images_upload_btn)
        self.images_count_label = QLabel("No images selected")
        images_layout.addWidget(self.images_count_label)
        images_widget.setLayout(images_layout)
        self.source_stack.addWidget(images_widget)'''
        # 2. Images upload widget
        images_widget = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(10, 10, 10, 10)

# Create a centered horizontal layout for the button and label
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)

# Button and label
        self.images_upload_btn = QPushButton("Select Images")
        self.images_upload_btn.setFixedSize(180, 100)
        self.images_upload_btn.clicked.connect(self.upload_images)

        self.images_count_label = QLabel("No images selected")

# Add to center layout
        center_layout.addWidget(self.images_upload_btn)
        center_layout.addSpacing(20)  # optional spacing between button and label
        center_layout.addWidget(self.images_count_label)

# Add the center layout to outer layout (with stretch to center it vertically)
        outer_layout.addStretch()
        outer_layout.addLayout(center_layout)
        outer_layout.addStretch()

        images_widget.setLayout(outer_layout)
        self.source_stack.addWidget(images_widget)

        
        # 3. RTSP widget
        rtsp_widget = QWidget()
        rtsp_layout = QHBoxLayout()
        rtsp_layout.setContentsMargins(10, 10, 10, 10)
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("Enter RTSP link (e.g., rtsp://example.com/stream)")
        self.rtsp_input.setFixedSize(180, 100)
        rtsp_layout.addWidget(self.rtsp_input)
        self.connect_rtsp_btn = QPushButton("Connect")
        self.connect_rtsp_btn.clicked.connect(self.connect_rtsp)
        self.connect_rtsp_btn.setFixedSize(180, 100)
        rtsp_layout.addWidget(self.connect_rtsp_btn)
        rtsp_widget.setLayout(rtsp_layout)
        self.source_stack.addWidget(rtsp_widget)
        
        source_layout.addWidget(self.source_stack)
        source_section.setLayout(source_layout)
        layout.addWidget(source_section)
        
        # Source info
        self.source_info_label = QLabel("No source selected")
        layout.addWidget(self.source_info_label)
        
        # Parameters section
        params_section = QWidget()
        params_layout = QGridLayout()
    
        def on_video_widget_clicked(self):
            self.reset_ui_and_state()
            self.current_input_type = 'video'
            self.video_widget.show()

        def on_image_widget_clicked(self):
            self.reset_ui_and_state()
            self.current_input_type = 'image'
            self.image_widget.show()

        def on_rtsp_widget_clicked(self):
              self.reset_ui_and_state()
              self.current_input_type = 'rtsp'
              self.rtsp_widget.show()
        
        # 1. Model selection
        params_layout.addWidget(QLabel("1. Select Model:"), 0, 0)
        self.model_select_btn = QPushButton("Browse Models...")
        self.model_select_btn.clicked.connect(self.select_model)
        self.model_label = QLabel("No model selected")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_select_btn)
        model_layout.addWidget(self.model_label)
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        params_layout.addWidget(model_widget, 0, 1)
        
        # 2. Classes file selection
        params_layout.addWidget(QLabel("2. Select Classes:"), 1, 0)
        self.classes_select_btn = QPushButton("Browse Classes...")
        self.classes_select_btn.clicked.connect(self.select_classes)
        self.classes_label = QLabel("Using default COCO classes")
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(self.classes_select_btn)
        classes_layout.addWidget(self.classes_label)
        classes_widget = QWidget()
        classes_widget.setLayout(classes_layout)
        params_layout.addWidget(classes_widget, 1, 1)
        
        # 3. Confidence threshold
        params_layout.addWidget(QLabel("3. Confidence Threshold:"), 2, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Default 0.5
        self.conf_label = QLabel("0.50")
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        conf_widget = QWidget()
        conf_widget.setLayout(conf_layout)
        params_layout.addWidget(conf_widget, 2, 1)
        
        # 4. IoU threshold
        params_layout.addWidget(QLabel("4. IoU Threshold:"), 3, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)  # Default 0.45
        self.iou_label = QLabel("0.45")
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        iou_widget = QWidget()
        iou_widget.setLayout(iou_layout)
        params_layout.addWidget(iou_widget, 3, 1)
        
        # 5. Device selection
        params_layout.addWidget(QLabel("5. Processing Device:"), 4, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        params_layout.addWidget(self.device_combo, 4, 1)
        
        # 6. Class filters
        params_layout.addWidget(QLabel("6. Class Filters (Optional):"), 5, 0)
        self.class_filter_btn = QPushButton("Select Classes to Detect")
        self.class_filter_btn.clicked.connect(self.select_class_filters)
        self.class_filter_label = QLabel("All classes selected")
        class_filter_layout = QHBoxLayout()
        class_filter_layout.addWidget(self.class_filter_btn)
        class_filter_layout.addWidget(self.class_filter_label)
        class_filter_widget = QWidget()
        class_filter_widget.setLayout(class_filter_layout)
        params_layout.addWidget(class_filter_widget, 5, 1)
        
        params_section.setLayout(params_layout)
        layout.addWidget(params_section)
        
        # Start button
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)  # Disabled until source and model are selected
        layout.addWidget(self.start_btn)
        
        page.setLayout(layout)
        return page
    
    def create_detection_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("YOLO Detection Results")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title_label)
        
        # Video/Image display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: black;")
        self.display_label.setMinimumSize(800, 600)
        
        # Wrap display in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.display_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.save_btn)
        # Back to Home button
        self.back_btn = QPushButton("Back to Home")
        self.back_btn.clicked.connect(self.go_to_page1)
        controls_layout.addWidget(self.back_btn)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)
        
        page.setLayout(layout)
        return page
    class YOLOv8Detector:
      def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def predict(self, frame):
        return self.model.predict(frame, imgsz=640)[0]

    def create_comparison_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Compare Two Models on Image/Video/RTSP")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title_label)

        # Combine all available models
        all_models = []
        for model_list in DEFAULT_MODELS.values():
            all_models.extend(model_list)

        # Model selectors
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model 1:"))
        self.model1_combo = QComboBox()
        self.model1_combo.addItems(all_models)
        model_layout.addWidget(self.model1_combo)

        model_layout.addWidget(QLabel("Select Model 2:"))
        self.model2_combo = QComboBox()
        self.model2_combo.addItems(all_models)
        model_layout.addWidget(self.model2_combo)
        layout.addLayout(model_layout)

        # Output displays
        splitter = QSplitter(Qt.Horizontal)

        model1_widget = QWidget()
        model1_layout = QVBoxLayout()
        model1_layout.addWidget(QLabel("Model 1 Output:"))
        self.model1_output = QLabel()
        self.model1_output.setAlignment(Qt.AlignCenter)
        self.model1_output.setStyleSheet("background-color: black;")
        self.model1_output.setMinimumSize(500, 400)
        model1_layout.addWidget(self.model1_output)
        model1_widget.setLayout(model1_layout)

        model2_widget = QWidget()
        model2_layout = QVBoxLayout()
        model2_layout.addWidget(QLabel("Model 2 Output:"))
        self.model2_output = QLabel()
        self.model2_output.setAlignment(Qt.AlignCenter)
        self.model2_output.setStyleSheet("background-color: black;")
        self.model2_output.setMinimumSize(500, 400)
        model2_layout.addWidget(self.model2_output)
        model2_widget.setLayout(model2_layout)

        splitter.addWidget(model1_widget)
        splitter.addWidget(model2_widget)
        layout.addWidget(splitter)

        # Input buttons
        input_layout = QHBoxLayout()
        self.compare_image_btn = QPushButton("Compare on Image")
        self.compare_image_btn.clicked.connect(self.compare_image)
        input_layout.addWidget(self.compare_image_btn)

        self.compare_video_btn = QPushButton("Compare on Video")
        self.compare_video_btn.clicked.connect(self.compare_video)
        input_layout.addWidget(self.compare_video_btn)

        self.compare_rtsp_btn = QPushButton("Compare on RTSP")
        self.compare_rtsp_btn.clicked.connect(self.compare_rtsp)
        input_layout.addWidget(self.compare_rtsp_btn)

        layout.addLayout(input_layout)

        # Back button
        back_btn = QPushButton("Back to Home")
        back_btn.clicked.connect(self.go_to_page1)
        layout.addWidget(back_btn)

        page.setLayout(layout)
        return page

    def cv2_to_pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img)
    class ModelDownloader:
    # Dictionary for model URLs
     MODEL_URLS = {
        "YOLOv8": {
            "YOLOv8n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "YOLOv8s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
            "YOLOv8m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
            "YOLOv8l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
            "YOLOv8x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
        },
        "YOLOv9": {
            "YOLOv9t": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt",
            "YOLOv9s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt",
            "YOLOv9m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9m.pt",
            "YOLOv9c": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt",
            "YOLOv9e": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9e.pt"
        },
        "YOLOv10": {
            "YOLOv10n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt",
            "YOLOv10s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt",
            "YOLOv10m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt",
            "YOLOv10l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10l.pt",
            "YOLOv10x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt"
        },
        "YOLOv11": {
            "YOLOv11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "YOLOv11s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
            "YOLOv11m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            "YOLOv11l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
            "YOLOv11x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
        }
    }
     def __init__(self, model_name):
        self.model_name = model_name
        self.pt_file = f"{self.model_name}.pt"

    def _download_if_needed(self):
        if not os.path.exists(self.pt_file):  # If the .pt file doesn't exist
            print(f"Downloading {self.model_name}...")
            # Check if the model is in the dictionary and fetch the corresponding URL
            for model_group, models in self.MODEL_URLS.items():
                if self.model_name in models:
                    model_url = models[self.model_name]
                    # Download the model file
                    urllib.request.urlretrieve(model_url, self.pt_file)
                    print(f"{self.model_name} downloaded successfully.")
                    return
            print(f"Error: {self.model_name} not found in predefined models.")
    def compare_video(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            model1 = YOLO(self.model1_combo.currentText())
            model2 = YOLO(self.model2_combo.currentText())

            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results1 = model1.predict(frame, imgsz=640)[0]
                results2 = model2.predict(frame, imgsz=640)[0]

                self.model1_output.setPixmap(self.cv2_to_pixmap(results1.plot()))
                self.model2_output.setPixmap(self.cv2_to_pixmap(results2.plot()))
                QApplication.processEvents()

            cap.release()


    def compare_rtsp(self):
        rtsp_link, ok = QInputDialog.getText(None, "RTSP Stream", "Enter RTSP URL:")
        if ok and rtsp_link:
            model1 = YOLO(self.model1_combo.currentText())
            model2 = YOLO(self.model2_combo.currentText())

            cap = cv2.VideoCapture(rtsp_link)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results1 = model1.predict(frame, imgsz=640)[0]
                results2 = model2.predict(frame, imgsz=640)[0]

                self.model1_output.setPixmap(self.cv2_to_pixmap(results1.plot()))
                self.model2_output.setPixmap(self.cv2_to_pixmap(results2.plot()))
                QApplication.processEvents()

            cap.release()

    def compare_image(self):
        model_name = self.model1_combo.currentText()  # example: 'YOLOv8n'
    
    # Download model if needed
        downloader = ModelDownloader(model_name)
        downloader._download_if_needed()

    # Load model using the downloaded file path
        model_path = downloader.pt_file  # This is the correct .pt file path
        model1 = YOLO(model_path)  # ✅ Pass the path, not just 'YOLOv8n'


    def start_detection(self):
        model1_name = self.model1_combo.currentText()
        model2_name = self.model2_combo.currentText()

        model1_path = f"{model1_name}.pt" if not model1_name.endswith(".pt") else model1_name
        model2_path = f"{model2_name}.pt" if not model2_name.endswith(".pt") else model2_name

        self.download_yolo_model(model1_path)
        self.download_yolo_model(model2_path)

        self.detector1 = YOLOv8Detector(model1_path)
        self.detector2 = YOLOv8Detector(model2_path)
        self.stop_flag = False

        self.compare_video()  # or compare_image() or compare_rtsp()
    def stop_detection(self):
       self.stop_flag = True    
    # Navigation methods
    def go_to_page1(self):
        self.stacked_widget.setCurrentIndex(0)
    
    def go_to_page2(self):
        self.stacked_widget.setCurrentIndex(1)
    
    def go_to_page3(self):
        self.stacked_widget.setCurrentIndex(2)
    
    # Helper methods
    def update_conf_label(self):
        value = self.conf_slider.value() / 100
        self.conf_label.setText(f"{value:.2f}")
    
    def update_iou_label(self):
        value = self.iou_slider.value() / 100
        self.iou_label.setText(f"{value:.2f}")
    
    def set_source_type(self, source_type):
        self.source_type = source_type
        if source_type == "video":
            self.source_stack.setCurrentIndex(0)
        elif source_type == "images":
            self.source_stack.setCurrentIndex(1)
        elif source_type == "rtsp":
            self.source_stack.setCurrentIndex(2)
    
    def upload_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if video_path:
              self.source_paths = video_path
              self.source_info_label.setText(f"🎬 Selected video: {os.path.basename(video_path)}")
              self.source_info_label.setVisible(True)
              self.check_start_button_status()

       

    def upload_images(self): 
       folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
       if folder_path:
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]

        if image_paths:
            self.source_paths = image_paths
            self.images_count_label.setText(f"{len(image_paths)} images selected")
            self.source_info_label.setText(f"🖼️ Selected {len(image_paths)} images from folder")
            self.check_start_button_status()

    
    def connect_rtsp(self):
        rtsp_link = self.rtsp_input.text().strip()
        if rtsp_link and (rtsp_link.startswith("rtsp://") or 
                          rtsp_link.startswith("rtmp://") or 
                          rtsp_link.startswith("http://")):
            self.source_paths = rtsp_link
            self.is_rtsp = True
            self.source_info_label.setText(f"📡Connected to RTSP stream: {rtsp_link}")
            self.check_start_button_status()
        else:
            QMessageBox.warning(self, "Invalid RTSP Link", 
                               "Please enter a valid RTSP link starting with rtsp://, rtmp://, or http://")
    
    def select_model(self):
        dialog = ModelSelectDialog(self)
        if dialog.exec_():
            self.model_path = dialog.get_selected_model()
            if self.model_path:
                 clean_path = self.model_path.strip('"').strip("'")
                 self.model_path = clean_path
                 self.model_label.setText(os.path.basename(self.model_path) if os.path.exists(self.model_path) 
                                        else self.model_path)
                 self.check_start_button_status()
    
    def select_classes(self):
        dialog = ClassesDialog(self)
        if dialog.exec_():
           # Step 1: Check for manually entered custom classes
         if dialog.custom_classes:  # <-- custom class list is not empty
            self.class_names = dialog.custom_classes
            self.classes_label.setText(f"{len(self.class_names)} classes selected")
        
        # Step 2: If custom class file selected from file tree
        elif dialog.get_selected_classes_file():
            self.classes_file = dialog.get_selected_classes_file()
            self.classes_label.setText(os.path.basename(self.classes_file))
            self.load_class_names()  # Load class names from file
        
        # Step 3: Default fallback to COCO
        else:
            self.classes_label.setText("Using default COCO classes")
            self.class_names = COCO_CLASSES
    
    def load_class_names(self):
        """Load class names from selected file"""
        try:
            with open(self.classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} classes from {self.classes_file}")
        except Exception as e:
            print(f"Error loading classes file: {e}")
            # Fall back to COCO classes
            self.class_names = COCO_CLASSES
            QMessageBox.warning(self, "Error Loading Classes", 
                               f"Failed to load classes file. Using default COCO classes instead.\nError: {str(e)}")
    
    def select_class_filters(self):
        """Select specific classes to detect"""
        if not self.class_names:
            QMessageBox.warning(self, "No Classes Available", 
                               "Please select a model and classes file first.")
            return
        
        # Create class selector dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Classes to Detect")
        dialog.setGeometry(200, 200, 400, 500)
        
        layout = QVBoxLayout()
        
        # Create scroll area for classes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        class_widget = QWidget()
        class_layout = QVBoxLayout()
        
        # Add checkboxes for all classes
        self.class_checkboxes = []
        for class_name in self.class_names:
            checkbox = QCheckBox(class_name)
            # Check if class was previously selected
            if self.selected_classes and class_name in self.selected_classes:
                checkbox.setChecked(True)
            elif not self.selected_classes:  # Default all checked if none selected before
                checkbox.setChecked(True)
            self.class_checkboxes.append(checkbox)
            class_layout.addWidget(checkbox)
        
        class_widget.setLayout(class_layout)
        scroll.setWidget(class_widget)
        layout.addWidget(scroll)
        
        # Add Select All / Deselect All buttons
        select_buttons = QWidget()
        select_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.toggle_all_classes(True))
        select_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.toggle_all_classes(False))
        select_layout.addWidget(deselect_all_btn)
        
        select_buttons.setLayout(select_layout)
        layout.addWidget(select_buttons)
        
        # Add OK/Cancel buttons
        buttons = QWidget()
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        buttons.setLayout(button_layout)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        # Execute dialog
        if dialog.exec_():
            # Get selected classes
            self.selected_classes = []
            for i, checkbox in enumerate(self.class_checkboxes):
                if checkbox.isChecked():
                    self.selected_classes.append(self.class_names[i])
            
            # Update label
            if not self.selected_classes:
                self.class_filter_label.setText("No classes selected")
            elif len(self.selected_classes) == len(self.class_names):
                self.class_filter_label.setText("All classes selected")
            else:
                self.class_filter_label.setText(f"{len(self.selected_classes)} classes selected")
    
    def toggle_all_classes(self, checked):
        """Toggle all class checkboxes"""
        for checkbox in self.class_checkboxes:
            checkbox.setChecked(checked)
    
    def check_start_button_status(self):
        """Enable start button if source and model are selected"""
        if self.source_paths and self.model_path:
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(False)
    
    def start_detection(self):
        """Start detection process"""
        # Switch to detection page
        self.go_to_page2()
        
        # Reset progress and status
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting detection...")
        
        # Get parameters
        conf_threshold = self.conf_slider.value() / 100
        iou_threshold = self.iou_slider.value() / 100
        device = "cuda" if self.device_combo.currentText() == "CUDA (GPU)" else "cpu"
        
        # Create processing thread
        self.processing_thread = VideoProcessingThread(
            source_path=self.source_paths,
            model_path=self.model_path,
            classes_file=self.classes_file,
            selected_classes=self.selected_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            source_type=self.source_type
        )
        
        # Connect signals
        self.processing_thread.update_frame.connect(self.update_display)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.finished_processing.connect(self.processing_finished)
        
        # Enable/disable buttons
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        
        # Start processing
        self.processing_thread.start()
        
        # Update status
        self.status_label.setText("Processing... Please wait.")
    
    def update_display(self, frame):
        """Update display with current frame"""
        # Convert OpenCV frame to QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        convert_to_qt_format = convert_to_qt_format.rgbSwapped()
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        
        # Resize pixmap to fit display while maintaining aspect ratio
        display_size = self.display_label.size()
        pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio)
        
        # Set pixmap to display label
        self.display_label.setPixmap(pixmap)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def processing_finished(self, message):
        """Handle processing finished"""
        self.status_label.setText(message)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # If there's a valid output path
        if hasattr(self.processing_thread, 'output_path') and self.processing_thread.output_path:
            self.output_path = self.processing_thread.output_path
        
        # Clean up thread
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
    
    def stop_processing(self):
        """Stop processing thread"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_label.setText("Processing stopped")
            self.stop_btn.setEnabled(False)
    
    def save_results(self):
        """Save detection results"""
        if not self.output_path:
            QMessageBox.information(self, "No Results", "No detection results available to save.")
            return
       
        # Ask user for save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        # Copy output files to user selected directory
        try:
            # For video
            if os.path.isfile(self.output_path):
                dest_file = os.path.join(save_dir, os.path.basename(self.output_path))
                if os.path.exists(dest_file):
                    # Add timestamp to avoid overwriting existing files
                    base, ext = os.path.splitext(os.path.basename(self.output_path))
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dest_file = os.path.join(save_dir, f"{base}_{timestamp}{ext}")
                
                # Copy file
                import shutil
                shutil.copy2(self.output_path, dest_file)
                self.status_label.setText(f"Results saved to {dest_file}")
            
            # For directory (images)
            elif os.path.isdir(self.output_path):
                dest_dir = os.path.join(save_dir, os.path.basename(self.output_path))
                if os.path.exists(dest_dir):
                    # Add timestamp to avoid overwriting existing directory
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dest_dir = f"{dest_dir}_{timestamp}"
                
                # Copy directory
                import shutil
                shutil.copytree(self.output_path, dest_dir)
                self.status_label.setText(f"Results saved to {dest_dir}")
            
            QMessageBox.information(self, "Save Successful", f"Results saved successfully to the selected directory.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving results: {str(e)}")
    
    def load_original(self):
        """Load original image for comparison"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Original Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if image_path:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                QMessageBox.warning(self, "Error", "Failed to load image")
                return
            
            # Convert to QPixmap and display
            h, w, ch = img.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            convert_to_qt_format = convert_to_qt_format.rgbSwapped()
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio)
            
            # Display
            self.original_label.setPixmap(pixmap)
    
    def load_processed(self):
        """Load processed image for comparison"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Processed Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if image_path:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                QMessageBox.warning(self, "Error", "Failed to load image")
                return
            
            # Convert to QPixmap and display
            h, w, ch = img.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            convert_to_qt_format = convert_to_qt_format.rgbSwapped()
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.processed_label.size(), Qt.KeepAspectRatio)
            
            # Display
            self.processed_label.setPixmap(pixmap)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>YOLOv8 Video Processing and Detection Tool</h2>
        
        <h3>Quick Start Guide:</h3>
        
        <h4>1. Upload Page:</h4>
        <ul>
            <li>Select source type (Video, Images, or RTSP stream)</li>
            <li>Upload your source file(s) or enter RTSP URL</li>
            <li>Select a YOLOv8 model or use one of the predefined models</li>
            <li>Optionally select a custom classes file</li>
            <li>Adjust confidence and IoU thresholds</li>
            <li>Choose processing device (GPU or CPU)</li>
            <li>Optionally filter specific classes to detect</li>
            <li>Click "Start Detection" to begin processing</li>
        </ul>
        
        <h4>2. Detection Page:</h4>
        <ul>
            <li>View the detection results in real-time</li>
            <li>Monitor progress through the progress bar</li>
            <li>Stop processing if needed</li>
            <li>Save results to your chosen location</li>
        </ul>
        
        <h4>3. Comparison Page:</h4>
        <ul>
            <li>Load original and processed images</li>
            <li>Compare detection results side by side</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Help")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())