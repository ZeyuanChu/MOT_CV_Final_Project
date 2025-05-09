import os
import sys
import logging
import json
from collections import defaultdict
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from collections import Counter
import fiftyone.utils.eval as foue
import matplotlib.pyplot as plt

# Fix for OpenMP conflict - this needs to be at the very start of the script
# before importing any numerical libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Alternative approach if the above doesn't work:
# os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os
import argparse

# Import the Sort tracker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path
from sort.sort import Sort  # Import the SORT tracker class
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
import math

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       module='threadpoolctl')

# Add after the initial imports, before any matplotlib usage
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Parse command line arguments
parser = argparse.ArgumentParser(description='Object tracking with YOLO')
parser.add_argument('--input_file', type=str, required=False, 
                   default='input_files/traffic.mp4', 
                   help='Path to input video file (default: input_files/traffic.mp4)')
parser.add_argument('--box_type', type=str, required=False, default='square', choices=['square', 'overlay'], help='Type of bounding box: square or segmentation overlay')
parser.add_argument('--output_file', type=str, required=False, default='result', help='Custom output filename (without extension)')
parser.add_argument('--conf_thresh', type=float, default=0.3,
                   help='Confidence threshold for YOLO detections (0-1)')
parser.add_argument('--iou_thresh', type=float, default=0.3,
                   help='IOU threshold for tracking (0-1)')
args = parser.parse_args()

# Initialize logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Update model loading code
def get_model_path(model_name):
    """Get path to model, download if doesn't exist"""
    # Get absolute path to models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        logging.info(f"Downloading {model_name} to models directory...")
        # Download model using YOLO's download method
        model = YOLO(model_name)
        
        # Get the downloaded model path (usually in current directory)
        downloaded_path = os.path.join(os.getcwd(), model_name)
        
        # Move the model to our models directory
        if os.path.exists(downloaded_path):
            # If models directory doesn't exist, create it
            os.makedirs(models_dir, exist_ok=True)
            # Move the file
            os.rename(downloaded_path, model_path)
            logging.info(f"Model moved to {model_path}")
        else:
            logging.error(f"Downloaded model not found at {downloaded_path}")
            return model_name  # Fallback to default location
    else:
        logging.info(f"Loading existing model from {model_path}")
    
    return model_path

# Update model loading section
model_name = 'yolo11x.pt'  # Using v8 as it's more stable for local saving
model = YOLO(get_model_path(model_name))
logging.info(f"Loaded YOLO detection model: {model_name}")

# If using overlay mode, ensure we get segmentation results
if args.box_type == 'overlay':
    seg_model_name = 'yolo11x-seg.pt'
    seg_model = YOLO(get_model_path(seg_model_name))
    logging.info(f"Loaded YOLO-seg model for overlay mode: {seg_model_name}")

# Initialize SORT tracker
mot_tracker = Sort(
    max_age=10,        # Maximum frames to keep track of objects that are not detected
    min_hits=3,        # Minimum detections before track is initialized
    iou_threshold=args.iou_thresh  # Use command line argument
)

# Open the video
video_path = args.input_file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the video filename without extension
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Clean up any existing datasets
for dataset_name in [f"{video_name}_tracking", f"{video_name}_mot_eval"]:
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

# Update directory structure creation
base_output_dir = os.path.join('output_files', video_name, args.box_type)
video_output_dir = base_output_dir
eval_output_dir = os.path.join(base_output_dir, 'evaluation_results')

# Create output directories
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(eval_output_dir, exist_ok=True)

# Set up video writer with custom output filename if provided
if args.output_file:
    output_filename = f'{args.output_file}.mp4'
else:
    output_filename = f'output_{video_name}_{args.box_type}.mp4'

# Keep video in base directory
out_path = os.path.join(video_output_dir, output_filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Initialize lists to store tracking data for evaluation
tracking_results = []
ground_truth = []

# For tracking statistics
track_history = defaultdict(lambda: {"class_id": None, "class_name": "", "frames": 0, "confidence": 0, 
                                   "dominant_color": None, "color_name": "", "size": "", 
                                   "size_percentage": 0, "previous_pos": None, "speed": 0, 
                                   "movement": "stationary"})
colors = np.random.rand(32, 3) * 255  # Random colors for visualization

# Resize display dimensions
display_width = int(width * 0.6)  # Main display takes 60% of original width
display_height = int(height * 0.7)

# Create a wider info panel
info_panel_width = 1000  # Fixed width for info panel
info_panel_height = display_height

# Create a single window
cv2.namedWindow('Combined View', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Combined View', display_width + info_panel_width, display_height)

frame_count = 0
active_tracks = {}  # To store currently active tracks for the info panel

# Log video processing start
logging.info(f"Starting processing video: {video_path}")

# Helper function for IoU calculation - Move this to the top of the file, after imports
def calculate_iou(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    xx1 = max(x1_1, x1_2)
    yy1 = max(y1_1, y1_2)
    xx2 = min(x2_1, x2_2)
    yy2 = min(y2_1, y2_2)
    
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    intersection = w * h
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

# Helper function to get dominant color
def get_dominant_color(image, mask=None):
    try:
        # Ensure image has 3 channels (BGR)
        if image.shape[-1] != 3:
            print(f"Warning: ROI doesn't have 3 channels, has {image.shape}")
            return None
            
        # If mask is provided, use it to extract pixels
        if mask is not None:
            # Make sure mask dimensions match image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Make sure mask is boolean or uint8
            if mask.dtype != bool:
                mask = mask > 0
                
            # Check if we have enough pixels in the mask
            if np.sum(mask) < 100:  # Need at least 100 pixels
                return None
                
            # Extract pixels using the mask
            masked_image = image.copy()
            masked_image[~mask] = [0, 0, 0]  # Set background to black
            
            # Get non-zero pixels (foreground)
            non_zero_indices = np.where(np.any(masked_image != [0, 0, 0], axis=-1))
            if len(non_zero_indices[0]) < 100:  # Additional safety check
                return None
                
            pixels = masked_image[non_zero_indices]
        else:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
        
        # Make sure we have pixels to analyze
        if len(pixels) == 0:
            return None
            
        # Resize to a smaller number of pixels for faster processing
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Use simple K-means in BGR space directly instead of HSV conversion
        # This avoids color space conversion issues
        kmeans_model = KMeans(n_clusters=5, n_init=10, random_state=42)
        kmeans_model.fit(pixels)
        
        # Get the cluster centers and sizes
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        
        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Find the largest cluster
        largest_cluster_idx = np.argmax(counts)
        dominant_color = centers[largest_cluster_idx].astype(int)
        
        # Check if the dominant color is too dark (possibly shadow)
        brightness = np.mean(dominant_color)
        if brightness < 30:
            # Try the second largest cluster if available
            if len(counts) > 1:
                # Get the second largest cluster
                sorted_indices = np.argsort(counts)[::-1]
                second_largest_idx = sorted_indices[1]
                second_color = centers[second_largest_idx].astype(int)
                second_brightness = np.mean(second_color)
                
                # Use second largest if it's brighter
                if second_brightness > 30:
                    dominant_color = second_color
        
        return dominant_color
        
    except Exception as e:
        print(f"Error in color detection: {str(e)}")
        return None

# Helper function to map RGB to color name
def get_color_name(rgb):
    # Convert BGR to RGB (OpenCV uses BGR)
    rgb = rgb[::-1]
    
    # Define expanded colors and their RGB values
    colors = {
        "red": (255, 0, 0),
        "dark red": (139, 0, 0),
        "salmon": (250, 128, 114),
        "orange": (255, 165, 0),
        "gold": (255, 215, 0),
        "yellow": (255, 255, 0),
        "lime": (50, 205, 50),
        "green": (0, 128, 0),
        "dark green": (0, 100, 0),
        "teal": (0, 128, 128),
        "cyan": (0, 255, 255),
        "light blue": (173, 216, 230),
        "blue": (0, 0, 255),
        "navy": (0, 0, 128),
        "purple": (128, 0, 128),
        "magenta": (255, 0, 255),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "tan": (210, 180, 140),
        "beige": (245, 245, 220),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "light gray": (211, 211, 211),
        "dark gray": (64, 64, 64),
        "black": (0, 0, 0)
    }
    
    # Convert RGB values to HSV for better color matching
    rgb_array = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
    
    # Extract HSV components
    h, s, v = hsv
    
    # Special case for low saturation (grayscale)
    if s < 30:
        if v < 50:
            return "black"
        elif v < 150:
            return "gray"
        else:
            return "white"
    
    # Special case for browns (which are tricky in HSV)
    if (h < 30 or h > 330) and s > 30 and v < 150:
        return "brown"
    
    # Use hue to determine basic color
    if h < 15 or h > 330:
        return "red"
    elif h < 45:
        return "orange"
    elif h < 75:
        return "yellow"
    elif h < 150:
        return "green"
    elif h < 195:
        return "cyan"
    elif h < 240:
        return "blue"
    elif h < 270:
        return "purple"
    elif h < 330:
        return "magenta"
    
    # Fallback to traditional RGB distance if HSV doesn't work well
    min_dist = float('inf')
    closest_color = "unknown"
    
    for color_name, color_rgb in colors.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    
    return closest_color

# Helper function to calculate object size
def calculate_size(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Calculate percentage of frame
    frame_area = frame_width * frame_height
    percentage = (area / frame_area) * 100
    
    # Categorize size
    if percentage < 1:
        return "small", percentage
    elif percentage < 10:
        return "medium", percentage
    else:
        return "large", percentage

# Helper function to calculate speed
def calculate_speed(current_pos, previous_pos, fps):
    if previous_pos is None:
        return 0, "stationary"
    
    # Calculate center points
    c1 = ((current_pos[0] + current_pos[2]) / 2, (current_pos[1] + current_pos[3]) / 2)
    c2 = ((previous_pos[0] + previous_pos[2]) / 2, (previous_pos[1] + previous_pos[3]) / 2)
    
    # Calculate distance moved
    distance = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    # Convert to speed (pixels per second)
    speed = distance * fps
    
    # Categorize movement
    if speed < 10:
        return speed, "stationary"
    elif speed < 100:
        return speed, "slow"
    elif speed < 300:
        return speed, "medium"
    else:
        return speed, "fast"

# After YOLO model initialization, add COCO class names
COCO_CLASSES = model.names  # YOLO model comes with COCO classes

# Update detection logging
def log_detections(frame_count, detections, classes=COCO_CLASSES):
    """Log detection details with class names"""
    class_counts = Counter()
    for det in detections:
        cls_id = int(det[5])
        class_counts[classes[cls_id]] += 1
    
    # Format class counts for logging
    class_info = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
    logging.info(f"[SORT] Frame {frame_count}: received {len(detections)} detections ({class_info})")
    return class_counts

# Add near the top where we create other directories
temp_frames_dir = os.path.join('temp_frames')
os.makedirs(temp_frames_dir, exist_ok=True)

try:
    # Main processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Run YOLO detection on the frame
        results = model(frame, verbose=False)
        
        # If in overlay mode, also get segmentation results
        if args.box_type == 'overlay':
            seg_results = seg_model(frame, verbose=False)
        
        # Get the detections in the format [x1, y1, x2, y2, confidence, class_id]
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].cpu().numpy()
                # Only add detections above confidence threshold
                if conf > args.conf_thresh:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf, cls])
        
        detections = np.array(detections)
        
        # Clear active tracks for this frame
        active_tracks = {}
        
        # Format for SORT: [x1, y1, x2, y2, confidence]
        if len(detections) > 0:
            sort_detections = detections[:, 0:5]
            
            # Update SORT tracker
            tracked_objects = mot_tracker.update(sort_detections)
            
            # Log detections with class information
            class_counts = log_detections(frame_count, detections)
        else:
            # No detections, still need to update tracker
            tracked_objects = mot_tracker.update()
        
        # Draw boxes and tracking IDs
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Find the class for this tracked object
            if len(detections) > 0:
                # Find closest detection to this track by calculating IoU
                ious = []
                for det in detections:
                    det_box = det[:4]
                    track_box = track[:4]
                    # Calculate IoU between track and detection
                    xx1 = max(det_box[0], track_box[0])
                    yy1 = max(det_box[1], track_box[1])
                    xx2 = min(det_box[2], track_box[2])
                    yy2 = min(det_box[3], track_box[3])
                    
                    w = max(0, xx2 - xx1)
                    h = max(0, yy2 - yy1)
                    
                    intersection = w * h
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                    union = det_area + track_area - intersection
                    
                    iou = intersection / union if union > 0 else 0
                    ious.append((iou, det))
                
                # Use the detection with highest IoU
                if ious:
                    best_match = max(ious, key=lambda x: x[0])
                    if best_match[0] > 0.5:  # Only use if IoU is sufficiently high
                        class_id = int(best_match[1][5])
                        class_name = model.names[class_id]
                        confidence = best_match[1][4]
                        
                        # Extract the region of interest for color analysis
                        roi = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Get dominant color
                        dominant_color = None
                        color_name = "unknown"
                        mask_data = None
                        roi_mask = None  # Initialize roi_mask to None by default
                        
                        # If in overlay mode, try to use the segmentation mask for color analysis
                        if args.box_type == 'overlay':
                            for seg_result in seg_results:
                                if hasattr(seg_result, 'masks') and seg_result.masks is not None:
                                    masks = seg_result.masks
                                    boxes = seg_result.boxes
                                    
                                    for i, mask_box in enumerate(boxes):
                                        m_x1, m_y1, m_x2, m_y2 = mask_box.xyxy[0].cpu().numpy()
                                        m_cls = mask_box.cls[0].cpu().numpy()
                                        
                                        box_iou = calculate_iou((x1, y1, x2, y2), (m_x1, m_y1, m_x2, m_y2))
                                        if box_iou > 0.5 and m_cls == class_id:
                                            try:
                                                # Get mask for this object
                                                mask_data = masks[i].data
                                                
                                                # Convert to numpy if it's a tensor
                                                if hasattr(mask_data, 'cpu'):
                                                    mask_data = mask_data.cpu().numpy()
                                                
                                                # Ensure correct dimensions (H,W)
                                                if len(mask_data.shape) == 3:
                                                    mask_data = mask_data[0]  # Take first channel if multiple
                                                
                                                # Resize if needed
                                                if mask_data.shape != (frame.shape[0], frame.shape[1]):
                                                    mask_data = cv2.resize(
                                                        mask_data.astype(np.uint8), 
                                                        (frame.shape[1], frame.shape[0])
                                                    )
                                                
                                                # Create boolean mask
                                                mask_bool = mask_data > 0.5
                                                
                                                # --- Mask smoothing (morphological closing + opening) ---
                                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                                                mask_uint = mask_bool.astype(np.uint8)
                                                mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel)
                                                mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN, kernel)
                                                mask_bool = mask_uint.astype(bool)
                                                # -------------------------------------------------------
                                                
                                                # Get region of interest with mask
                                                roi_mask = mask_bool[int(y1):int(y2), int(x1):int(x2)]
                                                break
                                            except Exception as e:
                                                print(f"Error getting mask for color analysis: {str(e)}")
                                                roi_mask = None
                        
                        # Get dominant color (use mask if available)
                        if roi.size > 0:  # Make sure ROI is not empty
                            dominant_color = get_dominant_color(roi, roi_mask)
                            if dominant_color is not None:
                                color_name = get_color_name(dominant_color)
                        
                        # Calculate object size
                        size_category, size_percentage = calculate_size((x1, y1, x2, y2), width, height)
                        
                        # Calculate speed and movement
                        previous_pos = track_history[track_id]["previous_pos"]
                        speed, movement = calculate_speed((x1, y1, x2, y2), previous_pos, fps)
                        
                        # Update track history
                        if track_id not in track_history or track_history[track_id]["class_id"] is None:
                            track_history[track_id]["class_id"] = class_id
                            track_history[track_id]["class_name"] = class_name
                        
                        # Update object characteristics
                        track_history[track_id]["confidence"] = confidence
                        track_history[track_id]["frames"] += 1
                        track_history[track_id]["dominant_color"] = dominant_color
                        track_history[track_id]["color_name"] = color_name
                        track_history[track_id]["size"] = size_category
                        track_history[track_id]["size_percentage"] = size_percentage
                        track_history[track_id]["previous_pos"] = (x1, y1, x2, y2)
                        track_history[track_id]["speed"] = speed
                        track_history[track_id]["movement"] = movement
                        
                        # Add to active tracks for info panel
                        active_tracks[track_id] = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "position": (int(x1), int(y1), int(x2), int(y2)),
                            "color_name": color_name,
                            "size": size_category,
                            "movement": movement
                        }
                        
                        # Draw the box and label based on the box_type
                        color = colors[track_id % 32].tolist()
                        
                        if args.box_type == 'square':
                                # Draw thicker bounding box
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                                
                                # Draw ID with larger font and better visibility
                                text = f"ID: {track_id}"
                                font_scale = 1.5
                                thickness = 3
                                
                                # Get text size for background
                                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                                     font_scale, thickness)
                                
                                # Calculate text position to keep it in frame
                                text_x = int(x1)
                                text_y = int(y1) - 10  # Default position above box
                                
                                # Adjust if too close to left edge
                                if text_x < 0:
                                    text_x = 0
                                
                                # Adjust if too close to right edge
                                if text_x + text_width > frame.shape[1]:
                                    text_x = frame.shape[1] - text_width - 10
                                
                                # If too close to top, put text below box instead
                                if text_y - text_height < 0:
                                    text_y = int(y1) + text_height + 30
                                
                                # Draw white background rectangle for black text
                                cv2.rectangle(frame, 
                                             (text_x, text_y - text_height - 5), 
                                             (text_x + text_width + 10, text_y + 5), 
                                             (255, 255, 255),  # White background
                                             -1)  # Filled rectangle
                                
                                # Draw black text
                                cv2.putText(frame, text, 
                                            (text_x + 5, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),  # Black text
                                            thickness)
                        else:  # 'overlay' mode
                            # Find the corresponding segmentation mask if available
                            if args.box_type == 'overlay':
                                # Search for a segmentation mask in the region of the bounding box
                                for seg_result in seg_results:
                                    if hasattr(seg_result, 'masks') and seg_result.masks is not None:
                                        masks = seg_result.masks
                                        boxes = seg_result.boxes
                                        
                                        # Find matching segmentation mask for this detection
                                        for i, mask_box in enumerate(boxes):
                                            m_x1, m_y1, m_x2, m_y2 = mask_box.xyxy[0].cpu().numpy()
                                            m_cls = mask_box.cls[0].cpu().numpy()
                                            
                                            # If box and class generally match the detection
                                            box_iou = calculate_iou((x1, y1, x2, y2), (m_x1, m_y1, m_x2, m_y2))
                                            if box_iou > 0.5 and m_cls == class_id:
                                                try:
                                                    # Get mask for this object - FIXED
                                                    # Access the raw data using .data property
                                                    mask_data = masks[i].data
                                                    
                                                    # Convert to numpy if it's a tensor
                                                    if hasattr(mask_data, 'cpu'):
                                                        mask_data = mask_data.cpu().numpy()
                                                    
                                                    # Ensure correct dimensions (H,W)
                                                    if len(mask_data.shape) == 3:
                                                        mask_data = mask_data[0]  # Take first channel if multiple
                                                    
                                                    # Resize if needed
                                                    if mask_data.shape != (frame.shape[0], frame.shape[1]):
                                                        mask_data = cv2.resize(
                                                            mask_data.astype(np.uint8), 
                                                            (frame.shape[1], frame.shape[0])
                                                        )
                                                    
                                                    # Create boolean mask
                                                    mask_bool = mask_data > 0.5
                                                    
                                                    # Create overlay
                                                    overlay = frame.copy()
                                                    overlay[mask_bool] = color
                                                    
                                                    # Blend with original frame
                                                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                                                    
                                                    # print(f"Applied mask for track {track_id}")
                                                except Exception as e:
                                                    print(f"Error applying mask: {str(e)}")
                                                    cv2.rectangle(frame, (int(x1), int(y1)), 
                                                                 (int(x2), int(y2)), color, 4)
                                                break
    
        # Resize the frame for display
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Add video name at the top of the tracking window
        cv2.putText(display_frame, f"Video: {video_name}", (20, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Create info panel with tracking data
        info_panel = np.zeros((info_panel_height, info_panel_width, 3), dtype=np.uint8)
        
        # Add video name at the top of the info panel
        cv2.putText(info_panel, f"Video: {video_name}", (20, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Add box type information
        cv2.putText(info_panel, f"Mode: {args.box_type}", (20, 70), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Add title with larger, bolder text
        cv2.putText(info_panel, "Object Tracking Information", (20, 110), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        
        # Add column headers with larger text and more space
        header = "ID | Class          | Conf | Color      | Size     | Movement"  # More spacing
        cv2.putText(info_panel, header, (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.line(info_panel, (20, 180), (info_panel_width-20, 180), (150, 150, 150), 2)
        
        # Add tracking data with larger, bolder text
        y_pos = 230
        for track_id, track_info in sorted(active_tracks.items()):
            # Get the color for this track ID
            color = colors[track_id % 32].tolist()
            
            # Format each field with more width
            id_str = f"{track_id:3d}"
            class_str = f"{track_info['class_name'][:15]:15s}"  # Increased width
            conf_str = f"{int(track_info['confidence'] * 100):3d}%"
            color_str = f"{track_info['color_name'][:10]:10s}"  # Increased width
            size_str = f"{track_info['size']:8s}"  # Increased width
            move_str = f"{track_info['movement']:10s}"  # Fixed width for movement
            
            # Combine with proper spacing
            text = f"{id_str} | {class_str} | {conf_str} | {color_str} | {size_str} | {move_str}"
            
            cv2.putText(info_panel, text, (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y_pos += 50  # Increased spacing between rows
            
            # Break if we run out of space
            if y_pos > info_panel_height - 50:
                cv2.putText(info_panel, "...", (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                break
        
        # Combine display frame and info panel horizontally
        combined_view = np.hstack((display_frame, info_panel))
        
        # Display the combined view
        cv2.imshow('Combined View', combined_view)
        out.write(frame)  # Write original size to output video
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Inside the main tracking loop, after processing each frame:
        if len(detections) > 0:
            # Store tracking results for evaluation
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                # Find matching detection to get class info
                matching_det = None
                for det in detections:
                    det_box = det[:4]
                    if calculate_iou((x1, y1, x2, y2), det_box) > 0.5:
                        matching_det = det
                        break

                tracking_results.append({
                    "frame": frame_count,
                    "id": int(track_id),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(active_tracks.get(int(track_id), {}).get('confidence', 1.0)),
                    "class_id": int(matching_det[5]) if matching_det is not None else 0  # Add class ID
                })
                
            # Store ground truth with class information
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                ground_truth.append({
                    "frame": frame_count,
                    "id": len(ground_truth) + 1,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class_id": int(cls)  # Add class ID
                })
            
            # Save frame for evaluation
            frame_path = os.path.join(temp_frames_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)

    # After the main processing loop (after writing the last frame)
    try:
        # Import evaluation
        from evaluation import evaluate_tracking_performance
        
        # Run evaluation
        logging.info("Starting evaluation...")
        results = evaluate_tracking_performance(
            tracking_results,
            ground_truth,
            frame_count,
            video_name,
            model.names,
            output_dir=eval_output_dir  # Add this parameter
        )
        
        # Cleanup temp frames
        if os.path.exists(temp_frames_dir):
            import shutil
            shutil.rmtree(temp_frames_dir)
        
        logging.info("Evaluation complete. Results saved in evaluation_results/")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
    finally:
        # Always cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    logging.info("\nProcess interrupted by user (Ctrl+C)")
    logging.warning("Video processing stopped early. Some features may be incomplete:")
    logging.warning("- Video file may be incomplete")
    logging.warning("- Interactive visualization not available")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Remove incomplete temp files
    if os.path.exists(temp_frames_dir):
        import shutil
        shutil.rmtree(temp_frames_dir)
    
    sys.exit(0)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    sys.exit(1)

finally:
    # Always try to clean up
    try:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except:
        pass