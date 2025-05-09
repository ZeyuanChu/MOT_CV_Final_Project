import fiftyone as fo
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from collections import defaultdict
import motmetrics as mm
from motmetrics import metrics
import cv2
from fiftyone.core.metadata import ImageMetadata
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Monkey patch np.asfarray before importing motmetrics
original_asfarray = getattr(np, 'asfarray', None)
np.asfarray = lambda x: np.asarray(x, dtype=float)

import motmetrics as mm
from motmetrics import metrics

def evaluate_tracking_performance(tracking_results, ground_truth, frame_count, video_name, model_classes, output_dir=None):
    """
    Comprehensive evaluation combining custom metrics and FiftyOne evaluation
    """
    # If output_dir is not provided, use the default path
    if output_dir is None:
        eval_output_dir = os.path.join('output_files', video_name, 'evaluation_results')
    else:
        eval_output_dir = output_dir
        
    os.makedirs(eval_output_dir, exist_ok=True)

    def save_plot(fig, name):
        """Helper to save plots"""
        plot_path = os.path.join(eval_output_dir, f"{name}.png")
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"Saved {name} plot to: {plot_path}")

    # Initialize FiftyOne dataset for evaluation
    dataset_name = f"{video_name}_eval"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    # Add frames to dataset
    for frame_num in range(1, frame_count + 1):
        frame_path = f"temp_frames/frame_{frame_num}.jpg"
        if not os.path.exists(frame_path):
            continue

        # Create sample with metadata
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            height, width = frame.shape[:2]
            
            sample = fo.Sample(filepath=os.path.abspath(frame_path))
            sample.metadata = ImageMetadata(
                size_bytes=os.path.getsize(frame_path),
                mime_type="image/jpeg",
                width=width,
                height=height,
                num_channels=3
            )
            
            # Add predictions and ground truth
            frame_preds = [p for p in tracking_results if p["frame"] == frame_num]
            frame_gt = [g for g in ground_truth if g["frame"] == frame_num]

            # Convert to FiftyOne format
            sample["predictions"] = fo.Detections(detections=[
                fo.Detection(
                    label=model_classes[p["class_id"]],
                    bounding_box=normalize_bbox(p["bbox"], sample),
                    confidence=p.get("confidence", 1.0)
                ) for p in frame_preds
            ])

            sample["ground_truth"] = fo.Detections(detections=[
                fo.Detection(
                    label=model_classes[g["class_id"]],
                    bounding_box=normalize_bbox(g["bbox"], sample)
                ) for g in frame_gt
            ])

            dataset.add_sample(sample)
        except Exception as e:
            logging.error(f"Error processing frame {frame_num}: {str(e)}")

    # Run FiftyOne evaluation
    results = dataset.evaluate_detections(
        "predictions",
        "ground_truth",
        eval_key="eval",
        compute_mAP=True,
        method="coco"
    )

    # Get list of detected classes
    detected_classes = set()
    for track in tracking_results:
        class_name = model_classes[track["class_id"]]
        detected_classes.add(class_name)

    # 1. ROC Curves per class (only for detected classes)
    roc_fig = plt.figure(figsize=(12, 8))
    for class_name in detected_classes:
        # Get class ID
        class_id = [k for k, v in model_classes.items() if v == class_name][0]
        
        # Get ground truth and predictions for this class
        y_true = []
        y_score = []
        
        # For each frame, build binary labels and scores
        for frame in range(1, frame_count + 1):
            # Get ground truth boxes for this frame and class
            frame_gt = [g for g in ground_truth if g["frame"] == frame and g["class_id"] == class_id]
            frame_pred = [p for p in tracking_results if p["frame"] == frame and p["class_id"] == class_id]
            
            # For each prediction, check if it matches any ground truth
            for pred in frame_pred:
                matched = False
                for gt in frame_gt:
                    if calculate_box_iou(pred["bbox"], gt["bbox"]) > 0.5:
                        matched = True
                        break
                y_true.append(1 if matched else 0)  # 0 for false positives
                y_score.append(pred.get("confidence", 1.0))
            
            # Add false negatives (missed ground truths)
            for gt in frame_gt:
                matched = False
                for pred in frame_pred:
                    if calculate_box_iou(pred["bbox"], gt["bbox"]) > 0.5:
                        matched = True
                        break
                if not matched:
                    y_true.append(1)
                    y_score.append(0.0)  # Lowest possible score for missed detection
            
            # Add true negatives (correct non-detections)
            # Get all other class predictions for this frame
            other_preds = [p for p in tracking_results if p["frame"] == frame and p["class_id"] != class_id]
            for other_pred in other_preds:
                y_true.append(0)  # It's correct that this isn't our class
                y_score.append(other_pred.get("confidence", 1.0))

        if len(y_true) > 0:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend()
    plt.grid(True)
    save_plot(roc_fig, "roc_curves")

    # 2. Precision-Recall Curves (only for detected classes)
    pr_fig = plt.figure(figsize=(12, 8))
    for class_name in detected_classes:
        # Get class ID
        class_id = [k for k, v in model_classes.items() if v == class_name][0]
        
        # Get ground truth and predictions for this class
        y_true = []
        y_score = []
        
        # For each frame, build binary labels and scores
        for frame in range(1, frame_count + 1):
            # Get ground truth boxes for this frame and class
            frame_gt = [g for g in ground_truth if g["frame"] == frame and g["class_id"] == class_id]
            
            # Get ALL predictions for this frame (not just this class)
            frame_pred = [p for p in tracking_results if p["frame"] == frame]
            
            # For each prediction, check if it matches any ground truth of this class
            for pred in frame_pred:
                matched = False
                # Only check for matches if prediction is of the target class
                if pred["class_id"] == class_id:
                    for gt in frame_gt:
                        if calculate_box_iou(pred["bbox"], gt["bbox"]) > 0.5:
                            matched = True
                            break
                    y_true.append(1 if matched else 0)  # False positive if no match
                else:
                    y_true.append(0)  # Prediction of wrong class = negative example
                y_score.append(pred.get("confidence", 1.0))
            
            # Add false negatives (missed ground truths)
            for gt in frame_gt:
                matched = False
                for pred in frame_pred:
                    if pred["class_id"] == class_id and calculate_box_iou(pred["bbox"], gt["bbox"]) > 0.5:
                        matched = True
                        break
                if not matched:
                    y_true.append(1)  # Missed ground truth = false negative
                    y_score.append(0.0)  # Lowest possible score
        
        if len(y_true) > 0:
            # Convert to numpy arrays for sklearn
            y_true = np.array(y_true)
            y_score = np.array(y_score)
            
            # Compute PR curve and AP
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            
            # Plot the curve
            plt.plot(recall, precision, label=f'{class_name} (AP={ap:.3f})')
            
            # Log some statistics to help debug
            logging.info(f"\nClass {class_name} statistics:")
            logging.info(f"Total examples: {len(y_true)}")
            logging.info(f"Positive examples: {np.sum(y_true == 1)}")
            logging.info(f"Negative examples: {np.sum(y_true == 0)}")
            logging.info(f"Average precision: {ap:.3f}")
    
    plt.title('Precision-Recall Curves by Class')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    save_plot(pr_fig, "precision_recall")

    # 3. IoU Curve
    iou_fig = plt.figure(figsize=(12, 8))
    iou_thresholds = np.linspace(0.1, 0.9, 30)
    iou_scores = calculate_iou_curve(tracking_results, ground_truth, iou_thresholds)
    plt.plot(iou_thresholds, iou_scores)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Detection Rate')
    plt.title('IoU Performance Curve')
    plt.grid(True)
    save_plot(iou_fig, "iou_curve")

    # 4. MOTA and 5. IDF1 over time
    try:
        acc = mm.MOTAccumulator(auto_id=True)
        mota_scores = []
        idf1_scores = []
        frame_numbers = range(1, frame_count + 1)

        for frame_num in frame_numbers:
            frame_gt = [g for g in ground_truth if g["frame"] == frame_num]
            frame_pred = [p for p in tracking_results if p["frame"] == frame_num]
            
            gt_ids = [g["id"] for g in frame_gt]
            gt_boxes = [g["bbox"] for g in frame_gt]
            
            pred_ids = [p["id"] for p in frame_pred]
            pred_boxes = [p["bbox"] for p in frame_pred]

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                acc.update(gt_ids, pred_ids, distances)
            
            # Calculate metrics for this frame
            frame_summary = calculate_mot_metrics(acc)
            try:
                mota_scores.append(float(frame_summary['mota'].iloc[0]))
                idf1_scores.append(float(frame_summary['idf1'].iloc[0]))
            except:
                mota_scores.append(0.0)
                idf1_scores.append(0.0)

        # Plot MOTA
        mota_fig = plt.figure(figsize=(12, 8))
        plt.plot(frame_numbers, mota_scores)
        plt.xlabel('Frame Number')
        plt.ylabel('MOTA Score')
        plt.title('MOTA over Time')
        plt.grid(True)
        save_plot(mota_fig, "mota_curve")

        # Plot IDF1
        idf1_fig = plt.figure(figsize=(12, 8))
        plt.plot(frame_numbers, idf1_scores)
        plt.xlabel('Frame Number')
        plt.ylabel('IDF1 Score')
        plt.title('IDF1 over Time')
        plt.grid(True)
        save_plot(idf1_fig, "idf1_curve")

        # 6. ID Switches over time
        switch_fig = plt.figure(figsize=(12, 8))
        switches = calculate_id_switches(tracking_results, frame_count)
        plt.plot(range(1, len(switches) + 1), switches)
        plt.xlabel('Frame Number')
        plt.ylabel('Cumulative ID Switches')
        plt.title('ID Switches over Time')
        plt.grid(True)
        save_plot(switch_fig, "id_switches")

        # 7. Track count over time
        count_fig = plt.figure(figsize=(12, 8))
        track_counts = [len([p for p in tracking_results if p["frame"] == frame]) 
                       for frame in range(1, frame_count + 1)]
        plt.plot(range(1, frame_count + 1), track_counts)
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Tracks')
        plt.title('Track Count over Time')
        plt.grid(True)
        save_plot(count_fig, "track_count")

    except Exception as e:
        logging.error(f"Error generating MOT metrics plots: {str(e)}")

    # Cleanup
    fo.delete_dataset(dataset_name)
    return results

def normalize_bbox(bbox, sample):
    """Convert absolute bbox coordinates to relative coordinates"""
    try:
        # Get frame dimensions from the image file
        frame = cv2.imread(sample.filepath)
        if frame is None:
            raise ValueError(f"Could not read image: {sample.filepath}")
            
        height, width = frame.shape[:2]
        
        x1, y1, x2, y2 = bbox
        return [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height
        ]
    except Exception as e:
        logging.error(f"Error normalizing bbox: {str(e)}")
        # Return original coordinates if normalization fails
        return [x1, y1, x2 - x1, y2 - y1]

def calculate_iou_curve(tracking_results, ground_truth, thresholds):
    """Calculate detection rate at different IoU thresholds"""
    detection_rates = []
    
    for threshold in thresholds:
        total_detections = 0
        correct_detections = 0
        
        # Group by frame
        for frame_num in set(p["frame"] for p in tracking_results):
            frame_preds = [p for p in tracking_results if p["frame"] == frame_num]
            frame_gt = [g for g in ground_truth if g["frame"] == frame_num]
            
            # For each prediction, check if it matches any ground truth with sufficient IoU
            for pred in frame_preds:
                total_detections += 1
                for gt in frame_gt:
                    if calculate_box_iou(pred["bbox"], gt["bbox"]) > threshold:
                        correct_detections += 1
        
        detection_rate = correct_detections / total_detections if total_detections > 0 else 0
        detection_rates.append(detection_rate)
    
    return detection_rates

def calculate_id_switches(tracking_results, frame_count):
    """Calculate cumulative ID switches over time"""
    switches = []
    cumulative_switches = 0
    
    # For each frame, compare IDs with previous frame
    for frame in range(1, frame_count):
        prev_frame_tracks = {p["id"]: p["bbox"] for p in tracking_results if p["frame"] == frame}
        curr_frame_tracks = {p["id"]: p["bbox"] for p in tracking_results if p["frame"] == frame + 1}
        
        # Check for ID switches
        for track_id, curr_bbox in curr_frame_tracks.items():
            if track_id in prev_frame_tracks:
                # If same ID but position changed significantly, might be an ID switch
                prev_bbox = prev_frame_tracks[track_id]
                if calculate_box_iou(prev_bbox, curr_bbox) < 0.5:
                    cumulative_switches += 1
        
        # Add disappeared and new tracks
        disappeared = set(prev_frame_tracks.keys()) - set(curr_frame_tracks.keys())
        appeared = set(curr_frame_tracks.keys()) - set(prev_frame_tracks.keys())
        cumulative_switches += len(disappeared) + len(appeared)
        
        switches.append(cumulative_switches)
    
    return switches

def calculate_box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate areas
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou 

def calculate_mot_metrics(acc):
    """Calculate MOT metrics using motmetrics"""
    mh = mm.metrics.create()
    
    metrics_list = [
        'mota', 'motp', 'idf1',
        'num_switches', 'num_fragmentations',
        'num_false_positives', 'num_misses'
    ]
    
    try:
        # Compute metrics (np.asfarray is already patched)
        summary = mh.compute(acc, metrics=metrics_list, name='acc')
        
        # Convert any remaining problematic arrays
        for col in summary.columns:
            if summary[col].dtype != float:
                summary[col] = np.asarray(summary[col], dtype=float)
        return summary
    except Exception as e:
        logging.error(f"Error in MOT metrics calculation: {str(e)}")
        # Return empty DataFrame with same structure
        import pandas as pd
        return pd.DataFrame(columns=metrics_list, index=['acc'], 
                          data=np.zeros((1, len(metrics_list)), dtype=float))

# At the end of the file, restore original if needed
if original_asfarray is not None:
    np.asfarray = original_asfarray 