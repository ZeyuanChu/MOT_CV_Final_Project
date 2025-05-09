# How to run the code:
- Run the code with the following command to have a square bounding box:
    - python3 final_code.py --input_file <filename> --box_type square --output_file <output_filename> --conf_thresh <0.0-1.0> --iou_thresh <0.0-1.0>
    - python3 final_code.py --input_file input_files/traffic.mp4 --box_type square --output_file traffic_square --conf_thresh 0.3 --iou_thresh 0.3
    - python3 final_code_colab.py --input_file input_files/traffic.mp4 --box_type square --output_file traffic_square --conf_thresh 0.3 --iou_thresh 0.3

- Run the code with the following command to have a segmentation overlay:
    - python3 final_code.py --input_file <filename> --box_type overlay --output_file <output_filename> --conf_thresh <0.0-1.0> --iou_thresh <0.0-1.0>
    - python3 final_code.py --input_file input_files/traffic.mp4 --box_type overlay --output_file traffic_overlay --conf_thresh 0.3 --iou_thresh 0.3
    - python3 final_code_colab.py --input_file input_files/traffic.mp4 --box_type overlay --output_file traffic_overlay --conf_thresh 0.3 --iou_thresh 0.3

# How to run the code in Google Colab:
- Go to the Colab implementation at https://colab.research.google.com/drive/1UgU_yBro7pnM11UKRJfyCd6eiIgn4_sw?usp=sharing
- Run through the script with any input.mp4 file.
- Similar logic just without cv2 functionality.

# Command line arguments:
- input_file: Path to input video file (default: input_files/traffic.mp4)
- box_type: Type of bounding box: square or segmentation overlay (default: square)
- output_file: Custom output filename (without extension) (default: result)
- conf_thresh: Confidence threshold for YOLO detections (0-1) (default: 0.3)
- iou_thresh: IOU threshold for tracking (0-1) (default: 0.3)

# Things to do:
- Provide more videos to test the code. Try to find longer videos with different types of objects and environments to track.
- Possible improve the color detection as it seems a little bit off.
- Try to figure out if we can run the code using Google Colab or something to speed up the process and make it real time, however this might not be possible due to the use of cv2 modules.
- Commenting/fixing the code structure and implementation. We will both make it look less ChatGPT like.

# Extra Notes/Things we can do:
- We should not focus on training the model, and more on utilizing the pretrained YOLO models.
    - YOLO11 is being used for square bounding boxes.
    - YOLO11-seg is being used for the overlay bounding boxes.
- We should not need the COCO dataset, as it is not needed since ultralytics has the pretrained models.

# Usefule Links:
- https://docs.ultralytics.com/models/yolo11/
- https://cocodataset.org/#download

## Evaluation
The system automatically performs comprehensive evaluation after processing each video, including:
- ROC curves per object class
    - ROC curves are a plot of the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
- Precision-Recall curves
    - Precision-Recall curves are a plot of the precision (P) against the recall (R) at various threshold settings.
- IoU performance curve
    - IoU performance curve is a plot of the IoU (Intersection over Union) against the threshold settings.
- MOTA (Multiple Object Tracking Accuracy)
    - MOTA is a metric that measures the accuracy of the tracking system.
- IDF1 Score tracking
    - IDF1 Score is a metric that measures the accuracy of the tracking system.
- ID switches analysis
    - ID switches analysis is a metric that measures the number of times the ID of an object changes.
- Track count statistics
    - Track count statistics is a metric that measures the number of objects being tracked.

Results are saved in output_files/<video_name>/evaluation_results/

# Threshold explanations:
- conf_thresh: Confidence threshold (0.0-1.0)
  - Lower values (0.1-0.3): Detect more objects, including distant/unclear ones, but more false positives
  - Default (0.25): Balanced detection
  - Higher values (0.4-0.7): Only detect clear/confident objects, fewer false positives

- iou_thresh: Intersection over Union threshold (0.0-1.0)
  - Lower values (0.1-0.2): Easier to track objects between frames, good for fast/distant objects
  - Default (0.3): Balanced tracking
  - Higher values (0.4-0.6): Stricter tracking, better for clear/close objects
