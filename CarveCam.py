import cv2
import sys
import numpy as np
import os
import threading
import time

# --- AI Detection/Tracking Import ---
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: 'ultralytics' library not found.")
    print("Please install it: pip install ultralytics")
    sys.exit(1)


# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

# --- Helper Functions ---

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    # Return the intersection over union value
    return iou


def select_target_by_roi(frame, status_callback=None):
    """Allows the user to manually select the target Region of Interest (ROI)."""
    if status_callback: status_callback("Select target skier...")

    temp_root = tk.Tk()
    temp_root.withdraw()
    messagebox.showinfo("Select Target",
                        "An OpenCV window will open.\n"
                        "Draw a box around the TARGET skier and press ENTER or SPACE.\n"
                        "This box will be used to identify the skier among AI detections.\n"
                        "Press 'c' to cancel selection.",
                        parent=temp_root)
    temp_root.destroy()

    window_name = "Select TARGET Skier - Press ENTER/SPACE to Confirm, C to Cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if roi == (0, 0, 0, 0):
        print("ERROR: ROI selection cancelled or invalid.")
        messagebox.showerror("Error", "Target selection cancelled or invalid.")
        return None
    print(f"Target ROI selected: {roi}")
    # Convert ROI from (x, y, w, h) to (x1, y1, x2, y2)
    roi_xyxy = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    return roi_xyxy

# --- center_crop_frame function remains the same ---
def center_crop_frame(frame, center_x, center_y, out_width, out_height):
    """Crops the frame around the center point, handling boundaries."""
    frame_h, frame_w = frame.shape[:2]
    crop_x1 = int(center_x - out_width / 2)
    crop_y1 = int(center_y - out_height / 2)
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, (crop_x1 + out_width) - frame_w)
    pad_bottom = max(0, (crop_y1 + out_height) - frame_h)
    crop_x1_adj = max(0, crop_x1)
    crop_y1_adj = max(0, crop_y1)
    crop_x2_adj = min(frame_w, crop_x1 + out_width)
    crop_y2_adj = min(frame_h, crop_y1 + out_height)

    cropped_region = frame[crop_y1_adj:crop_y2_adj, crop_x1_adj:crop_x2_adj]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_frame = cv2.copyMakeBorder(cropped_region, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        current_h, current_w = padded_frame.shape[:2]
        if current_h != out_height or current_w != out_width:
             padded_frame = cv2.resize(padded_frame, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
        return padded_frame
    else:
        current_h, current_w = cropped_region.shape[:2]
        if current_h != out_height or current_w != out_width:
            return cv2.resize(cropped_region, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        else:
            return cropped_region

# --- Core Processing Logic (Using YOLO Detection + Simple Tracking) ---
def run_detection_tracking_processing(input_video_path, output_video_path,
                                      yolo_model_name, # e.g., 'yolov8n.pt'
                                      out_width, out_height, no_display,
                                      start_time_sec, end_time_sec,
                                      confidence_threshold, # New: Detection confidence
                                      status_callback=None, progress_callback=None):
    """
    Performs object detection (YOLO) and simple tracking for centering.
    Args:
        yolo_model_name (str): Name of the YOLO model file (e.g., 'yolov8n.pt').
        confidence_threshold (float): Minimum confidence score for person detection.
        ... other args ...
    """
    if status_callback: status_callback("Initializing AI model...")
    print(f"Loading YOLO model: {yolo_model_name}")
    try:
        # Load the YOLO model (will download if not present)
        model = YOLO(yolo_model_name)
        # You might want to specify device='cuda' if you have a GPU and PyTorch setup
        # model = YOLO(yolo_model_name).to('cuda')
        print("YOLO model loaded successfully.")
    except Exception as e:
        errmsg = f"Error loading YOLO model '{yolo_model_name}': {e}\nCheck model name and internet connection."
        print(errmsg); messagebox.showerror("AI Model Error", errmsg)
        if status_callback: status_callback("Error: Failed to load AI model.")
        return

    # --- Video File Handling (same as before) ---
    if not os.path.exists(input_video_path):
        errmsg = f"Error: Input video not found at '{input_video_path}'"
        print(errmsg); messagebox.showerror("File Error", errmsg)
        if status_callback: status_callback("Error: Input file not found.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        errmsg = f"Error: Could not open video file: {input_video_path}"
        print(errmsg); messagebox.showerror("Video Error", errmsg)
        if status_callback: status_callback("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30; print("Warning: FPS read as 0, assuming 30.")
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_original_frames / fps if total_original_frames > 0 and fps > 0 else 0
    frame_height_orig, frame_width_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Input Video Info: {frame_width_orig}x{frame_height_orig} @ {fps:.2f} FPS, {total_original_frames} frames (~{duration_sec:.2f}s)")

    # --- Trimming Setup (same as before) ---
    start_msec = start_time_sec * 1000
    end_msec = end_time_sec * 1000 if end_time_sec > 0 else -1
    # (Validation for trim range is the same)
    if start_msec < 0: start_msec = 0
    if duration_sec > 0 and start_time_sec >= duration_sec: errmsg = "..."; print(errmsg); messagebox.showerror("Trim Error", errmsg); cap.release(); return # Simplified
    if end_msec > 0 and start_msec >= end_msec: errmsg = "..."; print(errmsg); messagebox.showerror("Trim Error", errmsg); cap.release(); return # Simplified

    start_frame_approx = int(start_time_sec * fps)
    end_frame_approx = int(end_time_sec * fps) if end_time_sec > 0 and end_time_sec <= duration_sec else total_original_frames
    frames_to_process = max(1, end_frame_approx - start_frame_approx)
    print(f"Processing from {start_time_sec:.2f}s to {end_time_sec if end_time_sec > 0 else duration_sec:.2f}s (approx frames {start_frame_approx} to {end_frame_approx})")
    print(f"Output Resolution: {out_width}x{out_height}")
    print(f"Detection Confidence Threshold: {confidence_threshold:.2f}")

    # --- Seek to Start Frame ---
    if start_msec > 0:
        if status_callback: status_callback(f"Seeking to {start_time_sec:.1f}s...")
        seek_ok = cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
        time.sleep(0.1) # Give it a moment
        ok, frame = cap.read()
        current_msec_after_seek = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"Seek attempt finished. Current position: {current_msec_after_seek/1000:.2f}s")
        if not seek_ok or abs(current_msec_after_seek - start_msec) > 1500: # Wider tolerance
             print("Warning: Seeking may be inaccurate.")
             messagebox.showwarning("Seek Warning", "Seeking might be inaccurate for this video.")
    else:
        ok, frame = cap.read()

    if not ok:
        errmsg = f"Error: Cannot read video frame at start time ({start_time_sec:.2f}s)."
        print(errmsg); messagebox.showerror("Video Error", errmsg)
        if status_callback: status_callback("Error: Cannot read start frame.")
        cap.release(); return

    frame_height, frame_width = frame.shape[:2] # Use actual dimensions of read frame

    # --- Resolution Warning ---
    if out_width > frame_width or out_height > frame_height:
         print("Warning: Output resolution > input. Black borders may appear.")
         # Maybe show warning box again if dims changed after seek

    # --- Initial Target Identification using ROI ---
    target_roi_xyxy = select_target_by_roi(frame, status_callback)
    if target_roi_xyxy is None:
        if status_callback: status_callback("Target selection cancelled.")
        cap.release(); return

    target_bbox = None # Will store the (x1, y1, x2, y2) of the tracked skier
    last_known_center = None
    initial_target_found = False

    if status_callback: status_callback("Detecting target in first frame...")
    try:
        results = model.predict(frame, classes=0, conf=confidence_threshold, verbose=False) # classes=0 for 'person'
        best_iou = -1

        if results and results[0].boxes: # Check if results exist and have boxes
            for box in results[0].boxes:
                if int(box.cls.item()) == 0: # Ensure it's a person
                    det_xyxy = box.xyxy[0].cpu().numpy().astype(int) # Get box coords
                    iou = calculate_iou(target_roi_xyxy, det_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        target_bbox = det_xyxy # Found potential target
                        initial_target_found = True

        if not initial_target_found or best_iou < 0.1: # Require some minimal overlap
            print("Error: Could not associate selected ROI with a detected person.")
            messagebox.showerror("Target Error", "Could not find a detected person matching the selected ROI.\nTry drawing the box more accurately or adjusting confidence threshold.")
            if status_callback: status_callback("Error: Target ID failed.")
            cap.release(); return
        else:
            print(f"Initial target identified with IoU: {best_iou:.2f}")
            cx = (target_bbox[0] + target_bbox[2]) / 2
            cy = (target_bbox[1] + target_bbox[3]) / 2
            last_known_center = (cx, cy)

    except Exception as e:
        errmsg = f"Error during initial detection: {e}"
        print(errmsg); messagebox.showerror("Detection Error", errmsg)
        if status_callback: status_callback("Error: Initial detection failed.")
        cap.release(); return


    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
         errmsg = f"Error: Could not open video writer for '{output_video_path}'"
         print(errmsg); messagebox.showerror("File Error", errmsg)
         if status_callback: status_callback("Error: Cannot create output file.")
         cap.release(); return

    print(f"\nProcessing video segment... Outputting to '{output_video_path}'")
    if not no_display: print("Press 'q' or ESC in the display window to stop early.")
    if status_callback: status_callback("Processing...")

    # --- Frame-by-Frame Detection & Tracking Loop ---
    frame_count_in_range = 0
    processed_count = 0
    stop_processing = False
    target_currently_tracked = True # Assume tracked initially
    first_iteration = True

    while not stop_processing:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if end_msec > 0 and current_msec > end_msec: break # Check End Time

        # Read next frame (skip on first iteration)
        if not first_iteration:
            ok, frame = cap.read()
            if not ok: break # End of video
        else:
            first_iteration = False # Use the frame already read for first loop

        frame_count_in_range += 1
        timer = cv2.getTickCount() # Timer for detection FPS calculation

        target_found_in_frame = False
        current_detections = []

        # --- Run YOLO Detection ---
        try:
            results = model.predict(frame, classes=0, conf=confidence_threshold, verbose=False)
            if results and results[0].boxes:
                 for box in results[0].boxes:
                     if int(box.cls.item()) == 0: # Person class
                         det_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                         conf = float(box.conf.item())
                         current_detections.append({'bbox': det_xyxy, 'conf': conf})
        except Exception as e:
            print(f"\nError during prediction on frame {current_frame_num}: {e}")
            # Decide how to handle: skip frame, stop? For now, treat as lost track.
            target_currently_tracked = False


        # --- Simple Tracking Logic (Nearest Neighbor by Center) ---
        if current_detections and last_known_center is not None:
            min_dist = float('inf')
            best_match_box = None

            for det in current_detections:
                box = det['bbox']
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                dist_sq = (cx - last_known_center[0])**2 + (cy - last_known_center[1])**2

                # Simple distance threshold (relative to frame size?) - needs tuning!
                # Example: Don't jump more than half the frame width/height instantly
                max_allowed_dist_sq = (frame_width * 0.3)**2 + (frame_height*0.3)**2
                if dist_sq < min_dist and dist_sq < max_allowed_dist_sq:
                     min_dist = dist_sq
                     best_match_box = box
                     target_found_in_frame = True

            if target_found_in_frame:
                target_bbox = best_match_box
                # Update last known center
                cx = (target_bbox[0] + target_bbox[2]) / 2
                cy = (target_bbox[1] + target_bbox[3]) / 2
                last_known_center = (cx, cy)
                target_currently_tracked = True
            else:
                 # No detection close enough to the last known position
                 target_currently_tracked = False
                 print(f"Tracking lost on frame ~{current_frame_num} - no close detection found.")
                 # Keep using the 'last_known_center' for cropping for now
        else:
             # No persons detected at all, or last_known_center is missing
             target_currently_tracked = False
             print(f"Tracking lost on frame ~{current_frame_num} - no persons detected.")
             # Keep using 'last_known_center' if available

        detect_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


        # --- Cropping & Writing Frame ---
        if last_known_center: # Always try to crop around the last known good position
            (center_x, center_y) = last_known_center
            cropped_frame = center_crop_frame(frame, center_x, center_y, out_width, out_height)
            writer.write(cropped_frame)
            processed_count += 1
        else:
             # Should not happen after initial ID, but as fallback write black
             print(f"Error: last_known_center is None on frame {current_frame_num}. Writing black frame.")
             black_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
             writer.write(black_frame)


        # --- Display (Optional) ---
        if not no_display:
            display_frame = frame.copy() # Draw on a copy
            # Draw all detections
            for det in current_detections:
                b = det['bbox']
                c = det['conf']
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1) # Yellow for all detections
                cv2.putText(display_frame, f'{c:.2f}', (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Highlight tracked target
            if target_currently_tracked and target_bbox is not None:
                b = target_bbox
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2) # Green for tracked target
                cv2.putText(display_frame, "TARGET", (b[0], b[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif last_known_center: # Mark last known spot if tracking lost
                 cv2.circle(display_frame, (int(last_known_center[0]), int(last_known_center[1])), 10, (0,0,255), -1)
                 cv2.putText(display_frame, "LOST", (int(last_known_center[0])+15, int(last_known_center[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add stats
            time_str = f"Time: {current_msec/1000:.2f}s"
            cv2.putText(display_frame, time_str, (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Detect FPS: {int(detect_fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
            frame_text = f"Frame: {current_frame_num}"
            cv2.putText(display_frame, frame_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

            try:
                display_orig_resized = cv2.resize(display_frame, (out_width // 2, out_height // 2))
                display_crop_resized = cv2.resize(cropped_frame, (out_width // 2, out_height // 2))
                combined_display = np.hstack((display_orig_resized, display_crop_resized))
                cv2.imshow("Detection (Left) vs Cropped (Right) - Press 'q'/ESC to stop", combined_display)
            except Exception as display_e:
                print(f"Warning: Error displaying frames: {display_e}")
                cv2.imshow("Cropped Output - Press 'q'/ESC to stop", cropped_frame) # Fallback

        # --- Update GUI Progress (same as before) ---
        if progress_callback:
            processed_in_intended_range = max(0, current_frame_num - start_frame_approx)
            progress_percent = int(100 * processed_in_intended_range / frames_to_process) if frames_to_process > 0 else 0
            progress_callback(min(100, progress_percent))

        # --- Check for Quit Key (same as before) ---
        if not no_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                 stop_reason = "'q'" if key == ord('q') else "ESC"
                 print(f"\n{stop_reason} pressed, stopping early.")
                 if status_callback: status_callback("Stopping...")
                 stop_processing = True

    # --- Cleanup ---
    print(f"\nProcessed {processed_count} frames successfully within the specified range.")
    print("Releasing resources...")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    # Clean up potential lingering YOLO resources if necessary (depends on library version)
    # try:
    #    torch.cuda.empty_cache() # If using CUDA
    # except NameError: pass # If torch not imported or used

    if not stop_processing:
        final_message = f"Processing complete. Output saved to:\n{output_video_path}"
        print(final_message)
        if status_callback: status_callback("Done!")
        # Messagebox shown by main thread check
    else:
         if status_callback: status_callback("Stopped.")

    if progress_callback: progress_callback(0)


# --- Tkinter GUI Application Class (Updated for YOLO) ---
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ski Tracker & Centering Tool (AI Version)")

        # GUI Variables
        self.input_video_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        # Removed: self.tracker_type
        self.yolo_model_name = tk.StringVar(value='yolov8n.pt') # Default to nano model
        self.confidence_threshold = tk.StringVar(value='0.40') # Default confidence
        self.output_width = tk.StringVar(value='1280')
        self.output_height = tk.StringVar(value='720')
        self.start_time_str = tk.StringVar(value='0')
        self.end_time_str = tk.StringVar(value='0')
        self.no_display = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready (Using YOLO AI Detection)")
        self.processing_active = False
        self.process_thread = None

        # --- GUI Layout ---
        mainframe = ttk.Frame(root, padding="10 10 10 10")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        current_row = 0
        # Input/Output (same as before)
        ttk.Label(mainframe, text="Input Video:").grid(column=0, row=current_row, sticky=tk.W, pady=2)
        ttk.Entry(mainframe, width=50, textvariable=self.input_video_path, state='readonly').grid(column=1, row=current_row, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(mainframe, text="Browse...", command=self.browse_input).grid(column=3, row=current_row, sticky=tk.W)
        current_row += 1
        ttk.Label(mainframe, text="Output Video:").grid(column=0, row=current_row, sticky=tk.W, pady=2)
        ttk.Entry(mainframe, width=50, textvariable=self.output_video_path, state='readonly').grid(column=1, row=current_row, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(mainframe, text="Save As...", command=self.browse_output).grid(column=3, row=current_row, sticky=tk.W)
        current_row += 1
        ttk.Separator(mainframe, orient=tk.HORIZONTAL).grid(column=0, row=current_row, columnspan=4, sticky='ew', pady=8)
        current_row += 1

        # --- Settings Frame ---
        settings_frame = ttk.Frame(mainframe)
        settings_frame.grid(column=0, row=current_row, columnspan=4, sticky=(tk.W, tk.E))
        settings_frame.columnconfigure(1, weight=1)

        # YOLO Model Selection (Dropdown or Entry)
        ttk.Label(settings_frame, text="AI Model:").grid(column=0, row=0, sticky=tk.W, padx=(0,5), pady=2)
        # Option 1: Simple Entry (user needs to know model names)
        # model_entry = ttk.Entry(settings_frame, width=15, textvariable=self.yolo_model_name)
        # model_entry.grid(column=1, row=0, sticky=tk.W)
        # Option 2: Combobox with common options
        model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'] # Add more if desired
        model_combo = ttk.Combobox(settings_frame, width=12, textvariable=self.yolo_model_name, values=model_options)
        model_combo.grid(column=1, row=0, sticky=tk.W)
        ttk.Label(settings_frame, text="(n=fastest, x=most accurate)").grid(column=2, row=0, columnspan=4, sticky=tk.W, padx=5)


        # Confidence Threshold
        ttk.Label(settings_frame, text="Confidence:").grid(column=0, row=1, sticky=tk.W, padx=(0,5), pady=5)
        conf_entry = ttk.Entry(settings_frame, width=6, textvariable=self.confidence_threshold)
        conf_entry.grid(column=1, row=1, sticky=tk.W)
        # Optionally add a Scale widget for confidence later

        # Output Size (reposition)
        ttk.Label(settings_frame, text="Output WxH:").grid(column=2, row=1, sticky=tk.W, padx=(15, 5), pady=5)
        width_entry = ttk.Entry(settings_frame, width=6, textvariable=self.output_width)
        width_entry.grid(column=3, row=1, sticky=tk.W)
        ttk.Label(settings_frame, text="x").grid(column=4, row=1, sticky=tk.W, padx=2)
        height_entry = ttk.Entry(settings_frame, width=6, textvariable=self.output_height)
        height_entry.grid(column=5, row=1, sticky=tk.W)


        # Trim Times (same as before)
        ttk.Label(settings_frame, text="Start Time (s):").grid(column=0, row=2, sticky=tk.W, padx=(0,5), pady=5)
        start_time_entry = ttk.Entry(settings_frame, width=7, textvariable=self.start_time_str)
        start_time_entry.grid(column=1, row=2, sticky=tk.W)
        ttk.Label(settings_frame, text="End Time (s):").grid(column=2, row=2, sticky=tk.W, padx=(15, 5), pady=5)
        end_time_entry = ttk.Entry(settings_frame, width=7, textvariable=self.end_time_str)
        end_time_entry.grid(column=3, row=2, sticky=tk.W)
        ttk.Label(settings_frame, text="(0 = to end)").grid(column=4, row=2, columnspan=2, sticky=tk.W, padx=5)
        current_row += 1 # Increment after the settings frame block

        # No Display Checkbox (same as before)
        display_check = ttk.Checkbutton(mainframe, text="Hide Preview Windows (Faster Processing)",
                                        variable=self.no_display, onvalue=True, offvalue=False)
        display_check.grid(column=0, row=current_row, columnspan=4, sticky=tk.W, pady=5)
        current_row += 1

        # --- Action Button, Progress Bar, Status Label (same as before) ---
        action_frame = ttk.Frame(mainframe)
        action_frame.grid(column=0, row=current_row, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        action_frame.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.grid(column=1, row=0, padx=10)
        self.progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(column=0, row=0, sticky=(tk.W, tk.E))
        current_row += 1
        status_label = ttk.Label(mainframe, textvariable=self.status_text, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(column=0, row=current_row, columnspan=4, sticky=(tk.W, tk.E), pady=5, ipady=2)

    # --- browse_input, browse_output, update_status, update_progress methods remain the same ---
    def browse_input(self):
        filetypes = (("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Input Video", filetypes=filetypes)
        if filepath:
            self.input_video_path.set(filepath)
            if not self.output_video_path.get():
                base, ext = os.path.splitext(filepath)
                suggested_output = f"{base}_centered_ai{ext}" # Suggest different name
                self.output_video_path.set(suggested_output)
            self.status_text.set("Input selected. Specify output and AI settings.")

    def browse_output(self):
        filetypes = (("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        initial_file = ""
        in_path = self.input_video_path.get()
        if in_path:
             base, ext = os.path.splitext(os.path.basename(in_path))
             initial_file = f"{base}_centered_ai.mp4" # Suggest different name

        filepath = filedialog.asksaveasfilename(title="Save Output Video As",
                                                 filetypes=filetypes,
                                                 defaultextension=".mp4",
                                                 initialfile=initial_file)
        if filepath:
            self.output_video_path.set(filepath)
            self.status_text.set("Output path selected.")

    def update_status(self, message):
        self.root.after(0, self.status_text.set, message)

    def update_progress(self, value):
         self.root.after(0, self.progress_bar.config, {'value': max(0, min(100, value))})


    def start_processing_thread(self):
        if self.processing_active:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # --- Input Validation ---
        input_path = self.input_video_path.get()
        output_path = self.output_video_path.get()
        model_name = self.yolo_model_name.get() # Get model name

        # Validate Paths/Model
        if not input_path: messagebox.showerror("Invalid Input", "Select input video."); return
        if not output_path: messagebox.showerror("Invalid Input", "Specify output video path."); return
        if input_path == output_path: messagebox.showerror("Invalid Input", "Input/output paths cannot be same."); return
        if not model_name: messagebox.showerror("Invalid Input", "Specify or select an AI model name (e.g., yolov8n.pt)."); return

        # Validate Dimensions
        try:
            width = int(self.output_width.get())
            height = int(self.output_height.get())
            if width <= 0 or height <= 0: raise ValueError("Dimensions must be positive")
        except ValueError: messagebox.showerror("Invalid Input", "Output width/height must be positive integers."); return

        # Validate Confidence
        try:
            conf_thresh = float(self.confidence_threshold.get())
            if not (0.0 <= conf_thresh <= 1.0):
                raise ValueError("Confidence must be between 0.0 and 1.0")
        except ValueError as e: messagebox.showerror("Invalid Input", f"Invalid confidence threshold: {e}"); return


        # Validate Trim Times (same as before)
        try:
            start_time = float(self.start_time_str.get() or 0.0)
            end_time = float(self.end_time_str.get() or 0.0)
            if start_time < 0: raise ValueError("Start time cannot be negative")
            if end_time < 0: raise ValueError("End time cannot be negative (use 0 for end)")
            if end_time != 0 and start_time >= end_time: raise ValueError("End time > start time")
        except ValueError as e: messagebox.showerror("Invalid Input", f"Invalid start/end time: {e}"); return


        # --- Start Processing ---
        self.processing_active = True
        self.start_button.config(state=tk.DISABLED)
        self.update_status("Starting AI processing...")
        self.update_progress(0)

        # Use the new processing function
        self.process_thread = threading.Thread(
            target=run_detection_tracking_processing,
            args=(input_path, output_path, model_name, width, height,
                  self.no_display.get(),
                  start_time, end_time, conf_thresh, # Pass model name & confidence
                  self.update_status, self.update_progress),
            daemon=True
        )

        # --- Thread monitoring logic (same as before) ---
        def on_thread_complete():
            self.processing_active = False
            self.start_button.config(state=tk.NORMAL)
            final_status = self.status_text.get()
            if final_status == "Done!":
                 messagebox.showinfo("Success", f"Processing complete. Output saved to:\n{output_path}")

        def check_thread():
            if self.process_thread is not None and self.process_thread.is_alive():
                self.root.after(100, check_thread)
            else:
                if self.processing_active:
                    on_thread_complete()

        self.process_thread.start()
        self.root.after(100, check_thread)


# --- Main execution ---
if __name__ == "__main__":
    # No need to check for cv2.TrackerCSRT_create anymore
    print("AI Skier Tracking Tool")
    print("Ensure 'ultralytics' is installed: pip install ultralytics")
    # Optional: Add check for GPU availability here if desired
    # import torch
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA available: {torch.cuda.is_available()}")

    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()