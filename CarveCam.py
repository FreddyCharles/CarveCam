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
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    iou = interArea / denom if denom > 0 else 0
    return iou

def select_target_by_roi(frame, status_callback=None):
    """Allows the user to manually select the target Region of Interest (ROI)."""
    if status_callback: status_callback("Select target skier...")
    temp_root = tk.Tk(); temp_root.withdraw()
    messagebox.showinfo("Select Target", "...", parent=temp_root) # Simplified message
    temp_root.destroy()
    window_name = "Select TARGET Skier - Press ENTER/SPACE to Confirm, C to Cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if roi == (0, 0, 0, 0):
        print("ERROR: ROI selection cancelled."); messagebox.showerror("Error", "Target selection cancelled."); return None
    print(f"Target ROI selected: {roi}")
    roi_xyxy = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    return roi_xyxy

# --- NEW Cropping function with optional blur ---
def center_crop_frame_v2(original_frame, smoothed_center_x, smoothed_center_y,
                       out_width, out_height, use_blur_padding=True, blur_ksize=99):
    """
    Crops the frame around the smoothed center point.
    Handles boundaries by either padding with black or a blurred background.
    """
    frame_h, frame_w = original_frame.shape[:2]

    # Calculate desired top-left corner based on SMOOTHED center
    crop_x1 = int(smoothed_center_x - out_width / 2)
    crop_y1 = int(smoothed_center_y - out_height / 2)
    crop_x2 = crop_x1 + out_width
    crop_y2 = crop_y1 + out_height

    # --- Calculate padding needed (how much the crop goes outside the frame) ---
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - frame_w)
    pad_bottom = max(0, crop_y2 - frame_h)

    # --- Get the coordinates of the valid region WITHIN the original frame ---
    crop_x1_adj = max(0, crop_x1)
    crop_y1_adj = max(0, crop_y1)
    crop_x2_adj = min(frame_w, crop_x2)
    crop_y2_adj = min(frame_h, crop_y2)

    # Crop the valid region from the original frame
    valid_cropped_region = original_frame[crop_y1_adj:crop_y2_adj, crop_x1_adj:crop_x2_adj]

    # Check if any padding is needed
    padding_needed = pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0

    if not padding_needed:
        # No padding needed, just resize the valid crop to the output size
        # Use INTER_LINEAR for resizing images usually looks better than INTER_NEAREST
        return cv2.resize(valid_cropped_region, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    else:
        # Padding is required
        if use_blur_padding:
            # --- Blurred Background Padding ---
            # 1. Blur the original frame significantly
            # Ensure ksize is odd
            ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
            blurred_frame = cv2.GaussianBlur(original_frame, (ksize, ksize), 0)

            # 2. Resize the blurred frame to the target output size
            background_canvas = cv2.resize(blurred_frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

            # 3. Calculate where the valid_cropped_region should be pasted onto the background
            # The position corresponds to where the valid crop was *within* the desired output view
            paste_x1 = pad_left
            paste_y1 = pad_top
            paste_x2 = out_width - pad_right
            paste_y2 = out_height - pad_bottom

            # 4. Ensure the dimensions match before pasting (due to potential rounding)
            valid_crop_h, valid_crop_w = valid_cropped_region.shape[:2]
            paste_h = paste_y2 - paste_y1
            paste_w = paste_x2 - paste_x1

            if valid_crop_h != paste_h or valid_crop_w != paste_w:
                 # If dimensions don't match perfectly, resize the valid crop slightly
                 # This might happen due to int() rounding differences
                 print(f"Warning: Resizing valid crop slightly before pasting ({valid_crop_w}x{valid_crop_h} vs {paste_w}x{paste_h})")
                 try:
                     valid_cropped_region = cv2.resize(valid_cropped_region, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR)
                 except cv2.error as e:
                     print(f"Error resizing valid crop: {e}. Pasting original size.")
                     # Attempt to paste original size if resize fails, might cause error
                     paste_x2 = paste_x1 + valid_crop_w
                     paste_y2 = paste_y1 + valid_crop_h
                     # Ensure paste coords are within bounds
                     paste_x2 = min(paste_x2, out_width)
                     paste_y2 = min(paste_y2, out_height)
                     # Recrop if necessary
                     valid_cropped_region = valid_cropped_region[:paste_y2-paste_y1, :paste_x2-paste_x1]


            # 5. Paste the valid region onto the blurred canvas
            try:
                background_canvas[paste_y1:paste_y2, paste_x1:paste_x2] = valid_cropped_region
                return background_canvas
            except ValueError as e:
                print(f"Error pasting cropped region onto background: {e}")
                print(f"Canvas shape: {background_canvas.shape}, Region shape: {valid_cropped_region.shape}")
                print(f"Paste Coords: y={paste_y1}:{paste_y2}, x={paste_x1}:{paste_x2}")
                # Fallback to black padding if paste fails
                return cv2.copyMakeBorder(valid_cropped_region, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])


        else:
            # --- Black Padding ---
            # Use INTER_NEAREST for padding borders usually
            padded_frame = cv2.copyMakeBorder(valid_cropped_region, pad_top, pad_bottom, pad_left, pad_right,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # Ensure final size is correct after padding
            current_h, current_w = padded_frame.shape[:2]
            if current_h != out_height or current_w != out_width:
                 padded_frame = cv2.resize(padded_frame, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
            return padded_frame

# --- Core Processing Logic (Updated with Smoothing) ---
def run_detection_tracking_processing(input_video_path, output_video_path,
                                      yolo_model_name,
                                      out_width, out_height, no_display,
                                      start_time_sec, end_time_sec,
                                      confidence_threshold,
                                      smoothing_factor, # NEW: Alpha for EMA
                                      use_blur_padding, # NEW: Blur flag
                                      blur_amount,      # NEW: Blur kernel size
                                      status_callback=None, progress_callback=None):
    """
    Performs object detection, EMA smoothing, and centering with optional blur padding.
    Args:
        smoothing_factor (float): EMA alpha (0 to 1). Lower = more smoothing.
        use_blur_padding (bool): Whether to use blurred background for padding.
        blur_amount (int): Kernel size for Gaussian blur (odd number).
        ... other args ...
    """
    # --- Initialization (YOLO load, Video load, Trim setup - mostly same) ---
    if status_callback: status_callback("Initializing AI model...")
    # ... (YOLO load same as before) ...
    try: model = YOLO(yolo_model_name); print("YOLO model loaded.")
    except Exception as e: errmsg = f"Error loading YOLO: {e}"; print(errmsg); messagebox.showerror("AI Error", errmsg); return
    # ... (Video load/validation same as before) ...
    if not os.path.exists(input_video_path): print("..."); messagebox.showerror("..."); return
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened(): print("..."); messagebox.showerror("..."); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # ... (Get video properties, trim validation same as before) ...
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_original_frames / fps if total_original_frames > 0 and fps > 0 else 0
    frame_height_orig, frame_width_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Input: {frame_width_orig}x{frame_height_orig} @ {fps:.2f}, Frames: {total_original_frames} (~{duration_sec:.2f}s)")

    start_msec = start_time_sec * 1000; end_msec = end_time_sec * 1000 if end_time_sec > 0 else -1
    # ... (Trim time validation) ...

    start_frame_approx = int(start_time_sec * fps)
    end_frame_approx = int(end_time_sec * fps) if end_time_sec > 0 and end_time_sec <= duration_sec else total_original_frames
    frames_to_process = max(1, end_frame_approx - start_frame_approx)
    print(f"Processing: {start_time_sec:.2f}s to {end_time_sec if end_time_sec > 0 else duration_sec:.2f}s")
    print(f"Output: {out_width}x{out_height}, Conf: {confidence_threshold:.2f}, Smooth: {smoothing_factor:.2f}, BlurPad: {use_blur_padding}, BlurK: {blur_amount}")

    # --- Seek & Initial Frame Read (same as before) ---
    if start_msec > 0:
        # ... (seek logic) ...
        if status_callback: status_callback(f"Seeking to {start_time_sec:.1f}s...")
        seek_ok = cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
        time.sleep(0.1)
        ok, frame = cap.read()
        current_msec_after_seek = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"Seek done. Pos: {current_msec_after_seek/1000:.2f}s")
        if not seek_ok or abs(current_msec_after_seek - start_msec) > 1500: print("Warning: Seek inaccurate.")
    else:
        ok, frame = cap.read()
    if not ok: print("Error reading start frame."); messagebox.showerror(...); cap.release(); return
    frame_height, frame_width = frame.shape[:2]

    # --- Initial Target Identification (same as before) ---
    target_roi_xyxy = select_target_by_roi(frame, status_callback)
    if target_roi_xyxy is None: cap.release(); return

    target_bbox = None          # Raw bbox of detected target this frame
    last_known_center = None    # Raw center used for tracking association
    smoothed_center = None      # Smoothed center used for cropping
    initial_target_found = False

    if status_callback: status_callback("Detecting target in first frame...")
    # ... (YOLO predict on first frame, find best IoU match - same as before) ...
    try:
        results = model.predict(frame, classes=0, conf=confidence_threshold, verbose=False)
        best_iou = -1
        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls.item()) == 0:
                    det_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    iou = calculate_iou(target_roi_xyxy, det_xyxy)
                    if iou > best_iou: best_iou = iou; target_bbox = det_xyxy; initial_target_found = True
        if not initial_target_found or best_iou < 0.1:
            print("Error identifying target."); messagebox.showerror(...); cap.release(); return
        else:
            print(f"Initial target identified (IoU: {best_iou:.2f})")
            # Initialize BOTH raw and smoothed centers
            cx = (target_bbox[0] + target_bbox[2]) / 2
            cy = (target_bbox[1] + target_bbox[3]) / 2
            last_known_center = (cx, cy)
            smoothed_center = (cx, cy) # Start smoothed center at the first detection
    except Exception as e:
        print(f"Error initial detect: {e}"); messagebox.showerror(...); cap.release(); return


    # --- Video Writer Setup (same as before) ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    if not writer.isOpened(): print("Error opening writer."); messagebox.showerror(...); cap.release(); return

    print("\nProcessing video segment...")
    if not no_display: print("Press 'q'/ESC to stop early.")
    if status_callback: status_callback("Processing...")

    # --- Main Loop ---
    frame_count_in_range = 0; processed_count = 0
    stop_processing = False; target_currently_tracked = True
    first_iteration = True

    while not stop_processing:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if end_msec > 0 and current_msec > end_msec: break # End Time

        if not first_iteration: # Read next frame
            ok, frame = cap.read()
            if not ok: break # End of video
        else: first_iteration = False

        frame_count_in_range += 1
        timer = cv2.getTickCount()
        original_frame_copy = frame.copy() # Keep original for potential blur bg

        target_found_in_frame = False
        current_detections = []

        # --- YOLO Detection (same) ---
        try:
            results = model.predict(frame, classes=0, conf=confidence_threshold, verbose=False)
            if results and results[0].boxes:
                 for box in results[0].boxes:
                     if int(box.cls.item()) == 0:
                         det_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                         conf = float(box.conf.item())
                         current_detections.append({'bbox': det_xyxy, 'conf': conf})
        except Exception as e: print(f"Predict Error f{current_frame_num}: {e}"); target_currently_tracked = False


        # --- Tracking Logic (Nearest Neighbor - same association) ---
        if current_detections and last_known_center is not None:
            min_dist = float('inf'); best_match_box = None
            for det in current_detections:
                box = det['bbox']; cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
                dist_sq = (cx - last_known_center[0])**2 + (cy - last_known_center[1])**2
                max_allowed_dist_sq = (frame_width * 0.4)**2 + (frame_height*0.4)**2 # Slightly larger allowance?
                if dist_sq < min_dist and dist_sq < max_allowed_dist_sq:
                     min_dist = dist_sq; best_match_box = box; target_found_in_frame = True

            if target_found_in_frame:
                target_bbox = best_match_box # Update raw bbox for display
                # --- EMA Smoothing ---
                raw_cx = (target_bbox[0] + target_bbox[2]) / 2
                raw_cy = (target_bbox[1] + target_bbox[3]) / 2
                if smoothed_center is None: # Should have been initialized, but safety check
                    smoothed_center = (raw_cx, raw_cy)
                else:
                    smooth_cx = smoothing_factor * raw_cx + (1.0 - smoothing_factor) * smoothed_center[0]
                    smooth_cy = smoothing_factor * raw_cy + (1.0 - smoothing_factor) * smoothed_center[1]
                    smoothed_center = (smooth_cx, smooth_cy)
                # -------------------
                last_known_center = (raw_cx, raw_cy) # Update raw center for next frame association
                target_currently_tracked = True
            else: # No close detection found
                 target_currently_tracked = False
                 print(f"Tracking lost f{current_frame_num}: No close match.")
                 # Keep previous smoothed_center, don't update last_known_center? Or allow drift? Let's keep smoothed.
        else: # No detections at all or missing last known center
             target_currently_tracked = False
             print(f"Tracking lost f{current_frame_num}: No detections/init.")
             # Keep previous smoothed_center if it exists

        detect_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


        # --- Cropping & Writing Frame (Use smoothed center & new function) ---
        if smoothed_center: # Crop around the smoothed position
            cropped_frame = center_crop_frame_v2(original_frame_copy, # Pass original
                                                smoothed_center[0], smoothed_center[1],
                                                out_width, out_height,
                                                use_blur_padding, blur_amount)
            writer.write(cropped_frame)
            processed_count += 1
        else: # Fallback if smoothed_center never got set
             print(f"Error: smoothed_center None f{current_frame_num}. Writing black.")
             black_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
             writer.write(black_frame)


        # --- Display (Optional - draw on 'frame', show 'cropped_frame') ---
        if not no_display:
             display_frame = frame # Draw on the frame read (not the copy)
             # ... (Draw all detections - yellow) ...
             for det in current_detections: b=det['bbox']; c=det['conf']; cv2.rectangle(display_frame,(b[0],b[1]),(b[2],b[3]),(0,255,255),1); cv2.putText(display_frame,f'{c:.2f}',(b[0], b[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)

             # Highlight tracked target (using raw target_bbox for accuracy)
             if target_currently_tracked and target_bbox is not None:
                 b = target_bbox; cv2.rectangle(display_frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),2); cv2.putText(display_frame,"TARGET",(b[0],b[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                 # Also draw smoothed center for comparison
                 if smoothed_center: cv2.circle(display_frame, (int(smoothed_center[0]), int(smoothed_center[1])), 5, (255,0,255), -1) # Magenta dot
             elif smoothed_center: # Mark last known smoothed spot if tracking lost
                  cv2.circle(display_frame, (int(smoothed_center[0]), int(smoothed_center[1])), 10, (0,0,255), -1) # Red dot
                  cv2.putText(display_frame,"LOST",(int(smoothed_center[0])+15, int(smoothed_center[1])+5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

             # ... (Add stats - time, FPS, frame#) ...
             time_str = f"Time: {current_msec/1000:.2f}s"; cv2.putText(display_frame, time_str, (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
             cv2.putText(display_frame, f"Detect FPS: {int(detect_fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
             frame_text = f"Frame: {current_frame_num}"; cv2.putText(display_frame, frame_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

             try: # Display combined
                 display_orig_resized = cv2.resize(display_frame, (out_width // 2, out_height // 2))
                 # Ensure cropped_frame exists before resizing
                 if 'cropped_frame' in locals() and cropped_frame is not None:
                     display_crop_resized = cv2.resize(cropped_frame, (out_width // 2, out_height // 2))
                     combined_display = np.hstack((display_orig_resized, display_crop_resized))
                 else: # Handle case where cropping might fail
                     combined_display = display_orig_resized
                 cv2.imshow("Detection (Left) vs Cropped (Right) - Press 'q'/ESC", combined_display)
             except Exception as display_e: print(f"Display Error: {display_e}")

        # --- Update GUI Progress & Check Quit Key (same) ---
        if progress_callback: # ... (progress update logic) ...
            processed_in_intended_range = max(0, current_frame_num - start_frame_approx)
            progress_percent = int(100 * processed_in_intended_range / frames_to_process) if frames_to_process > 0 else 0
            progress_callback(min(100, progress_percent))
        if not no_display: # ... (quit key check) ...
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: print("Stopping early."); stop_processing = True; status_callback("Stopping...")


    # --- Cleanup (same) ---
    print(f"\nProcessed {processed_count} frames.")
    cap.release(); writer.release(); cv2.destroyAllWindows()
    if not stop_processing: print("Done!"); status_callback("Done!")
    else: print("Stopped."); status_callback("Stopped.")
    if progress_callback: progress_callback(0)


# --- Tkinter GUI Application Class (Updated for Smoothing & Blur) ---
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ski Tracker & Centering Tool (AI + Smooth/Blur)")

        # GUI Variables
        self.input_video_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        self.yolo_model_name = tk.StringVar(value='yolov8n.pt')
        self.confidence_threshold = tk.StringVar(value='0.40')
        self.smoothing_factor = tk.DoubleVar(value=0.3) # Use DoubleVar for float
        self.use_blur_padding = tk.BooleanVar(value=True) # Enable blur by default
        self.blur_amount = tk.IntVar(value=99)          # Use IntVar for int
        self.output_width = tk.StringVar(value='1280')
        self.output_height = tk.StringVar(value='720')
        self.start_time_str = tk.StringVar(value='0')
        self.end_time_str = tk.StringVar(value='0')
        self.no_display = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready (Using YOLO AI + Smoothing)")
        self.processing_active = False
        self.process_thread = None

        # --- GUI Layout ---
        mainframe = ttk.Frame(root, padding="10 10 10 10")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        current_row = 0
        # Input/Output (same)
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
        settings_frame.columnconfigure(3, weight=1) # Allow sliders/entries space

        # Row 0: AI Model, Output Size
        ttk.Label(settings_frame, text="AI Model:").grid(column=0, row=0, sticky=tk.W, padx=(0,5), pady=2)
        model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        model_combo = ttk.Combobox(settings_frame, width=12, textvariable=self.yolo_model_name, values=model_options)
        model_combo.grid(column=1, row=0, sticky=tk.W)

        ttk.Label(settings_frame, text="Output WxH:").grid(column=2, row=0, sticky=tk.W, padx=(15, 5), pady=2)
        width_entry = ttk.Entry(settings_frame, width=6, textvariable=self.output_width)
        width_entry.grid(column=3, row=0, sticky=tk.W)
        ttk.Label(settings_frame, text="x").grid(column=4, row=0, sticky=tk.W, padx=2)
        height_entry = ttk.Entry(settings_frame, width=6, textvariable=self.output_height)
        height_entry.grid(column=5, row=0, sticky=tk.W)

        # Row 1: Confidence, Smoothing
        ttk.Label(settings_frame, text="Confidence:").grid(column=0, row=1, sticky=tk.W, padx=(0,5), pady=5)
        conf_entry = ttk.Entry(settings_frame, width=6, textvariable=self.confidence_threshold)
        conf_entry.grid(column=1, row=1, sticky=tk.W)

        ttk.Label(settings_frame, text="Smoothing:").grid(column=2, row=1, sticky=tk.W, padx=(15, 5), pady=5)
        # Use a Scale widget for smoothing factor
        smooth_scale = ttk.Scale(settings_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, variable=self.smoothing_factor, length=100)
        smooth_scale.grid(column=3, row=1, sticky=tk.W+tk.E)
        smooth_label = ttk.Label(settings_frame, textvariable=self.smoothing_factor, width=4) # Show value
        smooth_label.grid(column=4, row=1, sticky=tk.W, padx=2)
        self.smoothing_factor.trace_add("write", lambda *args: smooth_label.config(text=f"{self.smoothing_factor.get():.2f}"))


        # Row 2: Blur Options
        blur_check = ttk.Checkbutton(settings_frame, text="Blur Padding", variable=self.use_blur_padding, onvalue=True, offvalue=False)
        blur_check.grid(column=0, row=2, sticky=tk.W, pady=5)

        ttk.Label(settings_frame, text="Blur Amount:").grid(column=2, row=2, sticky=tk.W, padx=(15, 5), pady=5)
        blur_entry = ttk.Entry(settings_frame, width=6, textvariable=self.blur_amount)
        blur_entry.grid(column=3, row=2, sticky=tk.W)
        ttk.Label(settings_frame, text="(ksize, odd)").grid(column=4, row=2, columnspan=2, sticky=tk.W, padx=5)


        # Row 3: Trim Times (same)
        ttk.Label(settings_frame, text="Start Time (s):").grid(column=0, row=3, sticky=tk.W, padx=(0,5), pady=5)
        start_time_entry = ttk.Entry(settings_frame, width=7, textvariable=self.start_time_str)
        start_time_entry.grid(column=1, row=3, sticky=tk.W)
        ttk.Label(settings_frame, text="End Time (s):").grid(column=2, row=3, sticky=tk.W, padx=(15, 5), pady=5)
        end_time_entry = ttk.Entry(settings_frame, width=7, textvariable=self.end_time_str)
        end_time_entry.grid(column=3, row=3, sticky=tk.W)
        ttk.Label(settings_frame, text="(0=end)").grid(column=4, row=3, columnspan=2, sticky=tk.W, padx=5)
        current_row += 1 # Increment after the settings frame block

        # No Display Checkbox (same)
        display_check = ttk.Checkbutton(mainframe, text="Hide Preview Windows (Faster)", variable=self.no_display, onvalue=True, offvalue=False)
        display_check.grid(column=0, row=current_row, columnspan=4, sticky=tk.W, pady=5)
        current_row += 1

        # --- Action Button, Progress Bar, Status Label (same) ---
        action_frame = ttk.Frame(mainframe); action_frame.grid(column=0, row=current_row, columnspan=4, sticky=(tk.W, tk.E), pady=10); action_frame.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_thread); self.start_button.grid(column=1, row=0, padx=10)
        self.progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, length=300, mode='determinate'); self.progress_bar.grid(column=0, row=0, sticky=(tk.W, tk.E))
        current_row += 1
        status_label = ttk.Label(mainframe, textvariable=self.status_text, relief=tk.SUNKEN, anchor=tk.W); status_label.grid(column=0, row=current_row, columnspan=4, sticky=(tk.W, tk.E), pady=5, ipady=2)

    # --- browse_input, browse_output, update_status, update_progress methods remain the same ---
    def browse_input(self):
        filetypes = (("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Input Video", filetypes=filetypes)
        if filepath: self.input_video_path.set(filepath); self.suggest_output_name()

    def suggest_output_name(self):
         if not self.output_video_path.get() and self.input_video_path.get():
            base, ext = os.path.splitext(self.input_video_path.get())
            suggested_output = f"{base}_centered_smooth{ext}"
            self.output_video_path.set(suggested_output)
         self.status_text.set("Input selected. Specify output & settings.")

    def browse_output(self):
        filetypes = (("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        initial_file = ""
        in_path = self.input_video_path.get()
        if in_path: base, ext = os.path.splitext(os.path.basename(in_path)); initial_file = f"{base}_centered_smooth.mp4"
        filepath = filedialog.asksaveasfilename(title="Save As", filetypes=filetypes, defaultextension=".mp4", initialfile=initial_file)
        if filepath: self.output_video_path.set(filepath); self.status_text.set("Output path selected.")

    def update_status(self, message): self.root.after(0, self.status_text.set, message)
    def update_progress(self, value): self.root.after(0, self.progress_bar.config, {'value': max(0, min(100, value))})


    def start_processing_thread(self):
        if self.processing_active: messagebox.showwarning("Busy", "Processing active."); return

        # --- Input Validation (Add checks for new params) ---
        input_path = self.input_video_path.get()
        output_path = self.output_video_path.get()
        model_name = self.yolo_model_name.get()
        if not input_path: messagebox.showerror("Input Error", "Select input video."); return
        if not output_path: messagebox.showerror("Input Error", "Specify output video."); return
        if input_path == output_path: messagebox.showerror("Input Error", "Input/output same."); return
        if not model_name: messagebox.showerror("Input Error", "Select AI model."); return

        try: width = int(self.output_width.get()); height = int(self.output_height.get()); assert width > 0 and height > 0
        except: messagebox.showerror("Input Error", "Invalid output dimensions."); return
        try: conf_thresh = float(self.confidence_threshold.get()); assert 0.0 <= conf_thresh <= 1.0
        except: messagebox.showerror("Input Error", "Invalid confidence (0.0-1.0)."); return
        try: smooth_factor = float(self.smoothing_factor.get()); assert 0.0 < smooth_factor <= 1.0
        except: messagebox.showerror("Input Error", "Invalid smoothing factor (0.01-1.0)."); return # Note: Slider ensures this, but check anyway
        try: blur_ksize = int(self.blur_amount.get()); assert blur_ksize > 0 and blur_ksize % 2 != 0
        except: messagebox.showerror("Input Error", "Invalid blur amount (must be positive odd integer)."); return

        try: start_time = float(self.start_time_str.get() or 0.0); end_time = float(self.end_time_str.get() or 0.0); assert start_time >= 0 and end_time >= 0; assert end_time == 0 or start_time < end_time
        except: messagebox.showerror("Input Error", "Invalid start/end times."); return

        blur_enable = self.use_blur_padding.get()

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
                  start_time, end_time, conf_thresh,
                  smooth_factor, blur_enable, blur_ksize, # Pass new args
                  self.update_status, self.update_progress),
            daemon=True
        )

        # --- Thread monitoring logic (same) ---
        def on_thread_complete():
            self.processing_active = False; self.start_button.config(state=tk.NORMAL)
            final_status = self.status_text.get()
            if final_status == "Done!": messagebox.showinfo("Success", f"Output saved:\n{output_path}")
        def check_thread():
            if self.process_thread and self.process_thread.is_alive(): self.root.after(100, check_thread)
            elif self.processing_active: on_thread_complete()

        self.process_thread.start()
        self.root.after(100, check_thread)


# --- Main execution ---
if __name__ == "__main__":
    print("AI Skier Tracking Tool w/ Smoothing & Blur")
    print("Requires 'ultralytics': pip install ultralytics")
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()