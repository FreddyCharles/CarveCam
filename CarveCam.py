import cv2
import sys
import numpy as np
import os
import threading # To run processing in a separate thread

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk  # Themed widgets
from tkinter import filedialog
from tkinter import messagebox

# --- Existing Helper Functions (select_roi_manually, create_tracker, center_crop_frame) ---
# These remain largely the same as before.

def select_roi_manually(frame):
    """Allows the user to manually select the Region of Interest (ROI)."""
    # Make window topmost (helps on some systems)
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window

    messagebox.showinfo("Select ROI",
                        "An OpenCV window will open.\n"
                        "Draw a box around the skier and press ENTER or SPACE.\n"
                        "Press 'c' to cancel selection.",
                        parent=None) # Make messagebox topmost
    root.destroy() # Close the temporary root

    cv2.namedWindow("Select Skier ROI", cv2.WINDOW_NORMAL) # Allow resizing
    cv2.setWindowProperty("Select Skier ROI", cv2.WND_PROP_TOPMOST, 1) # Try to bring to front

    roi = cv2.selectROI("Select Skier ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Skier ROI")

    if roi == (0, 0, 0, 0):
        print("ERROR: ROI selection cancelled or invalid.")
        messagebox.showerror("Error", "ROI selection cancelled or invalid.")
        return None
    print(f"ROI selected: {roi}")
    return roi

def create_tracker(tracker_type):
    """Creates an OpenCV tracker object based on the specified type."""
    print(f"Initializing tracker: {tracker_type}")
    tracker = None
    try:
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'MOSSE':
             # Check if legacy module is available
             if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                 tracker = cv2.legacy.TrackerMOSSE_create()
             else:
                 print("Warning: Legacy MOSSE tracker not found in this OpenCV version. Try CSRT or KCF.")
                 return None # Indicate failure
        # Add other trackers if needed
        else:
            print(f"Warning: Tracker type '{tracker_type}' not recognized. Defaulting to CSRT.")
            tracker = cv2.TrackerCSRT_create()

        if tracker is None and tracker_type != 'MOSSE': # Handle case where default CSRT might fail if not built
             print(f"Error: Could not create tracker '{tracker_type}'. Is opencv-contrib-python installed correctly?")
             return None

    except AttributeError as e:
         print(f"Error creating tracker '{tracker_type}'. Might require 'opencv-contrib-python'. Error: {e}")
         messagebox.showerror("Tracker Error", f"Could not create tracker '{tracker_type}'.\nMake sure 'opencv-contrib-python' is installed.\n\nError: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred creating tracker: {e}")
        messagebox.showerror("Tracker Error", f"An unexpected error occurred creating the tracker:\n{e}")
        return None

    return tracker


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
        # Ensure exact size after padding
        current_h, current_w = padded_frame.shape[:2]
        if current_h != out_height or current_w != out_width:
             padded_frame = cv2.resize(padded_frame, (out_width, out_height), interpolation=cv2.INTER_NEAREST) # Use INTER_NEAREST to avoid blurring edges
        return padded_frame
    else:
        # Resize only if necessary (e.g., rounding issues)
        current_h, current_w = cropped_region.shape[:2]
        if current_h != out_height or current_w != out_width:
            return cv2.resize(cropped_region, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        else:
            return cropped_region


# --- Core Processing Logic (modified from original main) ---
def run_tracker_processing(input_video_path, output_video_path, tracker_type, out_width, out_height, no_display, status_callback=None, progress_callback=None):
    """
    Performs the object tracking and video centering.
    Args:
        input_video_path (str): Path to input video.
        output_video_path (str): Path to save output video.
        tracker_type (str): Type of OpenCV tracker to use.
        out_width (int): Width of output video.
        out_height (int): Height of output video.
        no_display (bool): If True, suppress OpenCV display windows.
        status_callback (func, optional): Function to update GUI status label.
        progress_callback(func, optional): Function to update GUI progress bar.
    """
    if status_callback: status_callback("Initializing...")

    tracker = create_tracker(tracker_type)
    if tracker is None:
        if status_callback: status_callback("Error: Failed to create tracker.")
        return # Exit if tracker creation failed

    if not os.path.exists(input_video_path):
        errmsg = f"Error: Input video not found at '{input_video_path}'"
        print(errmsg)
        messagebox.showerror("File Error", errmsg)
        if status_callback: status_callback("Error: Input file not found.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        errmsg = f"Error: Could not open video file: {input_video_path}"
        print(errmsg)
        messagebox.showerror("Video Error", errmsg)
        if status_callback: status_callback("Error: Cannot open video.")
        return

    ok, frame = cap.read()
    if not ok:
        errmsg = "Error: Cannot read the first frame of the video."
        print(errmsg)
        messagebox.showerror("Video Error", errmsg)
        if status_callback: status_callback("Error: Cannot read video.")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: # Handle cases where total_frames might not be readable
        total_frames = -1 # Indicate unknown total
        print("Warning: Could not determine total number of frames.")

    print(f"Input Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, ~{total_frames} frames")
    print(f"Output Video Resolution: {out_width}x{out_height}")
    if status_callback: status_callback("Input loaded. Select ROI...")

    if out_width > frame_width or out_height > frame_height:
         print("Warning: Output resolution is larger than input. Black borders may appear.")
         messagebox.showwarning("Resolution Warning",
                                "Output resolution is larger than input resolution.\n"
                                "The centered output may have black borders.")

    bbox = select_roi_manually(frame)
    if bbox is None:
        if status_callback: status_callback("ROI selection cancelled. Aborting.")
        cap.release()
        return

    try:
        ok = tracker.init(frame, bbox)
        if not ok: raise Exception("Tracker init returned false")
    except Exception as e:
        errmsg = f"Error initializing tracker: {e}\nTry selecting a clearer ROI or different tracker."
        print(errmsg)
        messagebox.showerror("Tracker Error", errmsg)
        if status_callback: status_callback("Error: Tracker init failed.")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Ensure this codec works for you
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
         errmsg = f"Error: Could not open video writer for '{output_video_path}'"
         print(errmsg)
         messagebox.showerror("File Error", errmsg)
         if status_callback: status_callback("Error: Cannot create output file.")
         cap.release()
         return

    print(f"\nProcessing video... Outputting to '{output_video_path}'")
    if not no_display: print("Press 'q' in the display window to stop early.")
    if status_callback: status_callback("Processing...")


    frame_count = 0
    processed_count = 0
    stop_processing = False # Flag to stop from 'q' key

    while not stop_processing:
        ok, frame = cap.read()
        if not ok:
            break # End of video

        frame_count += 1
        timer = cv2.getTickCount()

        ok_update, bbox = tracker.update(frame)

        fps_track = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok_update:
            (x, y, w, h) = [int(v) for v in bbox]
            center_x = x + w / 2
            center_y = y + h / 2
            cropped_frame = center_crop_frame(frame, center_x, center_y, out_width, out_height)
            writer.write(cropped_frame)
            processed_count += 1

            if not no_display:
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f"Tracker: {tracker_type}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
                cv2.putText(frame, f"FPS: {int(fps_track)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
                frame_text = f"Frame: {frame_count}"
                if total_frames > 0: frame_text += f"/{total_frames}"
                cv2.putText(frame, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

                try:
                    display_orig = cv2.resize(frame, (out_width // 2, out_height // 2)) # Smaller display
                    display_crop = cv2.resize(cropped_frame, (out_width // 2, out_height // 2))
                    combined_display = np.hstack((display_orig, display_crop))
                    cv2.imshow("Original (Left) vs Cropped (Right) - Press 'q' to stop", combined_display)
                except Exception as display_e: # Catch errors if resizing fails
                    print(f"Warning: Error displaying frames: {display_e}")
                    cv2.imshow("Cropped Output - Press 'q' to stop", cropped_frame) # Fallback


        else:
            # Tracking failure - write black frame
            print(f"Warning: Tracking failed on frame {frame_count}. Writing black frame.")
            black_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            writer.write(black_frame)
            if not no_display:
                cv2.putText(frame, "Tracking failure detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                try:
                    display_fail = cv2.resize(frame, (out_width // 2, out_height // 2))
                    cv2.imshow("Original (Left) vs Cropped (Right) - Press 'q' to stop", display_fail)
                except Exception as display_e:
                     print(f"Warning: Error displaying failure frame: {display_e}")


        # Update GUI Progress (if callbacks provided)
        if total_frames > 0 and progress_callback:
             progress_percent = int(100 * frame_count / total_frames)
             progress_callback(progress_percent)
        elif progress_callback:
             # Simple update if total frames unknown
             progress_callback(frame_count % 100) # Cycle 0-99

        # Check for Quit Key (only if display is enabled)
        if not no_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n'q' pressed, stopping early.")
                if status_callback: status_callback("Stopping...")
                stop_processing = True
            elif key == 27: # Esc key
                 print("\nESC pressed, stopping early.")
                 if status_callback: status_callback("Stopping...")
                 stop_processing = True


    # --- Cleanup ---
    print(f"\nProcessed {processed_count} frames successfully.")
    print("Releasing resources...")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    final_message = f"Processing complete. Output saved to:\n{output_video_path}"
    print(final_message)
    if status_callback: status_callback("Done!")
    # Don't show messagebox here if stopping early via 'q'
    if not stop_processing:
        messagebox.showinfo("Success", final_message)
    if progress_callback: progress_callback(0) # Reset progress


# --- Tkinter GUI Application Class ---
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ski Tracker & Centering Tool")
        # self.root.geometry("550x350") # Adjust size as needed

        # Variables to store GUI state
        self.input_video_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        self.tracker_type = tk.StringVar(value='CSRT') # Default tracker
        self.output_width = tk.StringVar(value='1280')
        self.output_height = tk.StringVar(value='720')
        self.no_display = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self.processing_active = False # Flag to prevent multiple runs

        # --- GUI Layout ---
        mainframe = ttk.Frame(root, padding="10 10 10 10")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Input File Selection
        ttk.Label(mainframe, text="Input Video:").grid(column=0, row=0, sticky=tk.W, pady=5)
        input_entry = ttk.Entry(mainframe, width=40, textvariable=self.input_video_path, state='readonly')
        input_entry.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(mainframe, text="Browse...", command=self.browse_input).grid(column=2, row=0, sticky=tk.W)

        # Output File Selection
        ttk.Label(mainframe, text="Output Video:").grid(column=0, row=1, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(mainframe, width=40, textvariable=self.output_video_path, state='readonly')
        output_entry.grid(column=1, row=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(mainframe, text="Save As...", command=self.browse_output).grid(column=2, row=1, sticky=tk.W)

        # Tracker Selection
        ttk.Label(mainframe, text="Tracker Type:").grid(column=0, row=2, sticky=tk.W, pady=5)
        tracker_options = ['CSRT', 'KCF', 'MOSSE'] # Add others if implemented
        tracker_combo = ttk.Combobox(mainframe, textvariable=self.tracker_type, values=tracker_options, state='readonly')
        tracker_combo.grid(column=1, row=2, sticky=tk.W, padx=5)

        # Output Dimensions
        dim_frame = ttk.Frame(mainframe)
        dim_frame.grid(column=0, row=3, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(dim_frame, text="Output Size (WxH):").pack(side=tk.LEFT)
        width_entry = ttk.Entry(dim_frame, width=7, textvariable=self.output_width)
        width_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(dim_frame, text="x").pack(side=tk.LEFT)
        height_entry = ttk.Entry(dim_frame, width=7, textvariable=self.output_height)
        height_entry.pack(side=tk.LEFT, padx=5)

        # No Display Checkbox
        display_check = ttk.Checkbutton(mainframe, text="Hide Preview Windows (Faster)",
                                        variable=self.no_display, onvalue=True, offvalue=False)
        display_check.grid(column=0, row=4, columnspan=3, sticky=tk.W, pady=5)


        # --- Action Button, Progress Bar, Status Label ---
        action_frame = ttk.Frame(mainframe)
        action_frame.grid(column=0, row=5, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        action_frame.columnconfigure(0, weight=1) # Make progress bar expand

        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.grid(column=1, row=0, padx=10) # Button on the right

        self.progress_bar = ttk.Progressbar(action_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(column=0, row=0, sticky=(tk.W, tk.E)) # Progress bar takes available space

        status_label = ttk.Label(mainframe, textvariable=self.status_text, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(column=0, row=6, columnspan=3, sticky=(tk.W, tk.E), pady=5, ipady=2)


    def browse_input(self):
        filetypes = (("MP4 files", "*.mp4"),
                     ("AVI files", "*.avi"),
                     ("MOV files", "*.mov"),
                     ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select Input Video", filetypes=filetypes)
        if filepath:
            self.input_video_path.set(filepath)
            # Auto-suggest output name
            if not self.output_video_path.get():
                base, ext = os.path.splitext(filepath)
                suggested_output = f"{base}_centered{ext}"
                self.output_video_path.set(suggested_output)
            self.status_text.set("Input selected. Specify output and settings.")

    def browse_output(self):
        filetypes = (("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        # Suggest filename based on input if possible
        initial_file = ""
        in_path = self.input_video_path.get()
        if in_path:
             base, ext = os.path.splitext(os.path.basename(in_path))
             initial_file = f"{base}_centered.mp4" # Default to mp4

        filepath = filedialog.asksaveasfilename(title="Save Output Video As",
                                                 filetypes=filetypes,
                                                 defaultextension=".mp4",
                                                 initialfile=initial_file)
        if filepath:
            self.output_video_path.set(filepath)
            self.status_text.set("Output path selected.")

    def update_status(self, message):
        # Ensure GUI updates happen on the main thread
        self.root.after(0, self.status_text.set, message)

    def update_progress(self, value):
         # Ensure GUI updates happen on the main thread
         self.root.after(0, self.progress_bar.config, {'value': value})


    def start_processing_thread(self):
        if self.processing_active:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # --- Input Validation ---
        input_path = self.input_video_path.get()
        output_path = self.output_video_path.get()
        tracker = self.tracker_type.get()
        try:
            width = int(self.output_width.get())
            height = int(self.output_height.get())
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")
        except ValueError:
            messagebox.showerror("Invalid Input", "Output width and height must be positive integers.")
            return

        if not input_path:
            messagebox.showerror("Invalid Input", "Please select an input video file.")
            return
        if not output_path:
            messagebox.showerror("Invalid Input", "Please specify an output video file path.")
            return
        if input_path == output_path:
             messagebox.showerror("Invalid Input", "Input and output file paths cannot be the same.")
             return
        if not tracker:
            messagebox.showerror("Invalid Input", "Please select a tracker type.")
            return

        # Disable button, set status
        self.processing_active = True
        self.start_button.config(state=tk.DISABLED)
        self.update_status("Starting...")
        self.update_progress(0)


        # --- Run processing in a separate thread ---
        # This prevents the GUI from freezing
        process_thread = threading.Thread(
            target=run_tracker_processing,
            args=(input_path, output_path, tracker, width, height, self.no_display.get(), self.update_status, self.update_progress),
            daemon=True # Allows closing the app even if thread is running (might leave files incomplete)
        )

        # Define what happens when the thread finishes (re-enable button)
        def on_thread_complete():
            self.processing_active = False
            self.start_button.config(state=tk.NORMAL)
            # Status is set by the processing function itself upon completion/error

        # Use 'after' to check thread status periodically without blocking GUI
        def check_thread():
            if process_thread.is_alive():
                self.root.after(100, check_thread) # Check again in 100ms
            else:
                on_thread_complete()

        process_thread.start()
        self.root.after(100, check_thread) # Start checking the thread

# --- Main execution ---
if __name__ == "__main__":
    # Check if opencv-contrib-python seems installed (basic check)
    if not hasattr(cv2, 'TrackerCSRT_create'):
         print("Warning: cv2.TrackerCSRT_create not found.")
         print("Ensure 'opencv-contrib-python' is installed (`pip install opencv-contrib-python`)")
         # Optionally show a messagebox, but print might be enough before GUI starts

    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()