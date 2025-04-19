import cv2
import sys
import argparse
import numpy as np
import os

def select_roi_manually(frame):
    """Allows the user to manually select the Region of Interest (ROI)."""
    print("\n" + "="*30)
    print(" Manual ROI Selection")
    print(" Draw a box around the skier and press ENTER or SPACE.")
    print(" Press 'c' to cancel selection.")
    print("="*30 + "\n")
    roi = cv2.selectROI("Select Skier ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Skier ROI")
    if roi == (0, 0, 0, 0):  # Check if selection was cancelled or invalid
        print("ERROR: ROI selection cancelled or invalid.")
        return None
    print(f"ROI selected: {roi}")
    return roi

def create_tracker(tracker_type):
    """Creates an OpenCV tracker object based on the specified type."""
    print(f"Initializing tracker: {tracker_type}")
    if tracker_type == 'CSRT':
        # Generally more accurate, but slower
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        # Faster than CSRT, but might drift more
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
         # Very fast, but less accurate
        return cv2.legacy.TrackerMOSSE_create()
    # Add other trackers if needed (e.g., MIL, TLD, MedianFlow)
    # Note: Some legacy trackers might need 'cv2.legacy.Tracker...'
    else:
        print(f"Warning: Tracker type '{tracker_type}' not recognized or requires legacy access. Defaulting to CSRT.")
        return cv2.TrackerCSRT_create()

def center_crop_frame(frame, center_x, center_y, out_width, out_height):
    """Crops the frame around the center point, handling boundaries."""
    frame_h, frame_w = frame.shape[:2]

    # Calculate desired top-left corner
    crop_x1 = int(center_x - out_width / 2)
    crop_y1 = int(center_y - out_height / 2)

    # --- Boundary Adjustments ---
    # Calculate padding needed if crop goes out of bounds
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, (crop_x1 + out_width) - frame_w)
    pad_bottom = max(0, (crop_y1 + out_height) - frame_h)

    # Adjust crop coordinates to stay within frame boundaries
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_w, crop_x1 + out_width)
    crop_y2 = min(frame_h, crop_y1 + out_height)

    # Crop the valid region from the original frame
    cropped_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Apply padding if necessary to reach the target output size
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # Using black padding (0, 0, 0). Change color if needed.
        padded_frame = cv2.copyMakeBorder(cropped_region, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Ensure the final padded frame EXACTLY matches the output size
        # (might be needed due to rounding issues if padding wasn't exact)
        padded_frame = cv2.resize(padded_frame, (out_width, out_height))
        return padded_frame
    else:
        # If no padding was needed, the cropped region should already match the output size
        # Resize just in case of minor off-by-one errors in calculation
        return cv2.resize(cropped_region, (out_width, out_height))


def main(args):
    # --- 1. Setup ---
    tracker = create_tracker(args.tracker)
    if not os.path.exists(args.input_video):
        print(f"Error: Input video not found at '{args.input_video}'")
        sys.exit(1)

    # Open video file
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.input_video}")
        sys.exit(1)

    # Read the first frame
    ok, frame = cap.read()
    if not ok:
        print("Error: Cannot read video file.")
        cap.release()
        sys.exit(1)

    # Get video properties
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")

    # Determine Output Resolution
    out_width = args.width
    out_height = args.height
    print(f"Output Video Resolution: {out_width}x{out_height}")

    if out_width > frame_width or out_height > frame_height:
         print("Warning: Output resolution is larger than input resolution.")
         print("         The centered output will have black borders when the skier is near the edge.")


    # --- 2. Initial Object Selection ---
    bbox = select_roi_manually(frame)
    if bbox is None:
        cap.release()
        sys.exit(1)

    # --- 3. Initialize Tracker ---
    try:
        ok = tracker.init(frame, bbox)
        if not ok:
            raise Exception("Tracker initialization failed")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Try selecting a clearer, larger ROI or a different tracker type.")
        cap.release()
        sys.exit(1)

    # --- 4. Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
    # Alternative: fourcc = cv2.VideoWriter_fourcc(*'XVID') # for .avi
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
         print(f"Error: Could not open video writer for '{args.output_video}'")
         cap.release()
         sys.exit(1)

    print(f"\nProcessing video... Outputting to '{args.output_video}'")
    print("Press 'q' in the display window to stop early.")

    # --- 5. Frame-by-Frame Processing Loop ---
    frame_count = 0
    processed_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break # End of video

        frame_count += 1
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate FPS for tracking
        fps_track = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            # Tracking success
            (x, y, w, h) = [int(v) for v in bbox]

            # Calculate center of the tracked object
            center_x = x + w / 2
            center_y = y + h / 2

            # Crop the frame around the center
            cropped_frame = center_crop_frame(frame, center_x, center_y, out_width, out_height)

            # Write the cropped frame to the output video
            writer.write(cropped_frame)
            processed_count += 1

            # --- Optional: Display results ---
            if not args.no_display:
                # Draw bounding box on original frame (for reference)
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f"Tracker: {args.tracker}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
                cv2.putText(frame, f"FPS: {int(fps_track)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

                # Display Original and Cropped
                # Resize original slightly for side-by-side display if needed
                display_orig = cv2.resize(frame, (out_width, out_height)) # Or some fixed size
                combined_display = np.hstack((display_orig, cropped_frame))
                cv2.imshow("Original (Tracking) vs Cropped (Output)", combined_display)
                # cv2.imshow("Original (Tracking)", frame)
                # cv2.imshow("Cropped Output", cropped_frame)


        else:
            # Tracking failure
            print(f"Warning: Tracking failed on frame {frame_count}. Writing black frame.")
            # Create a black frame of the correct output size
            black_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            writer.write(black_frame)

            # --- Optional: Display failure ---
            if not args.no_display:
                cv2.putText(frame, "Tracking failure detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow("Original (Tracking) vs Cropped (Output)", frame) # Just show original on failure

        # Update progress bar (simple text version)
        progress = int(50 * frame_count / total_frames)
        sys.stdout.write(f"\rProgress: [{'=' * progress}{' ' * (50 - progress)}] {frame_count}/{total_frames} frames")
        sys.stdout.flush()

        # Exit if 'q' is pressed (only works if display is enabled)
        if not args.no_display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping early.")
                break

    # --- 6. Cleanup ---
    print(f"\nProcessed {processed_count} frames successfully.")
    print("Releasing resources...")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track an object (skier) in a video and output a centered crop.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_video", help="Path to save the processed output video file (e.g., output.mp4).")
    parser.add_argument("-t", "--tracker", type=str, default="CSRT",
                        help="OpenCV tracker type to use (e.g., CSRT, KCF, MOSSE). Default: CSRT")
    parser.add_argument("-W", "--width", type=int, default=1280,
                        help="Width of the output video frame in pixels. Default: 1280")
    parser.add_argument("-H", "--height", type=int, default=720,
                        help="Height of the output video frame in pixels. Default: 720")
    parser.add_argument("--no-display", action="store_true",
                         help="Run without displaying the video frames (faster processing).")

    args = parser.parse_args()
    main(args)