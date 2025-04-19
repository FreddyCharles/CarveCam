# AI Skier Auto-Centering Tool

This Python application uses AI (YOLOv8) to automatically track a skier in a video and generate a new video centered on the tracked person. It includes features like motion smoothing, optional blurred background padding, video trimming, and automatic hardware acceleration detection (GPU/CPU).

![Screenshot of the GUI Application](placeholder_gui_screenshot.png)
*(Recommendation: Replace this line with an actual screenshot of your application's GUI named `placeholder_gui_screenshot.png` or similar)*

## Features

*   **AI-Powered Tracking:** Utilizes YOLOv8 object detection to locate the skier in each frame.
*   **Automatic Centering:** Crops the output video frame around the detected skier's position.
*   **Motion Smoothing:** Implements Exponential Moving Average (EMA) to reduce jitter caused by minor variations in detection boxes.
*   **Blurred Background Padding:** Optionally fills areas outside the original video bounds (when centering near edges) with a blurred version of the frame instead of black bars.
*   **Video Trimming:** Process only a specific segment of the input video using start and end times.
*   **Adjustable AI Model:** Select different YOLOv8 model sizes (n, s, m, l, x) to balance speed and accuracy.
*   **Configurable Detection:** Adjust the confidence threshold for person detection.
*   **Configurable Smoothing:** Control the amount of motion smoothing applied.
*   **Configurable Blur:** Enable/disable blur padding and adjust the blur intensity.
*   **Hardware Acceleration:** Automatically detects and utilizes available CUDA (NVIDIA) or MPS (Apple Silicon) GPUs for faster processing, falling back to CPU if none are found.
*   **Hardware Info Display:** Shows detected CPU core count and GPU details in the GUI.
*   **Graphical User Interface (GUI):** Provides an easy-to-use interface built with Tkinter.
*   **Optional Preview:** View the original video with detections and the final cropped output side-by-side during processing (can be disabled for speed).

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    This project relies on several libraries. Install them using pip:
    ```bash
    pip install ultralytics opencv-python numpy torch torchvision torchaudio
    ```
    *   `ultralytics`: For the YOLOv8 model and interface.
    *   `opencv-python`: For video reading/writing and image processing.
    *   `numpy`: For numerical operations (used by OpenCV and others).
    *   `torch`, `torchvision`, `torchaudio`: The PyTorch deep learning framework (required by YOLOv8 and for GPU detection). *Note: This can be a large download.*

4.  **YOLOv8 Model Download:** The first time you run the application, the selected YOLOv8 model file (e.g., `yolov8n.pt`) will be automatically downloaded by the `ultralytics` library if it's not already present. This requires an internet connection on the first run for each model variant chosen.

## Usage

1.  **Navigate to the Directory:** Open your terminal or command prompt and go to the project directory where you cloned the repository.
2.  **Activate Virtual Environment:** If you created one, activate it (see step 2 in Installation).
3.  **Run the Application:**
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your main Python script, e.g., `ski_tracker_gui.py`)*

4.  **Using the GUI:**
    *   **Hardware Info:** The application will display detected CPU cores and GPU information at the top.
    *   **Input Video:** Click "Browse..." to select your skiing video file.
    *   **Output Video:** Click "Save As..." to choose the location and name for the processed video. A suggestion will be provided based on the input filename.
    *   **Settings:**
        *   **AI Model:** Choose a YOLOv8 variant (e.g., `yolov8n.pt` for speed, `yolov8x.pt` for accuracy).
        *   **Confidence:** Set the minimum confidence score (0.0-1.0) for detecting a person. Lower values detect more but might include errors; higher values are stricter.
        *   **Smoothing:** Adjust the slider (0.01-1.0). Lower values result in smoother but potentially laggier centering; higher values are more responsive but can be jittery.
        *   **Blur Padding:** Check the box to enable the blurred background effect when centering near edges.
        *   **Blur Amount:** Set the kernel size (positive odd integer) for the Gaussian blur used in padding. Higher values mean more blur.
        *   **Output WxH:** Define the desired resolution for the output video.
        *   **Start/End Time (s):** Specify the segment (in seconds) of the input video to process. Use `0` for the end time to process until the end.
    *   **Hide Preview Windows:** Check this box to disable the OpenCV display windows during processing. This significantly speeds up the process, especially on slower machines.
    *   **Start Processing:** Click this button to begin.
    *   **Select Target Skier:** An OpenCV window will pop up showing the first frame (or the frame at the specified start time). Click and drag a box tightly around the skier you want to track, then press **Enter** or **Space**. This helps the AI identify the correct person initially.
    *   **Monitor Progress:** Watch the progress bar and status messages in the main GUI window. If preview is enabled, you can press `q` or `ESC` in the preview window to stop early.

## Technology Stack

*   **Python:** Core programming language.
*   **Tkinter:** Standard Python library for the GUI.
*   **OpenCV (`opencv-python`):** Video I/O, image manipulation, drawing.
*   **Ultralytics (`ultralytics`):** YOLOv8 object detection model and interface.
*   **PyTorch (`torch`):** Backend deep learning framework for YOLOv8 and GPU detection.
*   **NumPy (`numpy`):** Fundamental package for numerical computation.

## Known Issues & Limitations

*   **Tracking Robustness:** The tracking relies on finding the closest detected person to the previous location. It might lose the target or switch to another person if:
    *   The skier is occluded for a long duration.
    *   Multiple skiers cross paths very closely.
    *   Detection fails momentarily (e.g., due to blur, low confidence).
*   **Performance:** Processing speed is highly dependent on your hardware (CPU speed, GPU presence/power), the chosen YOLO model size, and the video resolution. Real-time performance might require a powerful GPU. Disabling the preview helps.
*   **Video Seeking:** Seeking to the exact start time in videos is not always frame-accurate and depends on the video encoding.
*   **Blur Padding Performance:** Enabling blur padding adds extra processing steps (blurring, resizing, pasting) which can slightly slow down processing compared to black padding.

## Future Improvements Ideas

*   Implement more advanced tracking algorithms (e.g., DeepSORT, ByteTrack) that use appearance features (Re-Identification) for better robustness against occlusion and target switching.
*   Replace EMA smoothing with a Kalman Filter for potentially smoother and more predictive tracking.
*   Allow selecting the target from multiple initial detections if more than one person is present.
*   Add batch processing capabilities for multiple videos.
*   Enhance the GUI with more detailed progress information or a dedicated cancel button.
*   Optimize detection frequency (e.g., run YOLO every N frames and interpolate/use tracker in between) for better CPU performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(Recommendation: Create a `LICENSE` file in your repository, perhaps containing the standard MIT license text)*

---

