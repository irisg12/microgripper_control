# Microgripper Vision Submodule

The `microgripper_vision` submodule is responsible for detecting and analyzing microgrippers in video frames. It uses computer vision techniques to process images and extract relevant features such as centroids, angles, and contours of microgrippers.

## Features
- **Microgripper Detection**: Detects microgrippers in video frames using OpenCV.
- **Contour Analysis**: Identifies and processes contours to extract geometric properties.
- **Visualization**: Displays processed video frames with detected features highlighted.

## Requirements
The following Python packages are required to run the submodule:
- `opencv-python`
- `numpy`

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Files
- `microgripperDetectionNew.py`: Main script for detecting microgrippers in video frames.
- `image_subscriber.py`: Script for subscribing to image streams and processing them in real-time.

## Usage
1. Ensure you have a video file (e.g., `new_vid2.mp4`) in the appropriate directory.
2. Run the detection script:
   ```bash
   python microgripperDetectionNew.py
   ```
3. The script will process the video and display the results in a window.

## Notes
- The detection algorithm uses adaptive thresholding and contour analysis to identify microgrippers.
- Adjust parameters such as `SEARCH_AREA` and kernel size in the script to fine-tune detection for specific use cases.

## Troubleshooting
- If OpenCV is not installed, install it using:
  ```bash
  pip install opencv-python
  ```
- Ensure the video file path is correct in the script.

## License
This submodule is part of the MicroGripperControl project and follows the same licensing terms.