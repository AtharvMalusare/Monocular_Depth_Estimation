# Monocular Depth Estimation with MiDaS

This project utilizes the MiDaS model for monocular depth estimation. The MiDaS model predicts depth information from a single image, which can be useful for various computer vision tasks. The implementation leverages PyTorch and OpenCV to process live video feed and display depth maps in real-time.

## Installation

Ensure you have the required packages by installing the dependencies:

```bash
pip install torch torchvision opencv-python matplotlib

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/Sudo_User/your-repo-name.git
    cd your-repo-name
    ```

2. Run the depth estimation script:
    ```bash
    python depth_estimation.py
    ```

   This script captures video from your webcam, processes each frame to estimate depth, and displays both the original video and the depth map.

## Code Overview

- **Dependencies**: `cv2`, `torch`, `matplotlib`
- **Model**: MiDaS for depth estimation
- **Script**: `depth_estimation.py` - Contains the implementation for loading the MiDaS model, capturing video frames, applying transformations, and predicting depth.

## Additional Information

For a more detailed exploration of monocular depth estimation, refer to my article:
[Comparative Study on Monocular Depth Estimation](https://medium.com/@atharvmalusare/a-comparative-study-on-monocular-depth-estimation-a12f6b847087)
