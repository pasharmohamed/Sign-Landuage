# Sign-Landuage
# Arabic Sign Language (ArSL) Translation System

A comprehensive deep learning system for recognizing and translating Arabic Sign Language (ArSL) using the KArSL dataset. This project encompasses end-to-end data processing, Transformer-based model training, and a 3D web avatar representation.

# Key Features

- Robust Feature Extraction: Utilizes MediaPipe Tasks API (HandLandmarker & PoseLandmarker) to extract 258 key spatial coordinates (X, Y, Z, Visibility) per frame from raw video data.
- Deep Learning Architecture: Custom Transformer Network built with TensorFlow/Keras to handle the temporal sequence data of 502 distinct Arabic signs.
- Optimized Data Pipeline: Custom tf.keras.utils.Sequence Data Generators to process over 21,000 .npy sequence files without memory limitations.
- Real-Time 3D Avatar (Web Demo): A frontend web application using Three.js to render a 3D avatar that performs sign language gestures based on exported JSON landmark data.
- Continuous Signing Support: Built-in state machine logic for live inference, enabling dynamic clipping and sequence normalization for real-time camera translation.

## Technology Stack

- Machine Learning: Python, TensorFlow, Keras, NumPy, OpenCV
- Computer Vision: Google MediaPipe (Pose & Hand Landmarker)
- Web Interface: HTML/CSS, JavaScript, Three.js, Kalidokit
- Environment: Google Colab (for GPU training), Local Environment

## Project Structure

sign-language-system/
|
|-- backend/
|   |-- extract_from_video.py
|   |-- colab_karsl_502.ipynb
|   |-- live_inference.py
|
|-- web-demo/
|   |-- index.html
|   |-- js/
|   |-- avatar.glb
|
|-- README.md

## Getting Started

### Prerequisites
Ensure you have Python 3.9 or higher installed along with the required libraries:
pip install tensorflow mediapipe opencv-python numpy

### Model Training
1. Upload the KArSL dataset to your cloud storage or local workspace.
2. Run the provided Colab Notebook (colab_karsl_502.ipynb).
3. The notebook will automatically extract landmarks, build the Data Generator, train the Transformer model, and save best_model.keras.

### Running the Web Demo
To view the 3D Avatar representation, start a local server in the web-demo directory:
cd web-demo
python -m http.server 8000
Then navigate to http://localhost:8000/ in your browser.

## Model Architecture Details
The translation model is a scaled Transformer designed for temporal sequence classification:
- Input Shape: (30, 258) representing 30 frames and 258 coordinates.
- Normalization: Batch Normalization on inputs to stabilize raw MediaPipe coordinates.
- Core: 4 Transformer Blocks (d_model=256, 8 attention heads).
- Output: Softmax classification across 502 ArSL vocabulary classes.
