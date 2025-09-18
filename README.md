🖼️ PixelPerfect: AI Image Quality Analyzer
PixelPerfect is a two-stage deep learning application designed to analyze the quality of an image. It first classifies an image as either "Good" or "Poor" quality. If the image is determined to be of poor quality, it then identifies the specific type of defect from a set of predefined categories.

The application is built using Python with the Streamlit framework for the user interface and TensorFlow/Keras for the machine learning models.

🚀 How to Run the App
1. Project Structure
Ensure your project directory is organized as follows:

image_analyzer/
├── app.py
├── transfer_binary_analyzer_balanced.keras
├── grouped_multiclass_analyzer.h5
└── README.md

app.py: The main Streamlit application file.

transfer_binary_analyzer_balanced.keras: The binary classifier model (Good/Bad).

grouped_multiclass_analyzer.h5: The multiclass classifier model (defect types).

2. Setup
First, you need to install the necessary libraries. Open your terminal in the project directory and run:

pip install streamlit tensorflow opencv-python Pillow

3. Execution
With the dependencies installed, you can start the application with a single command:

streamlit run app.py

This will open the application in your default web browser, where you can upload images for analysis.

🧠 Model Architecture
The application uses a two-stage deep learning pipeline for robust image analysis.

Stage 1: Binary Classification
This stage uses a Transfer Learning model based on VGG16 to classify images as "Good" or "Poor." The VGG16 model, pre-trained on the ImageNet dataset, acts as a powerful feature extractor. We've added custom layers on top to adapt it for our specific task.

Input: An image uploaded by the user.

Output: A binary prediction (Good or Poor).

Stage 2: Multiclass Classification
If the binary classifier detects a poor-quality image, it is passed to this second model. This is a custom Convolutional Neural Network (CNN) that specializes in identifying the type of defect.

Input: A poor-quality image from Stage 1.

Output: The specific type of defect (e.g., Noise, Blur, Compression).

📊 Defect Categories
The multiclass model is trained to detect the following types of image defects:

Blur: The image is out of focus or appears fuzzy.

Color Distortion: The colors in the image are unnatural or inaccurate.

Compression: The image has visible compression artifacts, such as square-like blocks.

Luminance/Contrast: The image is either too dark, too bright, or has washed-out colors.

Noise: The image appears grainy, with random specks of color or light.

Transmission Errors: The image contains corrupted data, often appearing as streaks or missing parts.

Other: Other types of image quality issues not covered by the main categories.
