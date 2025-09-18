# 🖼️ PixelPerfect: AI Image Quality Analyzer

**PixelPerfect** is a two-stage deep learning application designed to analyze the quality of an image.  

- **Stage 1:** Classifies an image as either **"Good"** or **"Poor"** quality.  
- **Stage 2:** If the image is "Poor," it identifies the **specific type of defect** from a set of predefined categories.  

The application is built using **Python**, with:  
- **Streamlit** for the user interface  
- **TensorFlow/Keras** for machine learning models  

---

## 🚀 How to Run the App

### 1. Project Structure
Ensure your project directory is organized as follows:

image_analyzer/
├── app.py
├── transfer_binary_analyzer_balanced.keras
├── grouped_multiclass_analyzer.h5
└── README.md


- **`app.py`** → The main Streamlit application file  
- **`transfer_binary_analyzer_balanced.keras`** → Binary classifier model (Good/Bad)  
- **`grouped_multiclass_analyzer.h5`** → Multiclass classifier model (defect types)  

---

### 2. Setup
Install the required libraries:

pip install streamlit tensorflow opencv-python Pillow


3. Run the Application

Start the app with:

streamlit run app.py


This will launch the application in your default web browser, where you can upload images for analysis.

🧠 Model Architecture

PixelPerfect uses a two-stage deep learning pipeline for robust image quality analysis.

🔹 Stage 1: Binary Classification

Model: Transfer Learning (VGG16 pre-trained on ImageNet)

Input: User-uploaded image

Output: Good or Poor

The VGG16 backbone acts as a feature extractor, with custom layers fine-tuned for binary classification.

🔹 Stage 2: Multiclass Classification

Model: Custom Convolutional Neural Network (CNN)

Input: Poor-quality image (from Stage 1)

Output: Specific defect type

📊 Defect Categories

The multiclass model can detect the following image quality issues:

Blur → Image is out of focus or fuzzy

Color Distortion → Colors look unnatural or inaccurate

Compression → Visible compression artifacts (e.g., blockiness)

Luminance/Contrast → Too dark, too bright, or washed-out

Noise → Grainy appearance with random specks

Transmission Errors → Corrupted data (streaks, missing parts)

Other → Any other type of image quality degradation
