import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
from PIL import Image
import io
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Image Quality Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    
    .result-good {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #155724;
        margin: 1rem 0;
    }
    
    .result-bad {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #721c24;
        margin: 1rem 0;
    }
    
    .defect-info {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        color: #856404;
    }
    
    .upload-section {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #262730;
        margin: 1rem 0;
        color:white;
    }
</style>
""", unsafe_allow_html=True)

# Define the reverse label mapping for multiclass predictions
DEFECT_TYPE_MAPPING = {
    0: 'Blur',
    1: 'Color_Distortion', 
    2: 'Compression',
    3: 'Luminance_Contrast',
    4: 'Noise',
    5: 'Other',
    6: 'Transmission_Error'
}

def recreate_binary_model():
    """Recreate the binary model architecture to handle compatibility issues"""
    try:
        # Recreate VGG16 base
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        for layer in vgg_base.layers:
            layer.trainable = False
        
        # Recreate the transfer learning model
        model = Sequential([
            vgg_base,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Could not recreate binary model: {str(e)}")
        return None

def recreate_multiclass_model():
    """Recreate the multiclass model architecture"""
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 classes for defect types
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Could not recreate multiclass model: {str(e)}")
        return None

# Model loading with caching
from google.colab import drive

@st.cache_resource
def load_models():
    """Load both binary (.keras) and multiclass (.h5) models directly from Google Drive"""
    try:
        # --- Mount Google Drive ---
        drive.mount('/content/drive', force_remount=True)
        st.info("📂 Google Drive mounted successfully!")

        # --- Paths to models in Drive ---
        binary_model_path = "/content/drive/MyDrive/transfer_binary_analyzer_balanced.keras"
        multiclass_model_path = "/content/drive/MyDrive/grouped_multiclass_analyzer.h5"
        
        # --- Check existence ---
        if not os.path.exists(binary_model_path):
            st.error(f"❌ Binary model not found at: {binary_model_path}")
            st.info("Upload 'transfer_binary_analyzer_balanced.keras' to Google Drive → MyDrive")
            return None, None
            
        if not os.path.exists(multiclass_model_path):
            st.error(f"❌ Multiclass model not found at: {multiclass_model_path}")
            st.info("Upload 'grouped_multiclass_analyzer.h5' to Google Drive → MyDrive")
            return None, None
        
        binary_model, multiclass_model = None, None
        
        # --- Binary model (.keras) ---
        try:
            binary_model = load_model(binary_model_path, compile=False)
            binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            st.success("✅ Binary model loaded from Google Drive (.keras)")
        except Exception as e:
            st.error(f"⚠️ Could not load binary model directly: {str(e)}")
            try:
                binary_model = recreate_binary_model()
                if binary_model is not None:
                    binary_model.load_weights(binary_model_path)
                    st.success("✅ Binary model weights loaded into recreated architecture from Drive")
            except Exception as e2:
                st.error(f"❌ Failed to load binary model weights: {str(e2)}")
        
        # --- Multiclass model (.h5) ---
        try:
            multiclass_model = load_model(multiclass_model_path, compile=False)
            multiclass_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            st.success("✅ Multiclass model loaded from Google Drive (.h5)")
        except Exception as e:
            st.error(f"⚠️ Could not load multiclass model directly: {str(e)}")
            try:
                multiclass_model = recreate_multiclass_model()
                if multiclass_model is not None:
                    multiclass_model.load_weights(multiclass_model_path)
                    st.success("✅ Multiclass model weights loaded into recreated architecture from Drive")
            except Exception as e2:
                st.error(f"❌ Failed to load multiclass model weights: {str(e2)}")
        
        return binary_model, multiclass_model
    
    except Exception as e:
        st.error(f"🚨 Fatal error loading models: {str(e)}")
        return None, None


def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed (OpenCV format)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image_quality(image, binary_model, multiclass_model):
    """Predict image quality using the two-stage model pipeline"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
        
        # Stage 1: Binary classification (Good vs Bad)
        binary_prediction = binary_model.predict(processed_image, verbose=0)[0][0]
        binary_confidence = float(binary_prediction)
        
        # Determine if image is good or bad (threshold = 0.5)
        is_good_quality = binary_confidence >= 0.5
        
        if is_good_quality:
            return True, binary_confidence, None
        else:
            # Stage 2: Multiclass classification (type of defect)
            multiclass_prediction = multiclass_model.predict(processed_image, verbose=0)[0]
            predicted_class_index = np.argmax(multiclass_prediction)
            defect_confidence = float(multiclass_prediction[predicted_class_index])
            defect_type = DEFECT_TYPE_MAPPING.get(predicted_class_index, "Unknown")
            
            return False, binary_confidence, {
                'type': defect_type,
                'confidence': defect_confidence,
                'class_index': predicted_class_index
            }
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🖼️ PixelPerfect – AI Image Quality Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
   
    
    with st.spinner("Loading AI models..."):
        binary_model, multiclass_model = load_models()
    
    if binary_model is None or multiclass_model is None:
        st.warning("⚠️ Some models failed to load. Please check the error messages above.")
        st.info("""
        **Setup Instructions:**
        1. Place your model files in the same directory as this script:
           - `transfer_binary_analyzer_balanced.keras` (binary classifier)
           - `grouped_multiclass_analyzer.h5` (multiclass classifier)
        2. Ensure you have the correct TensorFlow version
        3. Restart the app
        
        **Note:** The .keras format is preferred over .h5 for better compatibility.
        """)
        return
    
    
    
    # Create three columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Upload section
        st.markdown('<div class="section-header">📁 Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                
                # Preview section
                st.markdown('<div class="section-header">🖼️ Image Preview</div>', unsafe_allow_html=True)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                
                # Analysis section
                st.markdown('<div class="section-header">🔍 Quality Analysis</div>', unsafe_allow_html=True)
                
                with st.spinner("Analyzing image quality..."):
                    is_good, binary_conf, defect_info = predict_image_quality(
                        image, binary_model, multiclass_model
                    )
                
                if is_good is not None:
                    # Display results
                    if is_good:
                        st.markdown(
                            '<div class="result-good">✅ Good Quality Image</div>',
                            unsafe_allow_html=True
                        )
                        
                        
                    else:
                        st.markdown(
                            '<div class="result-bad">❌ Poor Quality Image</div>',
                            unsafe_allow_html=True
                        )
                        
                        if defect_info:
                            defect_type = defect_info['type'].replace('_', ' ').title()
                            st.markdown(
                                f'<div class="defect-info">'
                                f'<strong>🔧 Detected Defect:</strong> {defect_type}<br>'
                                
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Additional defect information
                            defect_descriptions = {
                                'Noise': 'Random variation in brightness or color information',
                                'Blur': 'Loss of sharpness or focus in the image',
                                'Color_Distortion': 'Inaccurate color reproduction',
                                'Compression': 'Artifacts from lossy compression algorithms',
                                'Luminance_Contrast': 'Issues with brightness and contrast levels',
                                'Transmission_Error': 'Errors introduced during data transmission',
                                'Other': 'Other types of image quality issues'
                            }
                            
                            description = defect_descriptions.get(
                                defect_info['type'], 
                                'Quality issue detected'
                            )
                            st.caption(f"💡 **About this defect:** {description}")
                
                else:
                    st.error("❌ Failed to analyze image. Please try again.")
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        
        else:
            # Instructions when no file is uploaded
            st.markdown(
                '<div class="upload-section">'
                '<h3>👆 Upload an image to get started</h3>'
                '<p>The AI will analyze your image and determine:</p>'
                '<ul style="text-align: left; display: inline-block;">'
                '<li>✅ If the image has good quality</li>'
                '<li>❌ If the image has quality issues</li>'
                '<li>🔧 What type of defect is present (if any)</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True
            )
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📋 About")
        st.info("""
        This app uses deep learning models to analyze image quality:
        
        **Stage 1:** Binary classification determines if an image is good or bad quality.
        
        **Stage 2:** If bad quality is detected, a multiclass classifier identifies the specific type of defect.
        """)
        
        st.markdown("### 🎯 Detectable Defects")
        defect_list = [
            "🌫️ Blur",
            "🎨 Color Distortion", 
            "🗜️ Compression Artifacts",
            "💡 Luminance/Contrast Issues",
            "📡 Transmission Errors",
            "🔊 Noise",
            "❓ Other Issues"
        ]
        for defect in defect_list:
            st.write(f"• {defect}")
        

if __name__ == "__main__":
    main()
