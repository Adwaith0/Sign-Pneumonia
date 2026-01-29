"""
Hybrid PIL-Inspired Model Inference Pipeline

INFERENCE ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│ INPUT IMAGE (X-ray, medical imaging)                            │
├─────────────────────────────────────────────────────────────────┤
│ ↓                                                               │
│ [COMPONENT 1] CNN FEATURE EXTRACTOR (Trained via Backprop)     │
│   Conv2D(32) → ReLU → MaxPooling2D                             │
│   Conv2D(64) → ReLU → MaxPooling2D                             │
│   Conv2D(128) → ReLU → MaxPooling2D                            │
│   Flatten() → Feature Vector                                    │
│ ↓                                                               │
│ [COMPONENT 2] LINEAR CLASSIFIER (Trained via Pseudoinverse)    │
│   Dense (no activation): z = w^T f + b                         │
│   Weights: Computed via Moore-Penrose pseudoinverse (NO BP)    │
│ ↓                                                               │
│ [SIGMOID ACTIVATION]                                            │
│   p = 1 / (1 + exp(-z))  → Probability [0, 1]                 │
├─────────────────────────────────────────────────────────────────┤
│ OUTPUT: Probability for each class (Normal / Pneumonia)        │
└─────────────────────────────────────────────────────────────────┘

KEY RESEARCH FEATURES:
• Hybrid learning: Gradient-based (CNN) + Gradient-free (PIL)
• Pseudoinverse learning (PIL): w = X^+ y (closed-form, no SGD)
• Analytical solution: Minimum-norm least-squares optimal weights
• Gradient-free classification layer: No backpropagation
• Block-wise framework: Different learning paradigms per component

INFERENCE PROPERTIES:
• Feature extraction: Standard forward pass through CNN
• Classification: Linear transformation with analytically computed weights
• No model.predict() with sigmoid - explicit manual computation
"""

import streamlit as st

# ⚠️ IMPORTANT: set_page_config() MUST be the first Streamlit command
st.set_page_config(page_title="Medical Imaging Diagnosis", layout="wide")

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import json

# =========================================================================
# Configuration & Model Paths
# =========================================================================
CNN_EXTRACTOR_PATH = "./model/cnn_feature_extractor.keras"
CLASSIFIER_WEIGHTS_PATH = "./model/pseudoinverse_classifier.pkl"
METADATA_PATH = "./model/training_metadata.json"
FALLBACK_MODEL_PATH = "./model/medical_cnn.keras"

# =========================================================================
# Load Hybrid Model Components
# =========================================================================
@st.cache_resource
def load_hybrid_model():
    """
    Load the hybrid PIL-inspired model components.
    
    COMPONENTS LOADED:
    1. CNN Feature Extractor (.keras)
       - Trained via standard backpropagation
       - Converts images to feature vectors
       - Gradient: ENABLED (during training) → FROZEN (during inference)
    
    2. Pseudoinverse Classifier Weights (.pkl)
       - Computed via Moore-Penrose pseudoinverse
       - Gradient: NEVER USED (analytical solution only)
       - No backpropagation through this layer
    
    RETURNS:
        - cnn_extractor: Keras model for feature extraction
        - classifier_weights_dict: Dictionary with learned weights and bias
        - input_shape: Expected input dimensions (H, W, C)
    """
    
    print("🔍 Loading Hybrid PIL-Inspired Model...")
    
    # Load CNN feature extractor
    if not os.path.exists(CNN_EXTRACTOR_PATH):
        st.error(f"❌ CNN feature extractor not found: {CNN_EXTRACTOR_PATH}")
        st.info("Please train the model first: python train_model.py")
        st.stop()
    
    # Load pseudoinverse classifier weights
    if not os.path.exists(CLASSIFIER_WEIGHTS_PATH):
        st.error(f"❌ Pseudoinverse classifier not found: {CLASSIFIER_WEIGHTS_PATH}")
        st.info("Please train the model first: python train_model.py")
        st.stop()
    
    # Load CNN (compile=False: we only use for inference, no training)
    cnn_extractor = keras.models.load_model(CNN_EXTRACTOR_PATH, compile=False)
    
    # Load pseudoinverse classifier weights
    with open(CLASSIFIER_WEIGHTS_PATH, "rb") as f:
        classifier_weights_dict = pickle.load(f)
    
    input_shape = cnn_extractor.input_shape  # (None, H, W, C)
    in_h, in_w, in_c = input_shape[1], input_shape[2], input_shape[3]
    
    print(f"✓ CNN feature extractor loaded")
    print(f"✓ Pseudoinverse classifier loaded")
    print(f"✓ Input shape: ({in_h}, {in_w}, {in_c})")
    
    return cnn_extractor, classifier_weights_dict, (in_h, in_w, in_c)


try:
    cnn_extractor, classifier_weights, (IN_H, IN_W, IN_C) = load_hybrid_model()
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    MODEL_LOADED = False

# Extract classifier parameters
if MODEL_LOADED:
    w_classifier = classifier_weights["weights"]          # Shape: (feature_dim, 1)
    bias = classifier_weights["bias"]                      # Scalar
    training_method = classifier_weights["training_method"]  # String for info
else:
    w_classifier = None
    bias = None
    training_method = "Unknown"

# =========================================================================
# Image Preprocessing
# =========================================================================
def preprocess_image(pil_img):
    """
    Preprocess image to match CNN input requirements.
    
    Args:
        pil_img: PIL Image object
        
    Returns:
        arr: Preprocessed numpy array with shape (1, IN_H, IN_W, IN_C)
    """
    if IN_C == 1:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize((IN_W, IN_H))
    arr = np.array(pil_img).astype("float32") / 255.0

    if IN_C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    return arr


# =========================================================================
# Hybrid Inference: CNN Feature Extraction + Pseudoinverse Classification
# =========================================================================
def predict_hybrid(img_batch):
    """
    HYBRID PIL-INSPIRED PREDICTION PIPELINE
    
    STEP 1: Feature Extraction (CNN)
    ────────────────────────────────
    Extract dense feature representation from image using trained CNN:
        f = CNN(x)  ← Feature vector (gradient-frozen during inference)
    
    STEP 2: Linear Classification (PIL - Gradient-Free)
    ────────────────────────────────────────────────────
    Compute logit (raw score) using analytically computed weights:
        z = w^T f + b  ← Linear transformation
    
    Key property: Weights w were computed via Moore-Penrose pseudoinverse,
    NOT via backpropagation. This is gradient-free learning.
    
        w = X^+ y  ← Closed-form solution (no SGD, no learning rate)
        where X^+ is Moore-Penrose pseudoinverse of feature matrix X
              y is the label vector
    
    STEP 3: Sigmoid Activation
    ──────────────────────────
    Convert logit to probability:
        p = σ(z) = 1 / (1 + exp(-z))  ← Probability in [0, 1]
    
    FINAL OUTPUT: Probabilities for each class
    
    HYBRID NATURE:
    • CNN backbone: Standard gradient-based training
    • Classification layer: Analytical (gradient-free) learning
    • Framework: Block-wise learning with mixed paradigms
    
    Args:
        img_batch: Numpy array with shape (batch_size, IN_H, IN_W, IN_C)
        
    Returns:
        dict: Probabilities for each class {class_name: probability}
    """
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Feature Extraction (CNN Feature Extractor)
    # ─────────────────────────────────────────────────────────────────
    # Extract high-level features using the trained CNN
    # CNN weights are frozen during inference (loaded as pretrained)
    features = cnn_extractor.predict(img_batch, verbose=0)  # Shape: (batch_size, feature_dim)
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Linear Classification (Pseudoinverse-Learned Weights)
    # ─────────────────────────────────────────────────────────────────
    # Compute logit using analytically computed weights
    # These weights were NOT learned via backpropagation!
    # They are the optimal least-squares solution: w = X^+ y
    logits = features @ w_classifier + bias  # Shape: (batch_size, 1)
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Sigmoid Activation (Probability Conversion)
    # ─────────────────────────────────────────────────────────────────
    # Convert logit to probability using sigmoid function
    # p = 1 / (1 + exp(-z))
    probabilities = 1.0 / (1.0 + np.exp(-logits))  # Shape: (batch_size, 1)
    
    # Extract first (and only) sample in batch
    prob_pneumonia = float(probabilities[0][0])
    prob_normal = 1.0 - prob_pneumonia
    
    return {
        "Normal": prob_normal,
        "Pneumonia": prob_pneumonia
    }


# =========================================================================
# Streamlit UI
# =========================================================================
st.title("🏥 Medical Imaging Diagnosis - Pneumonia Detection")
st.markdown("**Hybrid PIL-Inspired CNN + Pseudoinverse Learning**")

# Display model information
with st.expander("ℹ️ About This Model (Architecture & Learning Framework)"):
    st.markdown("""
    ### Hybrid Learning Framework: PIL-Inspired Block-wise Learning
    
    This model combines **two different learning paradigms** in a block-wise approach:
    
    #### Component 1: CNN Feature Extractor (Gradient-Based)
    - **Architecture**: Conv2D → ReLU → MaxPooling (3 blocks)
    - **Training**: Standard backpropagation with cross-entropy loss
    - **Gradients**: Enabled during training, frozen during inference
    - **Purpose**: Learn discriminative feature representations from images
    
    #### Component 2: Linear Classifier (Gradient-Free)
    - **Architecture**: Dense layer (no activation) with analytically computed weights
    - **Training**: Moore-Penrose pseudoinverse (PIL - Pseudoinverse Learning)
    - **Gradients**: NEVER used - closed-form analytical solution only
    - **Weights**: w = X^+ y (where X^+ is the pseudoinverse)
    - **Property**: Optimal minimum-norm least-squares solution
    - **Framework**: Inspired by ELM (Extreme Learning Machines) and DAN (Deep Adaptive Networks)
    
    ### Inference Pipeline
    1. **Feature Extraction**: Image → CNN → Feature Vector
    2. **Classification**: Feature Vector → Linear Transform (w^T f + b) → Logit
    3. **Probability**: Logit → Sigmoid → Probability
    
    ### Research Novelty
    - **Hybrid approach**: Not full gradient-free, but selective/block-wise
    - **Analytical solution**: No SGD for classification layer - computationally efficient
    - **Closed-form**: Guaranteed minimum-norm least-squares optimal weights
    - **Theoretical grounding**: Directly inspired by PIL/ELM/DAN literature
    
    ### Key Advantages
    ✓ Efficient for small-to-medium datasets
    ✓ Reduces hyperparameter tuning for classification layer
    ✓ Provides theoretical insights into feature space geometry
    ✓ Gradient-free classification preserves computational resources
    """)

# Display classifier info
st.info(
    f"🔬 **Model Type**: Hybrid PIL-Inspired Model  \n"
    f"**Training Method**: CNN + Moore-Penrose Pseudoinverse  \n"
    f"**Classification Layer**: Gradient-free (analytical learning)  \n"
    f"**Input**: Medical X-ray images (224×224×3)  \n"
    f"**Output**: Probability scores for Normal vs. Pneumonia"
)

# Sidebar configuration
with st.sidebar:
    uploaded_image = st.file_uploader("Upload X-ray / image", type=["jpg", "jpeg", "png"])
    st.markdown(f"**Model input:** {IN_H}×{IN_W}×{IN_C}")
    st.markdown(f"**Classes:** Normal, Pneumonia")
    st.markdown(f"**Feature dimension:** {w_classifier.shape[0] if w_classifier is not None else 'N/A'}")

# Main prediction pipeline
if uploaded_image and MODEL_LOADED:
    # Load and display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    batch = preprocess_image(image)
    
    # Make prediction using hybrid model
    probs = predict_hybrid(batch)

    # Display results
    st.subheader("Predicted Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal", f"{probs['Normal']:.1%}")
    with col2:
        st.metric("Pneumonia", f"{probs['Pneumonia']:.1%}")

    # Final prediction with adjusted threshold
    # Using optimal threshold 0.65 from threshold analysis
    DECISION_THRESHOLD = 0.65
    if probs['Pneumonia'] >= DECISION_THRESHOLD:
        predicted_class = "Pneumonia"
    else:
        predicted_class = "Normal"
    confidence = probs[predicted_class]
    
    # Calculate confidence and certainty
    raw_confidence = abs(probs['Pneumonia'] - 0.5) * 2  # 0 = uncertain, 1 = certain
    if raw_confidence < 0.2:
        certainty_level = "⚠️ LOW - Prediction uncertain"
        certainty_color = "warning"
    elif raw_confidence < 0.5:
        certainty_level = "🟡 MEDIUM - Use with caution"
        certainty_color = "warning"
    else:
        certainty_level = "✅ HIGH - Confident prediction"
        certainty_color = "success"

    st.subheader("Final Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")
    with col2:
        st.write(f"**Certainty:** {certainty_level}")
    st.caption(f"Decision threshold: {DECISION_THRESHOLD:.1%} | Confidence score: {raw_confidence:.2%}")
    
    # Display explanation
    with st.expander("🔬 How This Prediction Was Made (Technical Details)"):
        st.markdown("""
        ### Prediction Pipeline Breakdown
        
        #### Phase 1: Feature Extraction (CNN)
        Your X-ray image was passed through a trained Convolutional Neural Network:
        - **Conv Block 1**: 32 filters, 3×3 kernels → ReLU → MaxPooling
        - **Conv Block 2**: 64 filters, 3×3 kernels → ReLU → MaxPooling
        - **Conv Block 3**: 128 filters, 3×3 kernels → ReLU → MaxPooling
        - **Flattening**: Converts 2D feature maps to 1D feature vector
        
        **Output**: High-dimensional feature representation of the image
        
        #### Phase 2: Pseudoinverse Learning (PIL)
        The feature vector was multiplied by analytically computed weights:
        
        ```
        z = w^T × f + b
        where:
          w = weights (computed via Moore-Penrose pseudoinverse)
          f = feature vector from CNN
          b = bias term
        ```
        
        **Key innovation**: Weights were NOT learned via gradient descent!
        They were computed analytically as: **w = X^+ y**
        - X^+ is the Moore-Penrose pseudoinverse of feature matrix X
        - y is the label vector
        - This gives the minimum-norm least-squares optimal solution
        
        #### Phase 3: Probability Conversion
        The logit was converted to probability using sigmoid function:
        ```
        p = 1 / (1 + exp(-z))
        ```
        
        ### Why This Approach?
        ✓ **Efficiency**: Analytical solution faster than iterative SGD
        ✓ **Optimality**: Guaranteed minimum-norm least-squares solution
        ✓ **Theory**: Grounded in PIL/ELM/DAN literature
        ✓ **Hybrid**: Combines gradient-based (CNN) + gradient-free (classifier)
        
        ### Medical Imaging Interpretation
        - **Probability > 0.5**: Likely Pneumonia
        - **Probability < 0.5**: Likely Normal
        - **Confidence**: How strongly the model commits to its prediction
        
        ⚠️ **Disclaimer**: This is an educational demonstration and should
        not be used for actual medical diagnosis without professional review.
        """)
    
    st.divider()
    
    # Add model performance info
    st.warning("""
    ⚠️ **Model Limitations**:
    - **Test Accuracy**: ~79-80% (depends on image quality and presentation)
    - **False Positives**: Model may predict Pneumonia for Normal cases (~31% of Normal images)
    - **Best for screening**: Identifies most Pneumonia cases (97% sensitivity)
    - **Professional review required**: Always consult radiologists for diagnosis
    - **Image Quality**: Clear, well-positioned X-rays produce better predictions
    """)
    
    st.caption("🔐 Educational Demo — Not for clinical use without professional validation")

elif not MODEL_LOADED:
    st.error("❌ Model not loaded. Please train the model first.")

