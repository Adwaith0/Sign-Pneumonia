"""
Hybrid PIL-Inspired Model: CNN Feature Extractor + Pseudoinverse Learning

RESEARCH NOVELTY:
This module implements a hybrid block-wise analytical learning framework combining:

1. Convolutional Neural Network (CNN) as a feature extractor
   - Standard backpropagation training on Conv2D, ReLU, MaxPooling layers
   - Learns discriminative feature representations

2. Moore–Penrose Pseudoinverse Analytical Learning (PIL-inspired)
   - Gradient-free training for final classification layer
   - Analytical closed-form solution: w = X^+ y (where X^+ is pseudoinverse)
   - Avoids iterative gradient descent for the linear classifier

FRAMEWORK PHILOSOPHY:
- Block-wise learning: Different blocks use different learning paradigms
- Conv blocks: Standard backpropagation (gradient-based)
- Classification block: Analytical pseudoinverse (gradient-free)
- Not end-to-end gradient-free, but selective/hybrid approach
- Inspired by Extreme Learning Machines (ELM), Deep Adaptive Networks (DAN)

WORKFLOW:
1. Train CNN convolutional layers via standard backpropagation
2. Extract feature vectors from trained CNN
3. Compute optimal classification weights using Moore–Penrose pseudoinverse
4. Inference: CNN feature extraction → Linear classification (no softmax)
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
import json

# Dataset paths
train_dir = "./data/train"
val_dir   = "./data/val"

print("=" * 70)
print("CNN Feature Extractor + Pseudoinverse Learning")
print("=" * 70)

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
print("=" * 80)
print("HYBRID PIL-INSPIRED MODEL: CNN Feature Extraction + Pseudoinverse Learning")
print("=" * 80)

# ============================================================================
# STEP 1: Create CNN Feature Extractor (no Dense output layer)
# ============================================================================
print("\n[STEP 1] Building CNN feature extractor...")
print("         Conv2D, ReLU, MaxPooling layers only (no Dense layer)")
print("         These layers will be trained via standard backpropagation")

# Image preprocessing with enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # Brightness augmentation for X-rays
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# CNN Architecture: Convolutional layers with improved regularization
# (No Dense, Dropout, or Sigmoid layers for pseudoinverse approach)
# Enhanced with batch normalization and L2 regularization for better generalization
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3), kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten()
    # NOTE: No Dense, Dropout, or output layer
    # The Flatten output will be our feature vector
])

print(f"CNN feature extractor output shape: {cnn_model.output_shape}")

# ============================================================================
# STEP 2: Create temporary model with Dense layer for CNN pretraining
# ============================================================================
print("\n[STEP 2] CNN Pretraining via Standard Backpropagation")
print("         Training Conv2D, ReLU, MaxPooling layers with auxiliary task")

# We add a temporary dense layer just for training the CNN features
temp_model = Sequential([
    *cnn_model.layers,
    Dense(1, activation="sigmoid")  # Binary classification head (temporary)
])

temp_model.compile(
    optimizer=Adam(learning_rate=5e-4, decay=1e-6),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Pretrain the CNN on the classification task
# Increased epochs to allow better convergence with regularization
history = temp_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=40,  # Increased from 25 to 40 epochs
    validation_data=val_generator,
    validation_steps=len(val_generator),
    class_weight={0: 2.89, 1: 1.0}  # Apply class weights to handle imbalance
)

print("✅ CNN pretraining complete.")

# ============================================================================
# STEP 3: Extract feature vectors using trained CNN
# ============================================================================
print("\n[STEP 3] Feature Extraction Phase")
print("         Extracting feature vectors from trained CNN (gradient-free)")
print("         CNN weights are frozen; only used for feature transformation")

# Reset data generators (without shuffling for consistency)
train_datagen_no_augment = ImageDataGenerator(rescale=1./255)

train_generator_features = train_datagen_no_augment.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False  # Important: maintain order
)

# Extract all feature vectors from training set
all_features = []
all_labels = []

num_batches = len(train_generator_features)
for batch_idx in range(num_batches):
    images_batch, labels_batch = train_generator_features[batch_idx]
    
    # Extract features using CNN model (without the temporary dense layer)
    features_batch = cnn_model.predict(images_batch, verbose=0)
    
    all_features.append(features_batch)
    all_labels.append(labels_batch)
    
    if (batch_idx + 1) % 5 == 0:
        print(f"  Processed {batch_idx + 1}/{num_batches} batches")

# Concatenate all features and labels
X_features = np.vstack(all_features)  # Shape: (n_samples, feature_dim)
y_labels = np.hstack(all_labels)      # Shape: (n_samples,)

print(f"Extracted feature matrix shape: {X_features.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"Feature dimension: {X_features.shape[1]}")

# ============================================================================
# STEP 4: Compute classification weights using Moore–Penrose Pseudoinverse
# ============================================================================
print("\n[STEP 4] Pseudoinverse Learning (PIL) - Analytical Closed-Form Solution")
print("         Computing classification weights via Moore-Penrose pseudoinverse")

# Reshape labels for matrix operations
# For binary classification: y ∈ {0, 1} → Y ∈ R^(n × 1)
Y = y_labels.reshape(-1, 1)

# ┌─────────────────────────────────────────────────────────────────┐
# │           PSEUDOINVERSE LEARNING (PIL) - CORE ALGORITHM        │
# └─────────────────────────────────────────────────────────────────┘
#
# RESEARCH NOVELTY:
# Instead of iterative gradient descent (standard backpropagation),
# we compute optimal classification weights analytically.
#
# MATHEMATICAL FORMULATION:
#   Minimize: ||Xw - y||_2^2  (least squares objective)
#   Solution: w = X^+ y       (where X^+ is Moore-Penrose pseudoinverse)
#
# THEORETICAL BENEFITS:
#   - Closed-form solution (no hyperparameter tuning needed)
#   - Guaranteed to minimize least-squares error in feature space
#   - Gradient-free learning for classification layer
#   - Computationally efficient for small-to-medium datasets
#
# GRADIENT-FREE PROPERTY:
#   - No backpropagation through this layer
#   - No learning rate, momentum, or optimizer hyperparameters
#   - Purely analytical/deterministic computation
#

print("         Mathematical formulation:")
print("           minimize ||Xw - y||_2^2  (least squares)")
print("           solution: w = X^+ y  (pseudoinverse)")
print("         ")
print("         Computing X^+ using np.linalg.pinv (SVD-based)...")

X_pinv = np.linalg.pinv(X_features)  # Compute Moore-Penrose pseudoinverse via SVD
w_classifier = X_pinv @ Y             # Compute optimal weights analytically

print(f"✓ Pseudoinverse computed successfully")
print(f"  - Feature matrix X shape: {X_features.shape}")
print(f"  - Pseudoinverse X^+ shape: {X_pinv.shape}")
print(f"  - Classifier weights shape: {w_classifier.shape}")
print(f"  - Weight statistics:")
print(f"      Mean: {w_classifier.mean():.6f}")
print(f"      Std:  {w_classifier.std():.6f}")
print(f"      Min:  {w_classifier.min():.6f}")
print(f"      Max:  {w_classifier.max():.6f}")

# ============================================================================
# STEP 5: Save models and weights
# ============================================================================
print("\n[STEP 5] Model and Weights Persistence")
print("         Saving CNN feature extractor and pseudoinverse classifier weights")

os.makedirs("model", exist_ok=True)

# Save the CNN feature extractor (without the temporary dense layer)
cnn_model.save("model/cnn_feature_extractor.keras")
print("✅ Saved: model/cnn_feature_extractor.keras")

# Save the pseudoinverse classifier weights as pickle file
bias = np.mean(Y) - np.mean(X_features @ w_classifier)  # Compute bias

classifier_weights = {
    "weights": w_classifier,              # Analytically computed weights (shape: feature_dim, 1)
    "bias": bias,                         # Computed bias term
    "feature_dim": X_features.shape[1],   # Feature dimensionality
    "training_method": "Moore-Penrose Pseudoinverse (PIL)",
    "gradient_free": True,                # Indicates no gradient descent used
    "learning_framework": "Hybrid Block-wise Analytical Learning",
    "description": "Final classification layer weights computed via Moore-Penrose pseudoinverse"
}

with open("model/pseudoinverse_classifier.pkl", "wb") as f:
    pickle.dump(classifier_weights, f)
print("✅ Saved: model/pseudoinverse_classifier.pkl")
print("   - Weights (shape): (feature_dim, 1)")
print("   - Bias: scalar")
print("   - Training: Gradient-free (analytical)")

# Also save the full temporary model for reference
temp_model.save("model/medical_cnn.keras")
print("✅ Saved: model/medical_cnn.keras (reference)")

# Save training metadata
metadata = {
    "approach": "Hybrid PIL-Inspired Model",
    "framework": "Block-wise Analytical Learning",
    "cnn_training": "Standard backpropagation (Conv2D, ReLU, MaxPooling)",
    "classification_training": "Gradient-free pseudoinverse learning",
    "feature_extractor": "model/cnn_feature_extractor.keras",
    "classifier_weights": "model/pseudoinverse_classifier.pkl",
    "full_model_reference": "model/medical_cnn.keras",
    "components": {
        "conv_block_1": "Conv2D(32) -> ReLU -> MaxPooling2D",
        "conv_block_2": "Conv2D(64) -> ReLU -> MaxPooling2D",
        "conv_block_3": "Conv2D(128) -> ReLU -> MaxPooling2D",
        "feature_extractor": "Flatten() - outputs feature vector",
        "classifier": "Linear transformation (no activation)"
    }
}

with open("model/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: model/training_metadata.json")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE - Hybrid PIL-Inspired Model Ready!")
print("=" * 80)
print("\n📋 SUMMARY:")
print("-" * 80)
print("LEARNING FRAMEWORK: Hybrid Block-wise Analytical Learning")
print("")
print("COMPONENT 1: CNN Feature Extractor")
print("  ├─ Layers: Conv2D → ReLU → MaxPooling2D (3 blocks)")
print("  ├─ Training: Standard backpropagation")
print("  ├─ Gradient-based: YES")
print("  └─ Output: Feature vector")
print("")
print("COMPONENT 2: Linear Classifier (PIL)")
print("  ├─ Architecture: Dense layer (no activation)")
print("  ├─ Training: Moore-Penrose pseudoinverse")
print("  ├─ Gradient-based: NO (analytical solution)")
print("  ├─ Method: w = X^+ y (closed-form)")
print("  └─ Property: Minimum-norm least-squares solution")
print("")
print("Generated files:")
print("  ✓ model/cnn_feature_extractor.keras")
print("    └─ Trained CNN feature extraction component")
print("  ✓ model/pseudoinverse_classifier.pkl")
print("    └─ Analytically computed classification weights")
print("  ✓ model/medical_cnn.keras")
print("    └─ Reference full model (for compatibility)")
print("  ✓ model/training_metadata.json")
print("    └─ Training framework and architecture documentation")
print("")
print("🚀 Next step: Run evaluation script to assess hybrid model performance")
print("   - Compute confusion matrix, accuracy, precision, recall, F1-score")
print("   - Analyze feature space separability")
print("=" * 80)
