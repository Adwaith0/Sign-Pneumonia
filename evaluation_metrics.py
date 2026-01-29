"""
Evaluation Metrics Module: Comprehensive Hybrid Model Assessment

This module provides evaluation utilities for the hybrid PIL-inspired model:
- Confusion matrix
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Feature space visualization
- Detailed classification reports

FEATURES:
1. Binary classification metrics for medical imaging diagnosis
2. Support for batch predictions and aggregated metrics
3. Visualization-friendly output formats
4. JSON export for downstream analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)


class HybridModelEvaluator:
    """
    Evaluation class for Hybrid PIL-Inspired Model.
    
    Combines CNN feature extraction with pseudoinverse classifier
    for comprehensive performance assessment.
    """
    
    def __init__(self, cnn_extractor_path: str, classifier_weights_path: str):
        """
        Initialize evaluator with model components.
        
        Args:
            cnn_extractor_path: Path to saved CNN feature extractor (.keras)
            classifier_weights_path: Path to pseudoinverse classifier weights (.pkl)
        """
        print("🔍 Initializing HybridModelEvaluator...")
        
        # Load CNN feature extractor
        try:
            self.cnn_extractor = keras.models.load_model(cnn_extractor_path, compile=False)
            print(f"✓ Loaded CNN feature extractor from {cnn_extractor_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN extractor: {e}")
        
        # Load pseudoinverse classifier weights
        try:
            with open(classifier_weights_path, "rb") as f:
                self.classifier_dict = pickle.load(f)
            self.weights = self.classifier_dict["weights"]
            self.bias = self.classifier_dict["bias"]
            print(f"✓ Loaded pseudoinverse classifier from {classifier_weights_path}")
            print(f"  - Weight shape: {self.weights.shape}")
            print(f"  - Bias: {self.bias:.6f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier weights: {e}")
        
        self.input_shape = self.cnn_extractor.input_shape[1:]  # (H, W, C)
        print(f"✓ Model input shape: {self.input_shape}")
    
    def extract_features(self, image_batch: np.ndarray) -> np.ndarray:
        """
        Extract feature vectors from image batch using CNN.
        
        Args:
            image_batch: Array of shape (batch_size, H, W, C) with values in [0, 1]
            
        Returns:
            Feature vectors of shape (batch_size, feature_dim)
        """
        features = self.cnn_extractor.predict(image_batch, verbose=0)
        return features
    
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        """
        Compute logits (raw scores) using pseudoinverse classifier.
        
        IMPORTANT: This is gradient-free prediction!
        No backpropagation or sigmoid is applied here.
        
        Args:
            features: Feature vectors of shape (batch_size, feature_dim)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        logits = features @ self.weights + self.bias
        return logits
    
    def predict_probabilities(self, image_batch: np.ndarray) -> np.ndarray:
        """
        Full hybrid prediction pipeline:
        1. Extract features using CNN
        2. Compute logits using pseudoinverse classifier
        3. Apply sigmoid to get probabilities
        
        Args:
            image_batch: Array of shape (batch_size, H, W, C)
            
        Returns:
            Probabilities of shape (batch_size, 2) for [Normal, Pneumonia]
        """
        # Step 1: Feature extraction (CNN)
        features = self.extract_features(image_batch)
        
        # Step 2: Linear classification (PIL - gradient-free)
        logits = self.predict_logits(features)
        
        # Step 3: Convert to probabilities
        probs_pneumonia = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
        probs_normal = 1.0 - probs_pneumonia
        
        # Stack into shape (batch_size, 2)
        probabilities = np.hstack([probs_normal, probs_pneumonia])
        return probabilities
    
    def predict_classes(self, image_batch: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels with adjustable threshold.
        
        Args:
            image_batch: Array of shape (batch_size, H, W, C)
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Class predictions: 0 (Normal) or 1 (Pneumonia)
        """
        probs = self.predict_probabilities(image_batch)
        predictions = (probs[:, 1] >= threshold).astype(int)
        return predictions
    
    def evaluate_batch(self, image_batch: np.ndarray, true_labels: np.ndarray, 
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model on a batch of images.
        
        Args:
            image_batch: Array of shape (batch_size, H, W, C)
            true_labels: True labels: 0 (Normal) or 1 (Pneumonia)
            threshold: Decision threshold
            
        Returns:
            Dictionary with metrics
        """
        # Make predictions
        probs = self.predict_probabilities(image_batch)
        preds = self.predict_classes(image_batch, threshold)
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, preds),
            "precision": precision_score(true_labels, preds, zero_division=0),
            "recall": recall_score(true_labels, preds, zero_division=0),
            "f1_score": f1_score(true_labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(true_labels, probs[:, 1]),
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, preds)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
        
        # Additional statistics
        metrics["specificity"] = metrics["tn"] / (metrics["tn"] + metrics["fp"]) if (metrics["tn"] + metrics["fp"]) > 0 else 0.0
        metrics["sensitivity"] = metrics["recall"]  # Same as recall for binary classification
        
        metrics["threshold"] = threshold
        metrics["batch_size"] = len(true_labels)
        
        return metrics
    
    def evaluate_dataset(self, image_generator, num_batches: int = None, 
                        threshold: float = 0.5, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset using a Keras data generator.
        
        Args:
            image_generator: Keras ImageDataGenerator with flow_from_directory
            num_batches: Number of batches to evaluate (None = all)
            threshold: Decision threshold
            verbose: Print progress
            
        Returns:
            Aggregated evaluation metrics
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        num_batches_available = len(image_generator)
        if num_batches is None:
            num_batches = num_batches_available
        else:
            num_batches = min(num_batches, num_batches_available)
        
        if verbose:
            print(f"\n📊 Evaluating model on {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            images, labels = image_generator[batch_idx]
            
            # Predict
            probs = self.predict_probabilities(images)
            preds = self.predict_classes(images, threshold)
            
            all_probs.append(probs[:, 1])  # Pneumonia probabilities
            all_preds.append(preds)
            all_labels.append(labels)
            
            if verbose and (batch_idx + 1) % 5 == 0:
                print(f"  ✓ Processed {batch_idx + 1}/{num_batches} batches")
        
        # Aggregate results
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute overall metrics
        metrics = {
            "dataset_size": len(all_labels),
            "threshold": threshold,
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1_score": float(f1_score(all_labels, all_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
        
        # Sensitivity and specificity
        metrics["sensitivity"] = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0.0
        metrics["specificity"] = metrics["tn"] / (metrics["tn"] + metrics["fp"]) if (metrics["tn"] + metrics["fp"]) > 0 else 0.0
        
        # Classification report
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        if verbose:
            print(f"\n✅ Evaluation complete!")
        
        return metrics, all_labels, all_preds, all_probs
    
    def print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Print formatted metrics summary."""
        print("\n" + "=" * 70)
        print("HYBRID MODEL EVALUATION METRICS")
        print("=" * 70)
        print(f"\nDataset Size: {metrics.get('dataset_size', 'N/A')} samples")
        print(f"Decision Threshold: {metrics['threshold']:.2f}")
        print("\n📊 CLASSIFICATION METRICS:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        print("\n📈 MEDICAL IMAGING SPECIFIC:")
        print(f"  Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (True Negative Rate): {metrics['specificity']:.4f}")
        print("\n🔲 CONFUSION MATRIX:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  [[TN={metrics['tn']:5d}  FP={metrics['fp']:5d}]")
        print(f"   [FN={metrics['fn']:5d}  TP={metrics['tp']:5d}]]")
        print("\n" + "=" * 70)
    
    def save_metrics_to_json(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Save metrics to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics_serializable = convert_types(metrics)
        
        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"✅ Metrics saved to {output_path}")


def compute_roc_metrics(true_labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC curve metrics.
    
    Args:
        true_labels: True binary labels
        probabilities: Predicted probabilities for positive class
        
    Returns:
        Dictionary with ROC metrics and curves
    """
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc)
    }
