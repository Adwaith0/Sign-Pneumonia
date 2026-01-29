"""
Evaluation Script: Comprehensive Hybrid Model Assessment

This script evaluates the hybrid PIL-inspired CNN model on training, validation,
and test datasets. It computes:

1. Classification Metrics (Accuracy, Precision, Recall, F1-Score)
2. Medical Imaging Metrics (Sensitivity, Specificity, ROC-AUC)
3. Confusion Matrix Analysis
4. Detailed Classification Reports
5. Threshold Optimization Analysis
6. Feature Space Statistics

USAGE:
    python evaluate_model.py

OUTPUT:
    - Console: Formatted metrics reports
    - JSON: Detailed metrics for all datasets
    - CSV: Threshold analysis across different decision thresholds
"""

import os
import numpy as np
import json
import csv
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation_metrics import HybridModelEvaluator

# =========================================================================
# CONFIGURATION
# =========================================================================
CNN_EXTRACTOR_PATH = "./model/cnn_feature_extractor.keras"
CLASSIFIER_WEIGHTS_PATH = "./model/pseudoinverse_classifier.pkl"
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
TEST_DIR = "./data/test"
OUTPUT_DIR = "./evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# MAIN EVALUATION PIPELINE
# =========================================================================
def main():
    print("=" * 80)
    print("HYBRID PIL-INSPIRED MODEL: COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    print("\n[STEP 1] Initializing evaluator...")
    try:
        evaluator = HybridModelEvaluator(CNN_EXTRACTOR_PATH, CLASSIFIER_WEIGHTS_PATH)
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return
    
    # Data generators
    print("\n[STEP 2] Setting up data generators...")
    datagen_no_augment = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen_no_augment.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )
    
    val_generator = datagen_no_augment.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )
    
    test_generator = datagen_no_augment.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )
    
    print(f"✓ Training samples: {len(train_generator) * 32} (approx)")
    print(f"✓ Validation samples: {len(val_generator) * 32} (approx)")
    print(f"✓ Test samples: {len(test_generator) * 32} (approx)")
    
    # =========================================================================
    # EVALUATE ON ALL DATASETS (default threshold: 0.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[STEP 3] EVALUATING ON TRAINING SET (Threshold: 0.50)")
    print("=" * 80)
    train_metrics, train_labels, train_preds, train_probs = evaluator.evaluate_dataset(
        train_generator, threshold=0.5, verbose=True
    )
    evaluator.print_metrics_summary(train_metrics)
    
    print("\n" + "=" * 80)
    print("[STEP 4] EVALUATING ON VALIDATION SET (Threshold: 0.50)")
    print("=" * 80)
    val_metrics, val_labels, val_preds, val_probs = evaluator.evaluate_dataset(
        val_generator, threshold=0.5, verbose=True
    )
    evaluator.print_metrics_summary(val_metrics)
    
    print("\n" + "=" * 80)
    print("[STEP 5] EVALUATING ON TEST SET (Threshold: 0.50)")
    print("=" * 80)
    test_metrics, test_labels, test_preds, test_probs = evaluator.evaluate_dataset(
        test_generator, threshold=0.5, verbose=True
    )
    evaluator.print_metrics_summary(test_metrics)
    
    # =========================================================================
    # THRESHOLD OPTIMIZATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[STEP 6] THRESHOLD OPTIMIZATION ANALYSIS (on validation set)")
    print("=" * 80)
    print("\nTesting different decision thresholds to find optimal balance...")
    
    threshold_analysis = []
    thresholds_to_test = np.linspace(0.3, 0.8, 11)
    
    for threshold in thresholds_to_test:
        # Re-evaluate with different threshold
        preds_thresh = (val_probs >= threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics_thresh = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(val_labels, preds_thresh)),
            "precision": float(precision_score(val_labels, preds_thresh, zero_division=0)),
            "recall": float(recall_score(val_labels, preds_thresh, zero_division=0)),
            "f1_score": float(f1_score(val_labels, preds_thresh, zero_division=0)),
        }
        threshold_analysis.append(metrics_thresh)
        
        print(f"\n  Threshold: {threshold:.2f}")
        print(f"    Accuracy:  {metrics_thresh['accuracy']:.4f}")
        print(f"    Precision: {metrics_thresh['precision']:.4f}")
        print(f"    Recall:    {metrics_thresh['recall']:.4f}")
        print(f"    F1-Score:  {metrics_thresh['f1_score']:.4f}")
    
    # Find optimal threshold (F1-score)
    best_threshold = max(threshold_analysis, key=lambda x: x["f1_score"])
    print(f"\n✓ Best threshold (by F1-score): {best_threshold['threshold']:.2f}")
    print(f"  F1-Score: {best_threshold['f1_score']:.4f}")
    
    # =========================================================================
    # RE-EVALUATE TEST SET WITH OPTIMAL THRESHOLD
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"[STEP 7] TEST SET EVALUATION (Optimal Threshold: {best_threshold['threshold']:.2f})")
    print("=" * 80)
    
    test_metrics_optimal, _, test_preds_optimal, _ = evaluator.evaluate_dataset(
        test_generator, threshold=best_threshold['threshold'], verbose=False
    )
    evaluator.print_metrics_summary(test_metrics_optimal)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[STEP 8] SAVING RESULTS")
    print("=" * 80)
    
    # Create results dictionary
    all_results = {
        "model_info": {
            "framework": "Hybrid PIL-Inspired Model",
            "components": {
                "feature_extractor": "CNN (Conv2D, ReLU, MaxPooling)",
                "classifier": "Linear layer with Moore-Penrose pseudoinverse weights",
                "training_method_cnn": "Standard backpropagation",
                "training_method_classifier": "Gradient-free (analytical)",
            }
        },
        "evaluation_summary": {
            "default_threshold": 0.5,
            "optimal_threshold": best_threshold['threshold'],
            "datasets_evaluated": ["train", "val", "test"]
        },
        "training_set_metrics": train_metrics,
        "validation_set_metrics": val_metrics,
        "test_set_metrics": test_metrics,
        "test_set_metrics_optimal_threshold": test_metrics_optimal,
        "threshold_analysis": threshold_analysis
    }
    
    # Save comprehensive results
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    evaluator.save_metrics_to_json(all_results, results_path)
    
    # Save threshold analysis as CSV
    csv_path = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=threshold_analysis[0].keys())
        writer.writeheader()
        writer.writerows(threshold_analysis)
    print(f"✅ Threshold analysis saved to {csv_path}")
    
    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    
    print("\n📊 PERFORMANCE ACROSS DATASETS (Threshold: 0.50):")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    print(f"{'Training':<15} {train_metrics['accuracy']:<12.4f} {train_metrics['precision']:<12.4f} {train_metrics['recall']:<12.4f} {train_metrics['f1_score']:<12.4f}")
    print(f"{'Validation':<15} {val_metrics['accuracy']:<12.4f} {val_metrics['precision']:<12.4f} {val_metrics['recall']:<12.4f} {val_metrics['f1_score']:<12.4f}")
    print(f"{'Test':<15} {test_metrics['accuracy']:<12.4f} {test_metrics['precision']:<12.4f} {test_metrics['recall']:<12.4f} {test_metrics['f1_score']:<12.4f}")
    
    print(f"\n📊 MEDICAL IMAGING METRICS (Test Set, Threshold: 0.50):")
    print("-" * 80)
    print(f"Sensitivity (Recall):     {test_metrics['sensitivity']:.4f}")
    print(f"Specificity:              {test_metrics['specificity']:.4f}")
    print(f"ROC-AUC:                  {test_metrics['roc_auc']:.4f}")
    
    print(f"\n🎯 OPTIMAL THRESHOLD PERFORMANCE (Test Set, Threshold: {best_threshold['threshold']:.2f}):")
    print("-" * 80)
    print(f"Accuracy:   {test_metrics_optimal['accuracy']:.4f}")
    print(f"Precision:  {test_metrics_optimal['precision']:.4f}")
    print(f"Recall:     {test_metrics_optimal['recall']:.4f}")
    print(f"F1-Score:   {test_metrics_optimal['f1_score']:.4f}")
    
    print(f"\n📁 Output files:")
    print(f"  ✓ {results_path}")
    print(f"  ✓ {csv_path}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
