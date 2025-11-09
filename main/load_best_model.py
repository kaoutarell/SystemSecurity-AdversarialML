#!/usr/bin/env python3
"""
Load and test the best SAAE-DNN model (run_saae_011)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_data

# Define AttentionLayer for model loading
class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention Mechanism Layer
    Based on SAAE-DNN paper
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.Wa = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.ba = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # M = tanh(Wa * x' + ba)
        M = tf.nn.tanh(tf.matmul(inputs, self.Wa) + self.ba)
        # α_i = softmax(M_i)
        alpha = tf.nn.softmax(M, axis=-1)
        # v = x' * α^T (element-wise multiplication)
        v = inputs * alpha
        return v
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

# Model path
BEST_MODEL_PATH = Path(__file__).parent / 'results' / 'saae_runs' / 'run_saae_011' / 'best_model.keras'
METRICS_PATH = Path(__file__).parent / 'results' / 'saae_runs' / 'run_saae_011' / 'metrics_kddtest+.json'

def main():
    print("=" * 80)
    print("LOADING BEST MODEL - run_saae_011 (SAAE-DNN)")
    print("=" * 80)
    
    # Check if model exists
    if not BEST_MODEL_PATH.exists():
        print(f"❌ Model not found at: {BEST_MODEL_PATH}")
        return
    
    print(f"\n✓ Model found at: {BEST_MODEL_PATH}")
    
    # Load the model
    print("\n📦 Loading model...")
    try:
        model = tf.keras.models.load_model(
            BEST_MODEL_PATH,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Display model summary
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()
    
    # Load and display metrics
    if METRICS_PATH.exists():
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE (KDDTest+)")
        print("=" * 80)
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print("=" * 80)
    
    # Load test data
    print("\n📊 Loading NSL-KDD test data...")
    test_path = Path(__file__).parent.parent / 'nsl-kdd' / 'KDDTest+.txt'
    
    if test_path.exists():
        # Use same preprocessing as SAAE-DNN training
        from train_saae_dnn import load_nsl_kdd_data, preprocess_nsl_kdd
        
        try:
            # Load and preprocess data
            train_df = load_nsl_kdd_data(str(Path(__file__).parent.parent / 'nsl-kdd' / 'KDDTrain+.txt'), binary=True)
            test_df = load_nsl_kdd_data(str(test_path), binary=True)
            
            # Preprocess with same pipeline
            X_train, y_train, scaler, feature_cols = preprocess_nsl_kdd(
                train_df,
                test_df=None,
                use_statistical_filter=True
            )
            X_test, y_test, _, _ = preprocess_nsl_kdd(
                test_df,
                test_df=None,
                use_statistical_filter=True,
                scaler=scaler,
                feature_columns=feature_cols
            )
            
            print(f"✓ Test data loaded: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
        
        # Make predictions
        print("\n🔮 Making predictions on test set...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate some quick stats
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n" + "=" * 80)
        print("VERIFICATION - MODEL PERFORMANCE")
        print("=" * 80)
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:>6}  FP: {cm[0,1]:>6}")
        print(f"  FN: {cm[1,0]:>6}  TP: {cm[1,1]:>6}")
        print("=" * 80)
        
        # Show some example predictions
        print("\n📋 Sample Predictions (first 10 test samples):")
        print("-" * 80)
        print(f"{'Index':<8} {'True Label':<12} {'Predicted':<12} {'Probability':<15} {'Correct':<10}")
        print("-" * 80)
        for i in range(min(10, len(y_test))):
            label_true = "Attack" if y_test[i] == 1 else "Normal"
            label_pred = "Attack" if y_pred[i] == 1 else "Normal"
            prob = y_pred_proba[i][0] if len(y_pred_proba.shape) > 1 else y_pred_proba[i]
            correct = "✓" if y_test[i] == y_pred[i] else "✗"
            print(f"{i:<8} {label_true:<12} {label_pred:<12} {prob:<15.4f} {correct:<10}")
        print("-" * 80)
        except Exception as e:
            print(f"⚠️  Error loading test data: {e}")
            print(f"   Trying simpler approach...")
            # Fallback: just show model is loaded
            X_test, y_test = None, None
    else:
        print(f"⚠️  Test data not found at: {test_path}")
    
    print("\n✓ Best model loaded and ready for use!")
    print(f"   Model object: 'model' variable")
    print(f"   Use model.predict(X) to make predictions")
    
    return model

if __name__ == '__main__':
    model = main()
