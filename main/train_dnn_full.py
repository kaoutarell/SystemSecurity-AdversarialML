#!/usr/bin/env python3
"""
DNN IDS Training with Separate Test Set

This script trains a DNN model using:
- Training data: KDDTrain+.txt (split into train/validation)
- Test data: KDDTest+.txt (separate, unseen test set)

Usage:
    python train_dnn_with_test.py --train_path ../nsl-kdd/KDDTrain+.txt --test_path ../nsl-kdd/KDDTest+.txt
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.dnn import create_dnn_model
from dataset import load_data, preprocess_nsl_kdd
from train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Train DNN IDS with separate test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments (required)
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data (e.g., KDDTrain+.txt)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data (e.g., KDDTest+.txt)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split from training data')
    
    # Model hyperparameters
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--l1', type=float, default=1e-5,
                        help='L1 regularization coefficient')
    parser.add_argument('--l2', type=float, default=1e-4,
                        help='L2 regularization coefficient')
    parser.add_argument('--use_batchnorm', action='store_true', default=False,
                        help='Use batch normalization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.006,
                        help='Learning rate')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/runs',
                        help='Output directory for results')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=9281,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print(f"✓ TensorFlow version: {tf.__version__}\n")
    
    print("="*80)
    print("DNN IDS Training with Separate Test Set")
    print("="*80)
    print(f"Training file: {args.train_path}")
    print(f"Test file: {args.test_path}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Load and preprocess training data
    print("\n" + "="*80)
    print("Loading Training Data")
    print("="*80)
    
    train_df = load_data(args.train_path)
    X_train_full, y_train_full, scaler, feature_columns = preprocess_nsl_kdd(
        train_df, use_statistical_filter=True
    )
    
    print(f"Training data: {X_train_full.shape}")
    print(f"  Will split into: {(1-args.validation_split)*100:.0f}% train, {args.validation_split*100:.0f}% validation")
    
    # Load and preprocess test data
    print("\n" + "="*80)
    print("Loading Test Data")
    print("="*80)
    
    test_df = load_data(args.test_path)
    X_test, y_test, _, _ = preprocess_nsl_kdd(
        test_df,
        scaler=scaler,
        feature_columns=feature_columns,
        use_statistical_filter=True
    )
    
    print(f"Test data: {X_test.shape}")
    
    # Build model
    print("\n" + "="*80)
    print("Building Model")
    print("="*80)
    
    input_dim = X_train_full.shape[1]
    print(f"  Input features: {input_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  L1 regularization: {args.l1}")
    print(f"  L2 regularization: {args.l2}")
    print(f"  Batch normalization: {args.use_batchnorm}")
    
    model = create_dnn_model(
        input_dim=input_dim,
        dropout_rate=args.dropout,
        l1=args.l1,
        l2=args.l2,
        use_batchnorm=args.use_batchnorm,
        learning_rate=args.learning_rate
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    
    # Prepare artifacts
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'config': {
            'dropout': args.dropout,
            'l1': args.l1,
            'l2': args.l2,
            'use_batchnorm': args.use_batchnorm,
            'learning_rate': args.learning_rate,
            'seed': args.seed
        }
    }
    
    # Run training pipeline
    run_dir, history, metrics = run_training_pipeline(
        model=model,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        args=args,
        artifacts=artifacts,
        model_name='DNN IDS (Separate Test)',
        run_prefix='run_dnn_test'
    )
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Test Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Test Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
