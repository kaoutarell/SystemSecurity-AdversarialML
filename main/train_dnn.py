#!/usr/bin/env python3
"""
Dense Neural Network (DNN) IDS Training Script

Simplified training script that uses:
- models/dnn.py: Model architecture
- train.py: Common training pipeline
- dataset.py: Data preprocessing
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
    parser = argparse.ArgumentParser(description='Train Dense NN IDS on NSL-KDD Dataset')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data')
    parser.add_argument('--use_separate_test', action='store_true', default=False,
                        help='Use separate test file for evaluation')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--l1', type=float, default=1e-5,
                        help='L1 regularization coefficient (default: 1e-5)')
    parser.add_argument('--l2', type=float, default=1e-4,
                        help='L2 regularization coefficient (default: 1e-4)')
    parser.add_argument('--use_batchnorm', action='store_true', default=False,
                        help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=0.006,
                        help='Learning rate (default: 0.006)')
    
    parser.add_argument('--output_dir', type=str, default='results/runs',
                        help='Output directory (default: results/runs)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=9281,
                        help='Random seed for reproducibility (default: 9281)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
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
    print("Dense Neural Network Intrusion Detection System")
    print("="*80)
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("Data Preparation")
    print("="*80)
    
    train_df = load_data(args.train_path)
    X_train_full, y_train_full, scaler, feature_columns = preprocess_nsl_kdd(
        train_df, use_statistical_filter=True
    )
    
    # Load test data if provided
    if args.use_separate_test and args.test_path:
        test_df = load_data(args.test_path)
        X_test, y_test, _, _ = preprocess_nsl_kdd(
            test_df,
            scaler=scaler,
            feature_columns=feature_columns,
            use_statistical_filter=True
        )
        print(f"Test set: {X_test.shape}")
    else:
        # Will use validation split as test set
        X_test, y_test = None, None
        print("Will use validation split as test set")
    
    # Build model
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    
    input_dim = X_train_full.shape[1]
    print(f"  Input dimension: {input_dim}")
    
    model = create_dnn_model(
        input_dim=input_dim,
        dropout_rate=args.dropout,
        l1=args.l1,
        l2=args.l2,
        use_batchnorm=args.use_batchnorm,
        learning_rate=args.learning_rate
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    
    # Prepare artifacts for saving
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    # If no separate test set, use validation split
    if X_test is None:
        from sklearn.model_selection import train_test_split
        _, X_test_split, _, y_test_split = train_test_split(
            X_train_full, y_train_full,
            test_size=args.validation_split,
            random_state=921,
            stratify=y_train_full
        )
        X_test, y_test = X_test_split, y_test_split
    
    # Run training pipeline
    run_dir, history, metrics = run_training_pipeline(
        model=model,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        args=args,
        artifacts=artifacts,
        model_name='Dense NN IDS',
        run_prefix='run_ids'
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}\n")


if __name__ == '__main__':
    main()
