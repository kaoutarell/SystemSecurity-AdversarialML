#!/usr/bin/env python3
"""
CNN with Attention IDS Training Script

Simplified training script that uses:
- models/cnn_attention.py: Model architecture with ECA blocks
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

from models.cnn_attention import create_cnn_attention_model
from dataset import load_data, preprocess_nsl_kdd
from train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN + ECA Attention model for IDS'
    )
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data file')
    parser.add_argument('--use_separate_test', action='store_true',
                       help='Use separate test file for evaluation')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Model hyperparameters
    parser.add_argument('--image_size', type=int, default=12,
                       help='Image size for CNN (12x12=144 features, default: 12)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate (default: 0.4)')
    parser.add_argument('--l1', type=float, default=1e-5,
                       help='L1 regularization (default: 1e-5)')
    parser.add_argument('--l2', type=float, default=5e-4,
                       help='L2 regularization (default: 5e-4)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size (default: 2048, CRITICAL for CNN!)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/runs',
                       help='Output directory for results (default: results/runs)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=9281,
                       help='Random seed for reproducibility (default: 9281)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("\n" + "="*80)
    print("CNN + ECA Attention Intrusion Detection System")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print(f"✓ TensorFlow version: {tf.__version__}\n")
    
    # Load and preprocess data
    print("="*80)
    print("Data Preparation")
    print("="*80)
    
    train_df = load_data(args.train_path)
    X_train_full, y_train_full, scaler, feature_columns = preprocess_nsl_kdd(
        train_df, image_size=args.image_size, use_statistical_filter=True
    )
    
    # Load test data if provided
    if args.use_separate_test and args.test_path:
        test_df = load_data(args.test_path)
        X_test, y_test, _, _ = preprocess_nsl_kdd(
            test_df,
            scaler=scaler,
            feature_columns=feature_columns,
            image_size=args.image_size,
            use_statistical_filter=True
        )
        print(f"Test set: {X_test.shape}")
    else:
        # Will use validation split as test set
        X_test, y_test = None, None
        print("Will use validation split as test set")
    
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    
    print(f"  Input shape: ({args.image_size}, {args.image_size}, 1)")
    
    model = create_cnn_attention_model(
        image_size=args.image_size,
        dropout=args.dropout,
        l1=args.l1,
        l2=args.l2,
        learning_rate=args.learning_rate
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    
    # Prepare artifacts for saving
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'image_size': args.image_size
    }
    
    # If no separate test set, use validation split
    if X_test is None:
        from sklearn.model_selection import train_test_split
        _, X_test_split, _, y_test_split = train_test_split(
            X_train_full, y_train_full,
            test_size=args.validation_split,
            random_state=9281,
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
        model_name='CNN + ECA Attention',
        run_prefix='run_cnn'
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}\n")


if __name__ == "__main__":
    main()
