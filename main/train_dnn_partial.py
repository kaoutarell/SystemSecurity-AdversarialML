#!/usr/bin/env python3
"""
DNN IDS Training with Internal Data Splitting

This script trains a DNN model using only training data:
- Splits KDDTrain+.txt into: 64% train, 16% validation, 20% test
- Does NOT use separate test file
- Useful for rapid experimentation and cross-validation

Usage:
    python train_dnn_split_only.py --train_path ../nsl-kdd/KDDTrain+.txt
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.dnn import create_dnn_model
from dataset import load_data, preprocess_nsl_kdd
from train import train_model, evaluate_and_save, save_training_results, save_model_and_artifacts, get_next_run_dir, ensure_results_structure


def main():
    parser = argparse.ArgumentParser(
        description='Train DNN IDS with internal train/val/test split',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data (e.g., KDDTrain+.txt)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test set proportion from training data')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split from remaining data after test split')
    
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
    print("DNN IDS Training with Internal Split")
    print("="*80)
    print(f"Training file: {args.train_path}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    train_df = load_data(args.train_path)
    X_full, y_full, scaler, feature_columns = preprocess_nsl_kdd(
        train_df, use_statistical_filter=True
    )
    
    print(f"Full dataset: {X_full.shape}")
    
    # Split into train+val and test
    print("\n" + "="*80)
    print("Splitting Data")
    print("="*80)
    print(f"  Test split: {args.test_split*100:.0f}%")
    print(f"  Validation split (from remaining): {args.validation_split*100:.0f}%")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_full, y_full,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=y_full
    )
    
    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=y_trainval
    )
    
    train_pct = (len(X_train) / len(X_full)) * 100
    val_pct = (len(X_val) / len(X_full)) * 100
    test_pct = (len(X_test) / len(X_full)) * 100
    
    print(f"\nFinal split:")
    print(f"  Training:   {X_train.shape[0]:>6,} samples ({train_pct:.1f}%)")
    print(f"  Validation: {X_val.shape[0]:>6,} samples ({val_pct:.1f}%)")
    print(f"  Test:       {X_test.shape[0]:>6,} samples ({test_pct:.1f}%)")
    print(f"  Total:      {len(X_full):>6,} samples")
    
    # Build model
    print("\n" + "="*80)
    print("Building Model")
    print("="*80)
    
    input_dim = X_full.shape[1]
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
    
    # Prepare output directory
    ensure_results_structure()
    run_dir = get_next_run_dir(base_dir=Path(args.output_dir), prefix='run_dnn_split')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "="*80)
    print("Training Model")
    print("="*80)
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_dir=run_dir,
        model_name='DNN IDS (Internal Split)',
        verbose=1
    )
    
    # Save training results
    save_training_results(history, run_dir)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)
    
    metrics = evaluate_and_save(
        model=model,
        X_test=X_test,
        y_test=y_test,
        run_dir=run_dir,
        num_classes=2,
        binary=True,
        test_name='test'
    )
    
    # Save model and artifacts
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'config': {
            'dropout': args.dropout,
            'l1': args.l1,
            'l2': args.l2,
            'use_batchnorm': args.use_batchnorm,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'test_split': args.test_split,
            'validation_split': args.validation_split
        }
    }
    
    save_model_and_artifacts(model, artifacts, run_dir)
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
