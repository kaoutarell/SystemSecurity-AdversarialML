#!/usr/bin/env python3
"""
SAAE-DNN Training Script

Simplified training script that uses:
- models/saae_dnn.py: Model architecture with pretraining
- train.py: Common training pipeline
- dataset.py: Data preprocessing

Two-stage training:
1. SAAE pretraining (greedy layer-wise)
2. DNN classifier fine-tuning
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

from models.saae_dnn import create_saae_dnn_model
from dataset import load_data, preprocess_nsl_kdd
from train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Train SAAE-DNN model (paper implementation)'
    )
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data file')
    parser.add_argument('--test_21_path', type=str, default=None,
                       help='Path to KDDTest-21 file')
    
    # Model architecture (from paper)
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[90, 80],
                       help='SAAE latent dimensions (default: 90 80)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[50, 25, 10],
                       help='DNN hidden dimensions (default: 50 25 10)')
    
    # Classification type
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification (Normal vs Attack)')
    parser.add_argument('--multiclass', action='store_true',
                       help='Multi-class classification (5 classes)')
    
    # Training parameters (from paper Section 5.4)
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='SAAE pre-training epochs (default: 100)')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='SAAE-DNN training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--pretrain_lr', type=float, default=0.05,
                       help='SAAE pre-training learning rate (default: 0.05)')
    parser.add_argument('--train_lr', type=float, default=0.006,
                       help='SAAE-DNN training learning rate (default: 0.006)')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--l1', type=float, default=1e-5,
                       help='L1 regularization (default: 1e-5)')
    parser.add_argument('--l2', type=float, default=1e-4,
                       help='L2 regularization (default: 1e-4)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/saae_runs',
                       help='Output directory (default: results/saae_runs)')
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip SAAE pre-training (use random initialization)')
    parser.add_argument('--use_separate_test', action='store_true',
                       help='Use separate test file for evaluation')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=9281,
                       help='Random seed for reproducibility (default: 9281)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Default to binary if neither specified
    if not args.binary and not args.multiclass:
        args.binary = True
    
    # Map train_epochs to epochs for pipeline compatibility
    args.epochs = args.train_epochs
    args.learning_rate = args.train_lr
    
    print("\n" + "="*80)
    print("SAAE-DNN: Intrusion Detection System")
    print("Paper: Tang et al., 2020 - Symmetry")
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
    
    # Load data
    print("="*80)
    print("Data Preparation")
    print("="*80)
    
    # Load training data
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
    
    # Determine dimensions
    input_dim = X_train_full.shape[1]
    
    if args.binary:
        print("\nClassification type: Binary (Normal vs Attack)")
        num_classes = 2
    else:
        print("\nClassification type: Multi-class (Normal, Probe, DoS, U2R, R2L)")
        num_classes = 5
        # Convert to categorical if multi-class
        y_train_full = tf.keras.utils.to_categorical(y_train_full, num_classes)
        if X_test is not None:
            y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print(f"  Feature dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    
    # CRITICAL: Split into train and validation BEFORE pretraining
    # The old code only pretrains on the training portion, not the full dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.validation_split,
        random_state=9281,
        stratify=y_train_full if num_classes == 2 else y_train_full.argmax(axis=1)
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    
    # Build SAAE-DNN model
    print("\n" + "="*80)
    print("Building SAAE-DNN Model")
    print("="*80)
    print(f"  SAAE layers: {input_dim} -> {' -> '.join(map(str, args.latent_dims))}")
    print(f"  DNN layers: {args.latent_dims[-1]} -> {' -> '.join(map(str, args.hidden_dims))} -> {num_classes}")
    
    if not args.skip_pretrain:
        print(f"\n  Pre-training SAAE...")
        print(f"    Epochs: {args.pretrain_epochs}")
        print(f"    Learning rate: {args.pretrain_lr}")
        print(f"    Training samples for pretraining: {len(X_train):,}")
        
        model, pretrain_histories = create_saae_dnn_model(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dims=args.latent_dims,
            hidden_dims=args.hidden_dims,
            l1=args.l1,
            l2=args.l2,
            dropout=args.dropout,
            learning_rate=args.train_lr,
            pretrain=True,
            X_train=X_train,  # Use split training data, NOT full dataset
            pretrain_epochs=args.pretrain_epochs,
            pretrain_lr=args.pretrain_lr,
            pretrain_batch_size=args.batch_size
        )
    else:
        print("\n⚠ Skipping SAAE pre-training (using random initialization)")
        model = create_saae_dnn_model(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dims=args.latent_dims,
            hidden_dims=args.hidden_dims,
            l1=args.l1,
            l2=args.l2,
            dropout=args.dropout,
            learning_rate=args.train_lr,
            pretrain=False
        )
    
    print(f"\n  Total parameters: {model.count_params():,}")
    
    # Prepare artifacts for saving
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'latent_dims': args.latent_dims,
        'hidden_dims': args.hidden_dims
    }
    
    # If no separate test set, use validation set as test
    if X_test is None:
        X_test, y_test = X_val, y_val
    
    # Manual training without run_training_pipeline since we already split
    from utils import get_next_run_dir, ensure_results_structure
    from train import train_model, evaluate_and_save, save_training_results, save_model_and_artifacts
    
    ensure_results_structure()
    
    # Create run directory
    run_dir = get_next_run_dir(base_dir=Path(args.output_dir), prefix='run_saae_dnn')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print("\n" + "="*80)
    print("Training SAAE-DNN Classifier")
    print("="*80)
    print(f"  Training epochs: {args.train_epochs}")
    print(f"  Learning rate: {args.train_lr}")
    print(f"  Batch size: {args.batch_size}")
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.train_lr,
        run_dir=run_dir,
        model_name='SAAE-DNN',
        verbose=1
    )
    
    # Save training results
    save_training_results(history, run_dir)
    
    # Evaluate on test set
    metrics = evaluate_and_save(
        model=model,
        X_test=X_test,
        y_test=y_test,
        run_dir=run_dir,
        num_classes=num_classes,
        binary=(num_classes == 2) or args.binary,
        test_name='test'
    )
    
    # Save model and artifacts
    save_model_and_artifacts(model, artifacts, run_dir)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}\n")


if __name__ == "__main__":
    main()
