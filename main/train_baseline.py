#!/usr/bin/env python3
"""
Baseline IDS Training Script - Original Preprocessing Approach

This script uses the original simple preprocessing approach:
- Standard feature engineering (one-hot encoding, standard scaling)
- NO image transformation (traditional ML approach)
- Dense neural network architecture
- Consistent dataset splitting: 85% train, 15% validation (stratified)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from base_model import build_model, compile_model, train_model
from evaluate import evaluate_model, print_classification_report, plot_confusion_matrix, plot_training_curves, save_metrics, save_training_history
from utils import get_next_run_dir, ensure_results_structure

# Import NSL-KDD constants from dataset module
from dataset import (
    NSL_KDD_COLUMNS, 
    CATEGORICAL_COLUMNS, 
    TARGET_COLUMN, 
    DROP_COLUMNS
)


def load_data(filepath):
    """Load NSL-KDD data from file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None)
    
    # Handle different file formats
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]
    df.columns = NSL_KDD_COLUMNS
    
    print(f"  Loaded: {len(df):,} samples, {df.shape[1]} features")
    
    # Print class distribution
    print(f"\n  Class distribution:")
    print(df[TARGET_COLUMN].value_counts())
    
    return df


def preprocess_data_baseline(df, scaler=None, feature_columns=None):
    """
    Original baseline preprocessing approach:
    1. Handle missing values
    2. Create binary labels (0=normal, 1=attack)
    3. One-hot encode categorical features
    4. Standard scaling (Z-score normalization)
    
    Args:
        df: Raw NSL-KDD DataFrame
        scaler: Fitted StandardScaler (None for training data)
        feature_columns: Expected feature columns (None for training data)
    
    Returns:
        X: Preprocessed features (samples, features)
        y: Binary labels (samples,)
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    print("  Step 1: Handling missing values...")
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 2: Create binary labels
    print("  Step 2: Creating binary labels (0=normal, 1=attack)...")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
    y = df[TARGET_COLUMN].values
    
    # Step 3: One-hot encode categorical features
    print(f"  Step 3: One-hot encoding {CATEGORICAL_COLUMNS}...")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    # Align columns with training set (for test data)
    if feature_columns is not None:
        print("  Aligning features with training set...")
        df = df.reindex(columns=list(feature_columns) + DROP_COLUMNS, fill_value=0)
    else:
        feature_columns = df.drop(DROP_COLUMNS, axis=1).columns.tolist()
    
    # Extract features
    X = df.drop(DROP_COLUMNS, axis=1).values
    print(f"  Features after one-hot encoding: {X.shape[1]}")
    
    # Step 4: Standard scaling (Z-score normalization)
    print("  Step 4: Standard scaling (mean=0, std=1)...")
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("    Fitted new StandardScaler")
    else:
        X_scaled = scaler.transform(X)
        print("    Used existing StandardScaler")
    
    # Verify scaling
    print(f"    Data mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
    
    return X_scaled.astype('float32'), y.astype('float32'), scaler, feature_columns


def prepare_datasets_baseline(X_train_full, y_train_full, val_split=0.15, random_state=9281):
    """
    Split training data into train/validation sets with stratification.
    
    Args:
        X_train_full: Full training features
        y_train_full: Full training labels
        val_split: Validation set proportion (default: 0.15 = 15%)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, y_train, y_val: Train and validation splits
    """
    print(f"\nSplitting training data: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% validation (stratified)...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train_full  # Maintain class balance
    )
    
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Validation set: {X_val.shape[0]:,} samples")
    
    # Print distribution
    print(f"\n  Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\n  Validation set distribution:")
    unique, counts = np.unique(y_val, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_val)*100:.1f}%)")
    
    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser(description='Train Baseline IDS Model (Original Approach)')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data file')
    parser.add_argument('--test_path', type=str, help='Path to separate test data file')
    parser.add_argument('--use_separate_test', action='store_true', help='Use separate test set for final evaluation')
    parser.add_argument('--val_split', type=float, default=0.15, 
                       help='Validation split proportion (default: 0.15 = 15%, matches PyTorch)')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--l1', type=float, default=1e-5, help='L1 regularization')
    parser.add_argument('--l2', type=float, default=5e-4, help='L2 regularization')
    parser.add_argument('--use_batchnorm', action='store_true', help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change for early stopping')
    parser.add_argument('--clipnorm', type=float, default=0.0, help='Gradient clipping norm (0 to disable)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--save_models_dir', type=str, default='results/models', help='Directory to save models')
    parser.add_argument('--random_state', type=int, default=9281, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    
    # Setup
    ensure_results_structure()
    
    print("\n" + "="*60)
    print("Baseline IDS Training - Original Preprocessing Approach")
    print("="*60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✓ GPU enabled: {len(gpus)} device(s)")
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("Data Preparation")
    print("="*60)
    
    df = load_data(args.train_path)
    X_train_full, y_train_full, scaler, feature_columns = preprocess_data_baseline(df)
    
    # Always split training data for validation (stratified)
    X_train, X_val, y_train, y_val = prepare_datasets_baseline(
        X_train_full, y_train_full,
        val_split=args.val_split,
        random_state=args.random_state
    )
    
    # Load separate test set for final evaluation (if specified)
    if args.use_separate_test and args.test_path:
        print(f"\n  Loading separate test file for final evaluation: {args.test_path}")
        df_test = load_data(args.test_path)
        X_test, y_test, _, _ = preprocess_data_baseline(df_test, scaler=scaler, feature_columns=feature_columns)
        
        print(f"  Test samples: {len(X_test):,}")
        print(f"\n  Test set distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = 'Normal' if label == 0 else 'Attack'
            print(f"    {label_name}: {count:,} ({count/len(y_test)*100:.1f}%)")
    else:
        X_test, y_test = None, None
    
    # Build model
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    
    model = build_model(
        input_dim=X_train.shape[1],
        dropout_rate=args.dropout,
        l1=args.l1,
        l2=args.l2,
        use_batchnorm=args.use_batchnorm
    )
    
    model = compile_model(
        model,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm if args.clipnorm > 0 else None
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    
    if args.output_dir is None:
        output_dir = get_next_run_dir(base_dir='results/runs', prefix='run_baseline')
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    
    # Train with validation split
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"  Model: baseline_ids")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print("="*60 + "\n")
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_val,  # Use validation set during training
        y_test=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_delta=args.min_delta,
        verbose=1
    )
    
    model.save(output_dir / 'baseline_ids_model.keras')
    
    # Evaluate on test set (if available) or validation set
    eval_X = X_test if args.use_separate_test else X_val
    eval_y = y_test if args.use_separate_test else y_val
    eval_set_name = "Test" if args.use_separate_test else "Validation"
    
    print("\n" + "="*60)
    print(f"Final Evaluation on {eval_set_name} Set")
    print("="*60)
    
    metrics, y_pred = evaluate_model(model, eval_X, eval_y, verbose=0)
    print_classification_report(eval_y, y_pred)
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Save to run directory
    save_metrics(metrics, output_dir / 'metrics.json')
    save_training_history(history, output_dir / 'training_history.json')
    plot_confusion_matrix(eval_y, y_pred, output_dir / 'confusion_matrix.png')
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'scaler_type': 'StandardScaler',
        'preprocessing': 'baseline'
    }
    joblib.dump(artifacts, output_dir / 'artifacts.joblib')
    print(f"Artifacts saved to {output_dir / 'artifacts.joblib'}")
    
    # Also save to models directory for easy access
    models_dir = Path(args.save_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / 'baseline_ids_model.keras')
    joblib.dump(artifacts, models_dir / 'baseline_artifacts.joblib')
    print(f"Artifacts saved to {models_dir / 'baseline_artifacts.joblib'}")
    print(f"\n✓ Model and artifacts also saved to {models_dir}/")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Final model saved to: {models_dir / 'baseline_ids_model.keras'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
