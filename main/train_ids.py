#!/usr/bin/env python3
"""
Neural Network IDS Training Script

Improvements from SAAE-DNN:
- Statistical filtering (>80% zero removal) for better feature quality
- Comprehensive evaluation metrics
- Better progress reporting and visualization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from base_model import build_model, compile_model
from evaluate import plot_confusion_matrix, plot_training_curves
from utils import get_next_run_dir, ensure_results_structure


# NSL-KDD column names
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'
]

CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']
TARGET_COLUMN = 'outcome'
DROP_COLUMNS = ['outcome', 'level']


def load_data(filepath):
    """Load NSL-KDD data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]
    df.columns = NSL_KDD_COLUMNS
    print(f"  Loaded: {len(df):,} samples, {df.shape[1]} features")
    
    # Show class distribution
    print(f"  Class distribution:")
    unique, counts = np.unique(df[TARGET_COLUMN], return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 'normal' else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def calculate_zero_percentage(df, numeric_columns):
    """
    Calculate percentage of zeros for each numeric column.
    Used for statistical filtering (Paper Section 4.1.3).
    """
    zero_percentages = {}
    for col in numeric_columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        zero_percentages[col] = zero_pct
    return zero_percentages


def preprocess_nsl_kdd(df, scaler=None, feature_columns=None, features_to_keep=None):
    """
    Preprocess NSL-KDD data with statistical filtering.
    
    Statistical Filtering:
    - Remove features with >80% zero values
    - Results in 18 numeric + 84 one-hot = 102 features
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    print("  Step 1: Handling missing values...")
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 2: Create binary labels (0=normal, 1=attack)
    print("  Step 2: Creating binary labels (0=normal, 1=attack)...")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
    y = df[TARGET_COLUMN].values
    
    # Step 3: One-hot encode categorical features
    print(f"  Step 3: One-hot encoding {CATEGORICAL_COLUMNS}...")
    df_encoded = pd.get_dummies(df[CATEGORICAL_COLUMNS], columns=CATEGORICAL_COLUMNS)
    print(f"    One-hot features: {len(df_encoded.columns)} (from 3 categorical)")
    
    # Separate numeric features
    numeric_features = df.drop(columns=CATEGORICAL_COLUMNS + DROP_COLUMNS)
    
    # Step 4: Statistical filtering (ONLY for training data)
    if features_to_keep is None:
        print(f"  Step 4: Statistical Filtering (>80% zero removal)...")
        # Calculate zero percentages for numeric features
        zero_pcts = calculate_zero_percentage(numeric_features, numeric_features.columns)
        
        # Keep features with <= 80% zeros
        features_to_keep = [col for col, pct in zero_pcts.items() if pct <= 80.0]
        
        removed_features = [col for col, pct in zero_pcts.items() if pct > 80.0]
        print(f"    Original numeric features: {len(numeric_features.columns)}")
        print(f"    Features with >80% zeros: {len(removed_features)}")
        print(f"    Remaining numeric features: {len(features_to_keep)}")
        
        if removed_features and len(removed_features) <= 25:
            print(f"    Removed: {removed_features}")
        elif removed_features:
            print(f"    Removed (first 10): {removed_features[:10]}")
    else:
        print(f"  Step 4: Using pre-defined feature set (test data)...")
        print(f"    Using {len(features_to_keep)} filtered numeric features")
    
    # Keep only selected numeric features
    numeric_features_filtered = numeric_features[features_to_keep]
    
    # Combine: numeric (filtered) + one-hot encoded
    X = pd.concat([numeric_features_filtered, df_encoded], axis=1)
    
    # Align columns for test data
    if feature_columns is None:
        feature_columns = X.columns.tolist()
        print(f"    Total features: {len(feature_columns)}")
        print(f"      = {len(features_to_keep)} numeric + {len(df_encoded.columns)} one-hot")
        print(f"      Expected: ~102 features (18 numeric + 84 one-hot)")
    else:
        # Align with training columns
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
    
    X = X.values.astype(np.float32)
    
    # Step 5: StandardScaler normalization
    print(f"  Step 5: StandardScaler normalization...")
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler, feature_columns, features_to_keep


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=2048,
                patience=15, run_dir=None):
    """Train model with callbacks"""
    print("\n" + "="*80)
    print("Training Neural Network IDS")
    print("="*80)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(5, patience//3),
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if run_dir:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                str(run_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, run_dir, dataset_name='Test'):
    """Evaluate model and save results"""
    print(f"\n{'='*80}")
    print(f"Evaluation on {dataset_name}")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    
    # Calculate metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, classification_report, confusion_matrix)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("="*80)
    
    # Classification report
    target_names = ['Normal', 'Attack']
    
    print("\n" + classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("="*80)
    
    # Save results
    if run_dir:
        with open(run_dir / f'metrics_{dataset_name.lower()}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        plot_confusion_matrix(y_true, y_pred, 
                             run_dir / f'confusion_matrix_{dataset_name.lower()}.png')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train IDS Neural Network on NSL-KDD Dataset')
    
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
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--l1', type=float, default=5e-5,
                        help='L1 regularization coefficient')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='L2 regularization coefficient')
    parser.add_argument('--use_batchnorm', action='store_true', default=False,
                        help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    
    parser.add_argument('--output_dir', type=str, default='results/runs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Ensure results structure exists
    ensure_results_structure()
    
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
    print("Neural Network Intrusion Detection System")
    print("="*80)
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("Data Preparation")
    print("="*80)
    
    train_df = load_data(args.train_path)
    X_train_full, y_train_full, scaler, feature_columns, features_to_keep = preprocess_nsl_kdd(train_df)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.validation_split,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Load test data if provided
    if args.use_separate_test and args.test_path:
        test_df = load_data(args.test_path)
        X_test, y_test, _, _, _ = preprocess_nsl_kdd(
            test_df,
            scaler=scaler,
            feature_columns=feature_columns,
            features_to_keep=features_to_keep
        )
        print(f"Test set: {X_test.shape}")
    else:
        X_test = X_val
        y_test = y_val
        print("Using validation set as test set")
    
    # Build model
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    
    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")
    
    model = build_model(
        input_dim=input_dim,
        dropout_rate=args.dropout,
        l1=args.l1,
        l2=args.l2,
        use_batchnorm=args.use_batchnorm
    )
    
    model = compile_model(
        model,
        learning_rate=args.learning_rate
    )
    
    print(f"  Total parameters: {model.count_params():,}")
    
    # Create run directory
    run_dir = get_next_run_dir(base_dir=Path(args.output_dir), prefix='run_ids')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {run_dir}")
    
    # Train
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        run_dir=run_dir
    )
    
    # Save model
    model.save(run_dir / 'nn_ids_model.keras')
    print(f"\nModel saved: {run_dir / 'nn_ids_model.keras'}")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, run_dir, 'Test')
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    plot_training_curves(history, run_dir / 'training_curves.png')
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'features_to_keep': features_to_keep
    }
    joblib.dump(artifacts, run_dir / 'artifacts.joblib')
    print(f"Artifacts saved to {run_dir / 'artifacts.joblib'}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nResults saved to: {run_dir}")


if __name__ == '__main__':
    main()
