#!/usr/bin/env python3
"""
Common Training Module

Provides unified training logic for all IDS models:
- Dense NN (IDS)
- CNN with Attention
- SAAE-DNN

This module centralizes training workflows to eliminate duplication.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from evaluate import plot_confusion_matrix, plot_training_curves, evaluate_model


def get_next_run_dir(base_dir='results/runs', prefix='run'):
    """
    Get next available run directory with auto-incremented number.
    
    Args:
        base_dir: Base directory for runs
        prefix: Prefix for run directories
    
    Returns:
        Path to next run directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find existing run directories
    existing_runs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    
    if not existing_runs:
        next_num = 1
    else:
        # Extract numbers from existing runs
        numbers = []
        for run_dir in existing_runs:
            try:
                num = int(run_dir.name.replace(prefix + '_', ''))
                numbers.append(num)
            except ValueError:
                continue
        
        next_num = max(numbers) + 1 if numbers else 1
    
    # Create next run directory
    next_run = base_path / f"{prefix}_{next_num:03d}"
    next_run.mkdir(parents=True, exist_ok=True)
    
    return next_run


def ensure_results_structure():
    base_path = Path('results')
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / 'models').mkdir(exist_ok=True)
    (base_path / 'runs').mkdir(exist_ok=True)
    (base_path / 'attacks').mkdir(exist_ok=True)
    
def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=256,
                learning_rate=0.001, run_dir=None, model_name='model',
                verbose=1):
    """
    Universal training function for all IDS models.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate (used for info only, model should be pre-compiled)
        run_dir: Directory to save results
        model_name: Name of the model for logging
        verbose: Verbosity level
    
    Returns:
        history: Training history object
    """
    print("\n" + "="*80)
    print(f"Training {model_name}")
    print("="*80)
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    if run_dir:
        print(f"  Output directory: {run_dir}")
    print("="*80 + "\n")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
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
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\n✓ Training complete!")
    
    return history


def evaluate_and_save(model, X_test, y_test, run_dir, num_classes=2, 
                      binary=True, test_name='test'):
    """
    Evaluate model and save results.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        run_dir: Directory to save results
        num_classes: Number of classes
        binary: Whether binary classification
        test_name: Name for test set (e.g., 'test', 'kddtest+', 'kddtest-21')
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluation on {test_name.upper()}")
    print("="*80)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, num_classes=num_classes, binary=binary)
    
    # Save metrics
    metrics_file = run_dir / f'metrics_{test_name}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_file}")
    
    # Generate and save confusion matrix
    y_pred_probs = model.predict(X_test, verbose=0)
    
    if binary or num_classes == 2:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    cm_file = run_dir / f'confusion_matrix_{test_name}.png'
    plot_confusion_matrix(y_true, y_pred, cm_file)
    print(f"✓ Confusion matrix saved to {cm_file}")
    
    return metrics


def save_training_results(history, run_dir):
    """
    Save training history and plots.
    
    Args:
        history: Training history object
        run_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("Saving Training Results")
    print("="*80)
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_file = run_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved to {history_file}")
    
    # Save training curves
    curves_file = run_dir / 'training_curves.png'
    plot_training_curves(history, curves_file)
    print(f"✓ Training curves saved to {curves_file}")


def save_model_and_artifacts(model, artifacts, run_dir, model_filename='final_model.keras'):
    """
    Save model and preprocessing artifacts.
    
    Args:
        model: Trained Keras model
        artifacts: Dictionary of artifacts (scaler, feature_columns, etc.)
        run_dir: Directory to save results
        model_filename: Filename for the model
    """
    import joblib
    
    print("\n" + "="*80)
    print("Saving Model and Artifacts")
    print("="*80)
    
    # Save model
    model_file = run_dir / model_filename
    model.save(model_file)
    print(f"✓ Model saved to {model_file}")
    
    # Save artifacts
    artifacts_file = run_dir / 'artifacts.joblib'
    joblib.dump(artifacts, run_dir / 'artifacts.joblib')
    print(f"✓ Artifacts saved to {artifacts_file}")


def run_training_pipeline(
    model,
    X_train_full, y_train_full,
    X_test, y_test,
    args,
    artifacts,
    model_name='model',
    run_prefix='run'
):
    """
    Complete training pipeline for any IDS model.
    
    Args:
        model: Compiled Keras model
        X_train_full: Full training features
        y_train_full: Full training labels
        X_test: Test features
        y_test: Test labels
        args: Argument namespace with training parameters
        artifacts: Dictionary of preprocessing artifacts
        model_name: Name of the model
        run_prefix: Prefix for run directory (e.g., 'run_ids', 'run_cnn')
    
    Returns:
        run_dir: Path to results directory
        history: Training history
        metrics: Test metrics
    """
    # Ensure results structure
    ensure_results_structure()
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.validation_split,
        random_state=42,
        stratify=y_train_full if len(y_train_full.shape) == 1 else y_train_full.argmax(axis=1)
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create run directory
    run_dir = get_next_run_dir(base_dir=Path(args.output_dir), prefix=run_prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=getattr(args, 'learning_rate', 0.001),
        run_dir=run_dir,
        model_name=model_name,
        verbose=1
    )
    
    # Save training results
    save_training_results(history, run_dir)
    
    # Determine if binary classification
    num_classes = 2 if len(y_train_full.shape) == 1 else y_train_full.shape[1]
    binary = (num_classes == 2) or getattr(args, 'binary', True)
    
    # Evaluate on test set
    metrics = evaluate_and_save(
        model=model,
        X_test=X_test,
        y_test=y_test,
        run_dir=run_dir,
        num_classes=num_classes,
        binary=binary,
        test_name='test'
    )
    
    # Save model and artifacts
    save_model_and_artifacts(model, artifacts, run_dir)
    
    print("\n" + "="*80)
    print("Training Pipeline Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}\n")
    
    return run_dir, history, metrics
