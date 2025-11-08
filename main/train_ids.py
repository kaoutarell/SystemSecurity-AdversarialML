#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import build_model, compile_model, train_model
from dataset import load_data, preprocess_data, prepare_datasets, save_artifacts
from evaluate import evaluate_model, print_classification_report, plot_confusion_matrix, plot_training_curves, save_metrics, save_training_history
from utils import get_next_run_dir, ensure_results_structure


def main():
    parser = argparse.ArgumentParser(description='Train IDS Neural Network on NSL-KDD Dataset')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='nsl-kdd/KDDTrain+.txt',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, default='nsl-kdd/KDDTest+.txt',
                        help='Path to test data (for validation during training)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data for validation (if test_path not used)')
    parser.add_argument('--use_separate_test', action='store_true', default=False,
                        help='Use separate test file for validation instead of splitting train')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--l1', type=float, default=1e-5,
                        help='L1 regularization coefficient')
    parser.add_argument('--l2', type=float, default=1e-4,
                        help='L2 regularization coefficient')
    parser.add_argument('--use_batchnorm', action='store_true', default=True,
                        help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--clipnorm', type=float, default=0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change in validation loss to qualify as improvement')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')
    parser.add_argument('--save_models_dir', type=str, default='results/models',
                        help='Directory to save final model and artifacts')
    
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
    
    print("="*60)
    print("Neural Network Intrusion Detection System")
    print("="*60)
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("Data Preparation")
    print("="*60)
    
    df = load_data(args.train_path)
    df_processed, scaler, feature_columns = preprocess_data(df)
    print(f"  After preprocessing: {df_processed.shape[1]} features")
    
    if args.use_separate_test:
        print(f"\n  Using separate test file for validation: {args.test_path}")
        X_train = df_processed.drop(['outcome', 'level'], axis=1).values.astype('float32')
        y_train = df_processed['outcome'].values.astype('float32')
        
        df_test = load_data(args.test_path)
        df_test_processed, _, _ = preprocess_data(df_test, scaler=scaler, feature_columns=feature_columns)
        X_test = df_test_processed.drop(['outcome', 'level'], axis=1).values.astype('float32')
        y_test = df_test_processed['outcome'].values.astype('float32')
        
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_test):,}")
    else:
        print(f"\n  Splitting training data ({args.val_split*100:.0f}% for validation)")
        X_train, X_test, y_train, y_test = prepare_datasets(
            df_processed,
            test_size=args.val_split,
            random_state=args.random_state
        )
    
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
        output_dir = get_next_run_dir(base_dir='results/runs', prefix='run')
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Output directory: {output_dir}")
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=1
    )
    
    model.save(output_dir / 'nn_ids_model.keras')
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test, verbose=0)
    print_classification_report(y_test, y_pred)
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Save to run directory
    save_metrics(metrics, output_dir / 'metrics.json')
    save_training_history(history, output_dir / 'training_history.json')
    plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix.png')
    plot_training_curves(history, output_dir / 'training_curves.png')
    save_artifacts(scaler, feature_columns, output_dir / 'artifacts.joblib')
    
    # Also save to models directory for easy access
    models_dir = Path(args.save_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / 'nn_ids_model.keras')
    save_artifacts(scaler, feature_columns, models_dir / 'artifacts.joblib')
    print(f"\n✓ Model and artifacts also saved to {models_dir}/")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Final model saved to: {models_dir}/nn_ids_model.keras")


if __name__ == '__main__':
    main()
