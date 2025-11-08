#!/usr/bin/env python3
"""
CNN Channel Attention IDS Training Script

Modernized approach combining:
- CNN spatial feature extraction
- ECA (Efficient Channel Attention) mechanism
- Lessons learned: NO BatchNorm for NSL-KDD (distribution mismatch issue)
- Simplified architecture with proven regularization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Import existing data loading utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset import process_train_test_files, load_data
from evaluate import plot_confusion_matrix, plot_training_curves


class ECABlock(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) Block
    
    Simplified version without BatchNorm - better for NSL-KDD
    """
    def __init__(self, kernel_size=3, **kwargs):
        super(ECABlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(
            1, 
            kernel_size=self.kernel_size, 
            padding='same',
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        super(ECABlock, self).build(input_shape)
    
    def call(self, inputs):
        # Global Average Pooling
        y = self.gap(inputs)
        y = tf.expand_dims(y, axis=-1)
        
        # 1D Convolution for channel attention
        y = self.conv(y)
        y = tf.squeeze(y, axis=-1)
        
        # Sigmoid for attention weights
        y = tf.nn.sigmoid(y)
        y = tf.reshape(y, [-1, 1, 1, tf.shape(inputs)[-1]])
        
        # Apply attention
        return inputs * y
    
    def get_config(self):
        config = super(ECABlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


def build_cnn_attention_model(input_dim, image_size=12, dropout=0.4, l1=1e-5, l2=5e-4):
    """
    Build CNN with ECA Attention for IDS.
    
    Following paper methodology (Alrayes et al., 2024):
    - Input: (image_size, image_size, 1) 2D images
    - NO BatchNormalization (NSL-KDD distribution mismatch)
    - ECA attention for channel-wise feature importance
    - Proven regularization (L1, L2, Dropout)
    
    Args:
        input_dim: Placeholder (images come pre-shaped)
        image_size: Size of square image (12x12 = 144 features)
        dropout: Dropout rate
        l1: L1 regularization
        l2: L2 regularization
    
    Returns:
        Compiled Keras model
    """
    
    # Input is already 2D images: (batch, image_size, image_size, 1)
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name='input_images')
    x = inputs
    
    # First Conv Block + ECA (NO BatchNorm!)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), 
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal',
        name='conv1'
    )(x)
    x = ECABlock(kernel_size=3, name='eca1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = tf.keras.layers.Dropout(dropout * 0.5, name='dropout1')(x)
    
    # Second Conv Block + ECA
    x = tf.keras.layers.Conv2D(
        64, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal',
        name='conv2'
    )(x)
    x = ECABlock(kernel_size=3, name='eca2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = tf.keras.layers.Dropout(dropout * 0.7, name='dropout2')(x)
    
    # Global Average Pooling (reduces overfitting)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense classification head
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal',
        name='dense1'
    )(x)
    x = tf.keras.layers.Dropout(dropout, name='dropout3')(x)
    
    # Output
    outputs = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1*0.1, l2=l2*0.1),
        name='output'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_eca_ids')
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model with proven configuration."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=50, batch_size=2048, patience=10, min_delta=0.001,
                output_dir='results/runs'):
    """Train with proven callbacks configuration."""
    
    # Create output directory
    run_num = 1
    while (Path(output_dir) / f'run_cnn_{run_num:03d}').exists():
        run_num += 1
    
    run_dir = Path(output_dir) / f'run_cnn_{run_num:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CNN + ECA Attention IDS - Training Configuration")
    print("="*80)
    print(f"  Model: {model.name}")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_test):,}")
    print(f"  Early stopping patience: {patience} epochs")
    print(f"  Output directory: {run_dir}")
    print("="*80 + "\n")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(3, patience//2),
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training completed!")
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    return history, run_dir


def evaluate_model(model, X_test, y_test, run_dir):
    """Evaluate and save results."""
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)
    print("\nEvaluating model...")
    
    # Get metrics
    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    
    metrics = {}
    for name, value in zip(metric_names, results):
        metrics[name] = float(value)
    
    # Get predictions for calculating additional metrics
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics from sklearn if not in model metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
    if 'accuracy' not in metrics:
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    if 'precision' not in metrics:
        metrics['precision'] = float(precision_score(y_test, y_pred))
    if 'recall' not in metrics:
        metrics['recall'] = float(recall_score(y_test, y_pred))
    if 'auc' not in metrics:
        metrics['auc'] = float(roc_auc_score(y_test, y_pred_probs))
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  AUC:       {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print("="*80)
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + classification_report(
        y_test, y_pred, 
        target_names=['Normal', 'Attack'],
        digits=2
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("="*80)
    
    # Save metrics
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred, run_dir / 'confusion_matrix.png')
    
    print("\n✓ Results saved to:", run_dir)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN + ECA Attention model for IDS'
    )
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data file (optional)')
    parser.add_argument('--use_separate_test', action='store_true',
                       help='Use separate test file for validation')
    
    # Model hyperparameters
    parser.add_argument('--image_size', type=int, default=11,
                       help='Image size for CNN (11x11=121 features, 12x12=144)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--l1', type=float, default=1e-5,
                       help='L1 regularization')
    parser.add_argument('--l2', type=float, default=5e-4,
                       help='L2 regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Early stopping minimum delta')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/runs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CNN + ECA Attention Intrusion Detection System")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✓ GPU enabled: {len(gpus)} device(s)")
    print(f"✓ TensorFlow version: {tf.__version__}\n")
    
    # Load data using existing utilities
    # Use the new preprocessing pipeline
    if args.use_separate_test and args.test_path:
        # Complete preprocessing pipeline with train/validation split
        X_train, X_val, X_test, y_train, y_val, y_test = process_train_test_files(
            train_path=args.train_path,
            test_path=args.test_path,
            image_size=args.image_size,
            val_split=0.15,  # 85% train, 15% validation (matches PyTorch implementation)
            artifacts_path=Path(args.output_dir) / 'artifacts.joblib'
        )
        
        # For compatibility with existing code, treat validation as test during training
        # (we'll evaluate on the actual test set after training)
        X_train_combined = X_train
        X_test_for_training = X_val
        y_train_combined = y_train
        y_test_for_training = y_val
    else:
        raise ValueError("This script requires --use_separate_test and --test_path")
    
    print(f"\n  Data shapes:")
    print(f"    Training images: {X_train_combined.shape}")
    print(f"    Validation images: {X_test_for_training.shape}")
    print(f"    Test images: {X_test.shape}")
    
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    
    # Input shape is now (12, 12, 1) or (28, 28, 1) - already 2D images
    input_shape = X_train_combined.shape[1:]  # (height, width, channels)
    print(f"  Input shape: {input_shape}")
    
    model = build_cnn_attention_model(
        input_dim=144,  # 12x12, but model will receive 2D images
        image_size=args.image_size,
        dropout=args.dropout,
        l1=args.l1,
        l2=args.l2
    )
    
    model = compile_model(model, learning_rate=args.learning_rate)
    
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Output directory: {args.output_dir}\n")
    
    # Train
    history, run_dir = train_model(
        model, X_train_combined, y_train_combined, X_test_for_training, y_test_for_training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        output_dir=args.output_dir
    )
    
    # Evaluate on actual test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    metrics = evaluate_model(model, X_test, y_test, run_dir)
    
    # Save training curves
    plot_training_curves(history, run_dir / 'training_curves.png')
    
    # Save final model
    model.save(run_dir / 'final_model.keras')
    print(f"\n✓ Model saved to: {run_dir / 'final_model.keras'}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print(f"Final model saved to: {run_dir / 'final_model.keras'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
