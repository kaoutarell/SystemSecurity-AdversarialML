
import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate', 'outcome', 'level'
]

CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']
TARGET_COLUMN = 'outcome'
DROP_COLUMNS = ['outcome', 'level']


def load_data(filepath):
    """Load NSL-KDD dataset from file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = NSL_KDD_COLUMNS
    
    print(f"  Loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    print(f"\n  Class distribution:")
    print(df[TARGET_COLUMN].value_counts())
    return df


def preprocess_data(df):
    """
    Preprocess NSL-KDD dataset:
    - Convert numeric columns
    - Fill missing values
    - Scale numerical features with RobustScaler
    - Convert labels to binary (0=normal, 1=attack)
    - One-hot encode categorical features
    """
    df = df.copy()
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)
    num_cols = df.drop(DROP_COLUMNS + CATEGORICAL_COLUMNS, axis=1).columns
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    return df, scaler


def prepare_datasets(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""

    X = df.drop(DROP_COLUMNS, axis=1).values
    y = df[TARGET_COLUMN].values.astype(int)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    print(f"\nDataset split:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Testing:  {X_test.shape[0]:,} samples")
    print(f"  Features: {X_train.shape[1]}")

    print(f"\n  Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def build_model(input_dim, architecture='default'):
    """
    Build neural network model for intrusion detection.
    
    Architecture:
        - 4 hidden layers: 64 -> 128 -> 256 -> 64
        - ReLU activation
        - Dropout (0.3) after each hidden layer
        - L1/L2 regularization
        - Sigmoid output for binary classification
    """
    model = tf.keras.Sequential([
        # Layer 1
        tf.keras.layers.Dense(
            64, activation='relu', input_shape=(input_dim,),
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)
        ),
        tf.keras.layers.Dropout(0.3),
        
        # Layer 2
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)
        ),
        tf.keras.layers.Dropout(0.3),
        
        # Layer 3
        tf.keras.layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)
        ),
        tf.keras.layers.Dropout(0.3),
        
        # Layer 4
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)
        ),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=10, batch_size=256):
    """Train the neural network model."""
    print("\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_test.shape[0]:,}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print("\n Training completed!")
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    print("\nEvaluating model...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3]
    }
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"  Accuracy:  {metrics_dict['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics_dict['precision']*100:.2f}%")
    print(f"  Recall:    {metrics_dict['recall']*100:.2f}%")
    print(f"  Loss:      {metrics_dict['loss']:.4f}")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Attack'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:,}  (Correct: Normal)")
    print(f"  False Positives: {cm[0,1]:,}  (Normal → Attack)")
    print(f"  False Negatives: {cm[1,0]:,}  (Attack → Normal)")
    print(f"  True Positives:  {cm[1,1]:,}  (Correct: Attack)")
    
    return metrics_dict, y_pred, cm


def plot_training_history(history, save_path=None):
    """Plot training history (accuracy and loss curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Normal', 'Attack']
    )
    
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_model_and_results(model, history, metrics_dict, output_dir='output'):
    """Save trained model, history, and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to {output_dir}/...")

    model_path = output_dir / 'nn_ids_model.keras'
    model.save(model_path)
    print(f"  ✓ Model saved: {model_path}")

    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"  ✓ Training history saved: {history_path}")

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")


def main(args):
    """Main training pipeline."""
    print("="*60)
    print("Neural Network Intrusion Detection System")
    print("="*60)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n✓ GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print(f"✓ TensorFlow version: {tf.__version__}\n")

    print("="*60)
    print("Data Preparation")
    print("="*60)
    df = load_data(args.train_path)
    df_processed, scaler = preprocess_data(df)
    print(f"  After preprocessing: {df_processed.shape[1]} features")
    
    X_train, X_test, y_train, y_test = prepare_datasets(
        df_processed, 
        test_size=args.test_size,
        random_state=args.random_state
    )

    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    model = build_model(X_train.shape[1])
    print(f"  Total parameters: {model.count_params():,}")

    print("\n" + "="*60)
    print("Training")
    print("="*60)
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    metrics_dict, y_pred, cm = evaluate_model(model, X_test, y_test)

    save_model_and_results(model, history, metrics_dict, args.output_dir)

    print("\nGenerating visualizations...")
    plot_training_history(
        history, 
        save_path=Path(args.output_dir) / 'training_curves.png'
    )
    plot_confusion_matrix(
        cm, 
        save_path=Path(args.output_dir) / 'confusion_matrix.png'
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train neural network for intrusion detection on NSL-KDD dataset'
    )
    
    parser.add_argument(
        '--train_path', 
        type=str, 
        required=True,
        help='Path to training data file (e.g., KDDTrain+.txt)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output',
        help='Directory to save model and results (default: output)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=256,
        help='Training batch size (default: 256)'
    )
    
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    main(args)



# RUN THIS COMMAND
# CUDA_VISIBLE_DEVICES="" python train_nn.py \
#     --train_path ./nsl-kdd/KDDTrain+.txt \
#     --output_dir results \
#     --epochs 10 \
#     --batch_size 256 \
#     --test_size 0.2 \
#     --random_state 42