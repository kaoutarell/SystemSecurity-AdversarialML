#!/usr/bin/env python3
"""
CNN Channel Attention IDS Training Script

Improvements from SAAE-DNN:
- Statistical filtering (>80% zero removal) for better feature quality
- Comprehensive evaluation metrics
- Better progress reporting and visualization

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
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import existing data loading utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import plot_confusion_matrix, plot_training_curves


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
    """Calculate percentage of zeros for each numeric column."""
    zero_percentages = {}
    for col in numeric_columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        zero_percentages[col] = zero_pct
    return zero_percentages


def preprocess_nsl_kdd(df, scaler=None, feature_columns=None, features_to_keep=None, image_size=12):
    """
    Preprocess NSL-KDD data with statistical filtering + reshape to 2D images.
    
    Statistical Filtering:
    - Remove features with >80% zero values
    - Results in ~102 features (18 numeric + 84 one-hot)
    - Then reshape to (image_size, image_size, 1) for CNN
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
        zero_pcts = calculate_zero_percentage(numeric_features, numeric_features.columns)
        features_to_keep = [col for col, pct in zero_pcts.items() if pct <= 80.0]
        removed_features = [col for col, pct in zero_pcts.items() if pct > 80.0]
        
        print(f"    Original numeric features: {len(numeric_features.columns)}")
        print(f"    Features with >80% zeros: {len(removed_features)}")
        print(f"    Remaining numeric features: {len(features_to_keep)}")
        
        if removed_features and len(removed_features) <= 25:
            print(f"    Removed: {removed_features}")
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
    else:
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
    
    # Step 6: Reshape to 2D images for CNN
    print(f"  Step 6: Reshaping to ({image_size}, {image_size}, 1) images for CNN...")
    target_features = image_size * image_size
    
    if X.shape[1] < target_features:
        # Pad with zeros
        padding = target_features - X.shape[1]
        X = np.pad(X, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        print(f"    Padded {X.shape[1] - target_features} features to reach {target_features}")
    elif X.shape[1] > target_features:
        # Truncate
        X = X[:, :target_features]
        print(f"    Truncated to {target_features} features")
    
    # Reshape to (batch, height, width, channels)
    X_images = X.reshape(-1, image_size, image_size, 1)
    print(f"    Final shape: {X_images.shape}")
    
    return X_images, y, scaler, feature_columns, features_to_keep


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
    print("CNN + ECA Attention IDS - Training")
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


def evaluate_model(model, X_test, y_test, run_dir, dataset_name='Test'):
    """Evaluate and save results."""
    print(f"\n{'='*80}")
    print(f"Evaluation on {dataset_name}")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    
    # Calculate metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, classification_report, confusion_matrix, 
                                 roc_auc_score)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred_probs)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
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
                       help='Image size for CNN (12x12=144 features)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--l1', type=float, default=1e-5,
                       help='L1 regularization')
    parser.add_argument('--l2', type=float, default=5e-4,
                       help='L2 regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
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
    X_train_full, y_train_full, scaler, feature_columns, features_to_keep = preprocess_nsl_kdd(
        train_df, image_size=args.image_size
    )
    
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
            features_to_keep=features_to_keep,
            image_size=args.image_size
        )
        print(f"Test set: {X_test.shape}")
    else:
        X_test = X_val
        y_test = y_val
        print("Using validation set as test set")
    
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    
    print(f"  Input shape: {X_train.shape[1:]}")
    
    model = build_cnn_attention_model(
        input_dim=X_train.shape[1] * X_train.shape[2],
        image_size=args.image_size,
        dropout=args.dropout,
        l1=args.l1,
        l2=args.l2
    )
    
    model = compile_model(model, learning_rate=args.learning_rate)
    
    print(f"  Total parameters: {model.count_params():,}")
    
    # Train
    history, run_dir = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        output_dir=args.output_dir
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, run_dir, 'Test')
    
    # Save training curves
    plot_training_curves(history, run_dir / 'training_curves.png')
    
    # Save final model
    model.save(run_dir / 'final_model.keras')
    print(f"\n✓ Model saved to: {run_dir / 'final_model.keras'}")
    
    # Save artifacts
    import joblib
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'features_to_keep': features_to_keep,
        'image_size': args.image_size
    }
    joblib.dump(artifacts, run_dir / 'artifacts.joblib')
    print(f"✓ Artifacts saved to: {run_dir / 'artifacts.joblib'}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
