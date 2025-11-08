
import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import joblib
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

    # Some NSL-KDD files include a difficulty column at the end. If the
    # number of columns is greater than our expected schema, drop the last
    # column (difficulty) so the column count matches `NSL_KDD_COLUMNS`.
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        print("  Detected extra column(s) (e.g. difficulty). Removing trailing columns to match schema...")
        # keep only the first N expected columns
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]

    # Assign column names (will raise if mismatch remains)
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
def build_model(input_dim, architecture='default', dropout_rate=0.3, l1=1e-5, l2=1e-4):
    model = tf.keras.Sequential([
        # Layer 1
        tf.keras.layers.Dense(
            64, activation='relu', input_shape=(input_dim,),
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Layer 2
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Layer 3
        tf.keras.layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Layer 4
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        
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
                epochs=10, batch_size=256, callbacks=None):
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
        callbacks=callbacks,
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


def save_model_and_results(model, history, metrics_dict, output_dir='output', scaler=None, feature_columns=None):
    """Save trained model, history, and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving results to {output_dir}/...")

    model_path = output_dir / 'nn_ids_model.keras'
    model.save(model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Also save a copy in the common models directory for quick reuse by other
    # scripts (adversarial attack, inference). If a `models` directory exists in
    # the repo root we'll place a copy there so other tools can reference it.
    models_dir = Path('models')
    try:
        models_dir.mkdir(exist_ok=True)
        models_copy = models_dir / 'nn_ids_model.keras'
        # use TensorFlow save (copying the saved model folder/file)
        if model_path.is_dir():
            # saved model as directory -> copytree
            import shutil
            if models_copy.exists():
                shutil.rmtree(models_copy)
            shutil.copytree(model_path, models_copy)
        else:
            # single file format (.keras/.h5)
            import shutil
            shutil.copy2(model_path, models_copy)

        print(f"  ✓ Model copy saved to: {models_copy}")
    except Exception:
        # non-fatal: continue if copying fails
        pass

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

    # Save preprocessing artifacts (scaler + feature column ordering) so
    # other scripts (attacks/inference) can reproduce preprocessing.
    try:
        artifacts = {}
        if scaler is not None:
            artifacts['scaler'] = scaler
        if feature_columns is not None:
            artifacts['feature_columns'] = list(feature_columns)

        if artifacts:
            artifacts_path = output_dir / 'artifacts.joblib'
            joblib.dump(artifacts, artifacts_path)
            print(f"  ✓ Preprocessing artifacts saved: {artifacts_path}")
            # also copy to models/ alongside the model copy
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(artifacts_path, models_dir / 'artifacts.joblib')
    except Exception:
        pass


def next_run_dir(base_name: str = 'results_nn_run') -> Path:
    """Create a new incremental results directory based on base_name.

    Examples: results_nn_run_001, results_nn_run_002, ...
    """
    parent = Path('.')
    # Find existing dirs that match pattern base_name_### (numeric suffix)
    siblings = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(f"{base_name}_")]
    max_idx = 0
    for s in siblings:
        parts = s.name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        suffix = parts[1]
        if not suffix.isdigit():
            continue
        try:
            idx = int(suffix)
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue

    next_idx = max_idx + 1
    candidate = Path(f"{base_name}_{next_idx:03d}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


class IDSModel:
    """Helper wrapper to load a trained Keras model and preprocessing artifacts
    and run reproducible preprocessing + inference for NSL-KDD samples.
    """
    def __init__(self, model_path: str = 'models/nn_ids_model.keras', artifacts_path: str = 'models/artifacts.joblib'):
        self.model_path = Path(model_path)
        self.artifacts_path = Path(artifacts_path)
        self.model = None
        self.artifacts = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f'Model not found: {self.model_path}')
        if not self.artifacts_path.exists():
            raise FileNotFoundError(f'Artifacts not found: {self.artifacts_path}')

        # load model and artifacts
        self.model = tf.keras.models.load_model(str(self.model_path))
        self.artifacts = joblib.load(str(self.artifacts_path))
        return self

    def preprocess_file(self, filepath: str, n_samples: int = None):
        """Load up to n_samples from a raw NSL-KDD file and apply the same
        preprocessing used during training (one-hot, reorder features, scale
        numeric columns using the saved scaler).
        Returns: (X, original_index, features_df, y)
        where y is binary labels (0=normal, 1=attack), or None if outcome not available
        """
        df = pd.read_csv(filepath, header=None)
        # drop trailing difficulty column if present
        if df.shape[1] > len(NSL_KDD_COLUMNS):
            df = df.iloc[:, :len(NSL_KDD_COLUMNS)]
        df.columns = NSL_KDD_COLUMNS

        # Extract true labels before preprocessing
        y = None
        if TARGET_COLUMN in df.columns:
            y = (df[TARGET_COLUMN] != 'normal').astype(int).values
            if n_samples is not None:
                y = y[:n_samples]

        # numeric conversion + fill
        for col in ['duration', 'wrong_fragment']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)

        # one-hot encode categorical columns
        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)

        features = df.drop(DROP_COLUMNS, axis=1)

        # get feature ordering the scaler expects
        feature_columns = self.artifacts.get('feature_columns')
        if feature_columns is None:
            raise RuntimeError('artifacts must contain feature_columns')

        features = features.reindex(columns=feature_columns, fill_value=0)

        # determine numeric columns the scaler expects
        scaler = self.artifacts.get('scaler')
        if scaler is None:
            raise RuntimeError('artifacts must contain scaler')

        n_num = None
        if hasattr(scaler, 'n_features_in_'):
            n_num = int(scaler.n_features_in_)
        elif hasattr(scaler, 'scale_'):
            n_num = int(len(scaler.scale_))

        if n_num is None:
            if hasattr(scaler, 'feature_names_in_'):
                num_cols = list(scaler.feature_names_in_)
            else:
                raise RuntimeError('Could not determine numeric columns expected by scaler')
        else:
            num_cols = feature_columns[:n_num]

        # apply scaler only to numeric columns
        num_df = features.loc[:, num_cols].astype(float).copy()
        num_scaled = scaler.transform(num_df)
        features.loc[:, num_cols] = num_scaled

        X = features.values.astype(float)

        if n_samples is not None:
            X = X[:n_samples]
            indices = features.index[:n_samples]
        else:
            indices = features.index

        return X, indices, features, y

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('Model not loaded; call load() first')
        probs = self.model.predict(X)
        preds = (probs > 0.5).astype(int).flatten()
        return probs, preds


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

    # Decide output directory early so callbacks can write there
    out_dir = Path(args.output_dir)
    if getattr(args, 'increment', False):
        out_dir = next_run_dir(args.base_output_dir)

    model = None
    # If user provided a pre-trained model path, try to load it. This supports
    # complete Keras models saved with `model.save(...)` (.keras/.h5 or SavedModel
    # directories) as well as raw weight files that can be loaded via
    # `model.load_weights(...)` after building the architecture.
    if getattr(args, 'load_model', None):
        load_path = Path(args.load_model)
        if not load_path.exists():
            raise FileNotFoundError(f"Specified load_model path not found: {load_path}")

        # Attempt to load a full model first
        try:
            print(f"  Loading full Keras model from: {load_path}")
            loaded = tf.keras.models.load_model(str(load_path))
            model = loaded
            print("  ✓ Loaded full model from provided path")
        except Exception as e:
            print(f"  Could not load full model ({e}). Will try to build architecture and load weights.")
            # Build architecture and try to load weights (use provided hyperparams)
            model = build_model(
                X_train.shape[1],
                dropout_rate=args.dropout,
                l1=args.l1,
                l2=args.l2
            )
            try:
                model.load_weights(str(load_path))
                print("  ✓ Weights loaded into built model")
            except Exception as e2:
                raise RuntimeError(f"Failed to load weights from {load_path}: {e2}")
    else:
        model = build_model(
            X_train.shape[1],
            dropout_rate=args.dropout,
            l1=args.l1,
            l2=args.l2
        )

    # Create optimizer per CLI args (allows changing LR / clipping)
    opt = None
    if args.optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=(args.clipnorm if args.clipnorm>0 else None))
    elif args.optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9, clipnorm=(args.clipnorm if args.clipnorm>0 else None))
    else:
        print(f"Unknown optimizer {args.optimizer}, defaulting to Adam")
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Re-compile model with chosen optimizer
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # Prepare callbacks
    callbacks = []
    # ModelCheckpoint: save best weights
    try:
        chk_path = out_dir / 'best_weights.h5'
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(str(chk_path), save_best_only=True, save_weights_only=True, monitor='val_loss'))
    except Exception:
        pass

    if args.reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.reduce_factor, patience=args.patience, min_lr=args.min_lr, verbose=1))

    if args.early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience, restore_best_weights=True, verbose=1))

    print(f"  Total parameters: {model.count_params():,}")

    print("\n" + "="*60)
    print("Training")
    print("="*60)
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    metrics_dict, y_pred, cm = evaluate_model(model, X_test, y_test)

    # Save model and artifacts (include scaler + feature ordering for reproducible preprocessing)
    # feature columns are the columns after preprocessing and after dropping the target
    feature_columns = df_processed.drop(DROP_COLUMNS, axis=1).columns

    # Choose output directory: if incremental mode is enabled, create a new
    # incremented folder under args.base_output_dir, otherwise use args.output_dir
    out_dir = Path(args.output_dir)
    if getattr(args, 'increment', False):
        out_dir = next_run_dir(args.base_output_dir)

    save_model_and_results(model, history, metrics_dict, str(out_dir), scaler=scaler, feature_columns=feature_columns)

    print("\nGenerating visualizations...")
    plot_training_history(
        history,
        save_path=out_dir / 'training_curves.png'
    )
    plot_confusion_matrix(
        cm,
        save_path=out_dir / 'confusion_matrix.png'
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
        '--load_model',
        type=str,
        default=None,
        help='Optional path to a pre-trained Keras model (.keras/.h5 or SavedModel dir) or weights file to load before training/evaluation'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output',
        help='Directory to save model and results (default: output)'
    )

    parser.add_argument(
        '--base_output_dir',
        type=str,
        default='results_nn_run',
        help='Base name for incrementing results directories (default: results_nn_run)'
    )

    parser.add_argument(
        '--increment',
        action='store_true',
        default=True,
        help='If set (default), create an auto-incremented results folder for this run'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate to use in the model (default: 0.3)'
    )

    parser.add_argument(
        '--l1',
        type=float,
        default=1e-5,
        help='L1 regularization factor (default: 1e-5)'
    )

    parser.add_argument(
        '--l2',
        type=float,
        default=1e-4,
        help='L2 regularization factor (default: 1e-4)'
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
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for optimizer (default: 1e-3)'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='Optimizer to use: adam|sgd (default: adam)'
    )

    parser.add_argument(
        '--clipnorm',
        type=float,
        default=0.0,
        help='Gradient clipping norm (0 = disabled)'
    )

    parser.add_argument(
        '--reduce_lr',
        action='store_true',
        default=True,
        help='Enable ReduceLROnPlateau callback (default: True)'
    )

    parser.add_argument(
        '--reduce_factor',
        type=float,
        default=0.5,
        help='Factor by which the LR will be reduced on plateau (default: 0.5)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Number of epochs with no improvement after which LR is reduced (default: 3)'
    )

    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-7,
        help='Minimum learning rate for ReduceLROnPlateau (default: 1e-7)'
    )

    parser.add_argument(
        '--early_stop',
        action='store_true',
        default=True,
        help='Enable EarlyStopping (default: True)'
    )

    parser.add_argument(
        '--es_patience',
        type=int,
        default=8,
        help='EarlyStopping patience in epochs (default: 8)'
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