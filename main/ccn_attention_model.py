"""
CNN Channel Attention Intrusion Detection System - BINARY CLASSIFICATION VERSION

Based on: "CNN Channel Attention Intrusion Detection System Using NSL-KDD Dataset"
Authors: Fatma S. Alrayes, Mohammed Zakariah, Syed Umar Amin, Zafar Iqbal Khan, Jehad Saad Alqurni

This version adapts the paper's architecture for BINARY classification (normal vs. attack)
while maintaining the CNN + ECA attention mechanism that achieves high accuracy.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
from pathlib import Path


class ECALayer(layers.Layer):
    """
    Efficient Channel Attention (ECA) Layer
    
    The ECA module captures cross-channel interactions without dimensionality reduction.
    It uses 1D convolution with kernel size k to model local cross-channel interactions.
    """
    def __init__(self, kernel_size=3, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # Global Average Pooling will reduce spatial dimensions
        self.gap = layers.GlobalAveragePooling2D()
        
        # 1D Convolution for channel attention
        self.conv = layers.Conv1D(
            1, 
            kernel_size=self.kernel_size, 
            padding='same', 
            use_bias=False
        )
        
        super(ECALayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        
        # Global Average Pooling: (batch, height, width, channels) -> (batch, channels)
        y = self.gap(inputs)
        
        # Reshape for 1D conv: (batch, channels) -> (batch, channels, 1)
        y = tf.expand_dims(y, axis=-1)
        
        # Apply 1D convolution: (batch, channels, 1) -> (batch, channels, 1)
        y = self.conv(y)
        
        # Remove last dimension: (batch, channels, 1) -> (batch, channels)
        y = tf.squeeze(y, axis=-1)
        
        # Sigmoid activation to get attention weights
        y = tf.nn.sigmoid(y)
        
        # Reshape attention weights: (batch, channels) -> (batch, 1, 1, channels)
        y = tf.reshape(y, [-1, 1, 1, tf.shape(inputs)[-1]])
        
        # Apply attention weights to input: element-wise multiplication
        return inputs * y
    
    def get_config(self):
        config = super(ECALayer, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


def build_cnn_channel_attention_binary(input_shape=(12, 12, 1)):
    """
    Build CNN with Channel Attention model for BINARY classification.
    
    This adapts the paper's architecture (which achieved 99.728% for 5-class)
    to binary classification (normal vs. attack).
    
    Architecture:
    - Input: Reshaped NSL-KDD features as 2D images (12x12 or 28x28)
    - Multiple Conv2D + ECA + MaxPooling blocks
    - Global Average Pooling
    - Dense layers for binary classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
                    Use (12, 12, 1) or (28, 28, 1)
    
    Returns:
        model: Compiled Keras model for binary classification
    """
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # First Conv Block + ECA
    x = layers.Conv2D(32, (3, 3), padding='same', 
                     kernel_initializer='he_normal',
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = ECALayer(kernel_size=3, name='eca1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Second Conv Block + ECA
    x = layers.Conv2D(64, (3, 3), padding='same',
                     kernel_initializer='he_normal',
                     name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = ECALayer(kernel_size=3, name='eca2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Third Conv Block + ECA
    x = layers.Conv2D(128, (3, 3), padding='same',
                     kernel_initializer='he_normal',
                     name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = ECALayer(kernel_size=3, name='eca3')(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation='relu', 
                    kernel_initializer='he_normal',
                    name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    
    x = layers.Dense(64, activation='relu',
                    kernel_initializer='he_normal', 
                    name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # Output layer - BINARY classification (sigmoid activation)
    outputs = layers.Dense(1, activation='sigmoid', 
                          name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_ECA_Binary_IDS')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model for binary classification.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        ]
    )
    
    return model


def preprocess_nslkdd_for_cnn_binary(X, target_shape=(12, 12)):
    """
    Preprocess NSL-KDD features for CNN (binary classification version).
    
    Steps:
    1. Pad/truncate to square number of features
    2. MinMax normalization to [0, 1]
    3. Reshape into square images
    
    Args:
        X: Input features (samples, features) - should be after one-hot encoding
        target_shape: Target image shape (12x12 or 28x28)
    
    Returns:
        X_reshaped: Data reshaped to (samples, height, width, 1)
    """
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Calculate target number of features needed
    target_features = target_shape[0] * target_shape[1]
    
    # Pad features if necessary
    if n_features < target_features:
        padding = target_features - n_features
        X_padded = np.pad(X, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    elif n_features > target_features:
        # Truncate if too many features
        X_padded = X[:, :target_features]
    else:
        X_padded = X
    
    # MinMax normalization to [0, 1] range
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler.fit_transform(X_padded)
    
    # Reshape to square matrices
    X_reshaped = X_normalized.reshape(n_samples, target_shape[0], target_shape[1], 1)
    
    return X_reshaped, scaler


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=128, verbose=1, class_weight=None):
    """
    Train the binary classification model.
    
    Args:
        model: Compiled Keras model
        X_train: Training images (samples, height, width, 1)
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Verbosity mode
        class_weight: Optional class weights for imbalanced data
    
    Returns:
        history: Training history
    """
    
    print("\n" + "="*80)
    print("CNN Channel Attention Binary IDS - Training Configuration")
    print("="*80)
    print(f"  Model: {model.name}")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Classification: Binary (0=Normal, 1=Attack)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    if class_weight:
        print(f"  Class weights: {class_weight}")
    print("="*80 + "\n")
    
    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_cnn_eca_binary_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose
    )
    
    print("\n✓ Training completed!")
    print(f"✓ Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the binary classification model.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (0=normal, 1=attack)
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("\n" + "="*80)
    print("Model Evaluation - Binary Classification")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, confusion_matrix, classification_report)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.5f} ({accuracy*100:.3f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\n  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'prediction_probs': y_pred_probs
    }


def load_model(model_path):
    """
    Load a saved model from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={'ECALayer': ECALayer}
    )
    return model


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CNN Channel Attention Binary IDS")
    print("Binary Classification: Normal (0) vs. Attack (1)")
    print("="*80)
    
    # Build model with 12x12 input
    print("\nBuilding model with 12x12 input shape...")
    model_12x12 = build_cnn_channel_attention_binary(input_shape=(12, 12, 1))
    model_12x12 = compile_model(model_12x12)
    
    print("\nModel Architecture Summary (12x12 input):")
    model_12x12.summary()
    
    print(f"\nTotal parameters: {model_12x12.count_params():,}")
    
    # Build model with 28x28 input (alternative)
    print("\n" + "="*80)
    print("\nBuilding model with 28x28 input shape...")
    model_28x28 = build_cnn_channel_attention_binary(input_shape=(28, 28, 1))
    model_28x28 = compile_model(model_28x28)
    
    print("\nModel Architecture Summary (28x28 input):")
    model_28x28.summary()
    
    print("\n" + "="*80)
    print("\nKey Features:")
    print("  ✓ ECA (Efficient Channel Attention) mechanism")
    print("  ✓ CNN architecture with Conv2D layers")
    print("  ✓ Global Average Pooling")
    print("  ✓ Binary classification (Normal vs. Attack)")
    print("  ✓ Sigmoid activation for output")
    print("  ✓ Binary crossentropy loss")
    print("  ✓ Expected accuracy: 97-99% on NSL-KDD")
    print("="*80)
    
    print("\n" + "="*80)
    print("Data Preprocessing Requirements:")
    print("="*80)
    print("1. One-hot encode categorical features")
    print("2. Convert labels to binary: 0=normal, 1=attack")
    print("3. Pad features to 144 (for 12x12) or 784 (for 28x28)")
    print("4. Apply MinMax normalization to [0, 1]")
    print("5. Reshape to 2D images: (samples, height, width, 1)")
    print("="*80)