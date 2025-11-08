"""
Model architecture and training logic for IDS Neural Network.
"""

import tensorflow as tf
from tensorflow.keras import regularizers
from pathlib import Path


def build_model(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4):
    """
    Build neural network model for intrusion detection.
    
    Architecture:
        - 4 hidden layers: 64 -> 128 -> 256 -> 64
        - ReLU activation
        - Dropout after each hidden layer
        - L1/L2 regularization
        - Sigmoid output for binary classification
    
    Args:
        input_dim: Number of input features
        dropout_rate: Dropout rate (default: 0.3)
        l1: L1 regularization coefficient (default: 1e-5)
        l2: L2 regularization coefficient (default: 1e-4)
    
    Returns:
        Compiled Keras Sequential model
    """
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
    
    return model


def compile_model(model, learning_rate=0.001, optimizer_name='adam', clipnorm=None):
    """
    Compile the model with specified optimizer and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer ('adam' or 'sgd')
        clipnorm: Gradient clipping norm (None to disable)
    
    Returns:
        Compiled model
    """
    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm if clipnorm and clipnorm > 0 else None
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=clipnorm if clipnorm and clipnorm > 0 else None
        )
    else:
        print(f"Unknown optimizer {optimizer_name}, defaulting to Adam")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile with binary crossentropy loss and metrics
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def create_callbacks(output_dir, reduce_lr=True, reduce_factor=0.5, patience=3,
                     min_lr=1e-7, early_stop=True, es_patience=8):
    """
    Create training callbacks for model checkpointing, learning rate reduction, and early stopping.
    
    Args:
        output_dir: Directory to save model checkpoints
        reduce_lr: Enable ReduceLROnPlateau callback
        reduce_factor: Factor to reduce learning rate
        patience: Epochs to wait before reducing LR
        min_lr: Minimum learning rate
        early_stop: Enable EarlyStopping callback
        es_patience: Epochs to wait before early stopping
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ModelCheckpoint: save best weights
    checkpoint_path = output_dir / 'nn_ids_model.keras'
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )
    
    # ReduceLROnPlateau: reduce learning rate when validation loss plateaus
    if reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_factor,
                patience=patience,
                min_lr=min_lr,
                verbose=1
            )
        )
    
    # EarlyStopping: stop training if validation loss doesn't improve
    if early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=es_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
    
    return callbacks


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64,
                callbacks=None, verbose=1):
    """
    Train the model on provided data.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of Keras callbacks
        verbose: Verbosity level
    
    Returns:
        Training history object
    """
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    print("\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_test):,}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\n Training completed!")
    
    return history


def load_model(model_path):
    """
    Load a saved Keras model.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = tf.keras.models.load_model(str(model_path))
    return model
