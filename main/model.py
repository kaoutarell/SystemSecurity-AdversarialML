import tensorflow as tf
from tensorflow.keras import regularizers
from pathlib import Path


def build_model(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4, use_batchnorm=False):
    """
    Simple, proven architecture for NSL-KDD IDS.
    
    Architecture: 64 -> 64 -> 32 -> 1
    
    Key design decisions:
    - NO BatchNormalization (causes instability with train/test distribution mismatch)
    - Simple dropout regularization
    - Moderate L1/L2 regularization
    - He normal initialization for ReLU activations
    """
    
    model = tf.keras.Sequential(name='ids_model')
    
    # First hidden layer: 64 units
    model.add(tf.keras.layers.Dense(
        64,
        input_shape=(input_dim,),
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    ))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Second hidden layer: 64 units
    model.add(tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    ))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Third hidden layer: 32 units
    model.add(tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    ))
    model.add(tf.keras.layers.Dropout(dropout_rate * 0.5))
    
    # Output layer: binary classification
    model.add(tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.L1L2(l1=l1*0.1, l2=l2*0.1)
    ))
    
    return model


def build_model_with_residual(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4, use_batchnorm=True):
    """
    Alternative architecture with residual connection for better training stability.
    Use this if the simpler model still shows instability.
    """
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Project input to consistent dimension
    x = tf.keras.layers.Dense(
        32,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    )(inputs)
    
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    x = tf.keras.layers.Activation('relu')(x)
    residual = x
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Second block with residual
    x = tf.keras.layers.Dense(
        32,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    )(x)
    
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    x = tf.keras.layers.Add()([x, residual])  # Residual connection
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.7)(x)
    
    # Compression block
    x = tf.keras.layers.Dense(
        16,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        kernel_initializer='he_normal'
    )(x)
    
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    x = tf.keras.layers.Activation('relu')(x)
    
    # Output
    outputs = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.L1L2(l1=l1*0.1, l2=l2*0.1)
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ids_residual_model')
    
    return model


def build_lightweight_model(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4):
    """
    Ultra-lightweight model if overfitting persists.
    This is the minimum viable architecture for NSL-KDD.
    """
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            24, 
            input_shape=(input_dim,),
            activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
            kernel_initializer='he_normal'
        ),
        tf.keras.layers.Dropout(dropout_rate),
        
        tf.keras.layers.Dense(
            12,
            activation='relu',
            kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
            kernel_initializer='he_normal'
        ),
        tf.keras.layers.Dropout(dropout_rate * 0.7),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='ids_lightweight_model')
    
    return model


def compile_model(model, learning_rate=0.001, clipnorm=None):
    """
    Compile model with standard Adam optimizer.
    Simple and proven configuration.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm if clipnorm and clipnorm > 0 else None,
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


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64, 
                patience=5, min_delta=0.001, verbose=1, class_weight=None):
    """
    Training function with early stopping and learning rate reduction.
    """
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"  Model: {model.name}")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_test):,}")
    print(f"  Early stopping patience: {patience} epochs")
    print(f"  Class weights: {'Enabled' if class_weight else 'Disabled'}")
    print("="*60 + "\n")
    
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
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=False
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose
    )
    
    print("\n✓ Training completed!")
    
    return history


def load_model(model_path):
    """Load a saved model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = tf.keras.models.load_model(str(model_path))
    return model
