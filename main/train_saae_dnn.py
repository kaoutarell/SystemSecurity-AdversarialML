#!/usr/bin/env python3
"""
SAAE-DNN: Stacked Attention AutoEncoder with Deep Neural Network
Implementation based on the paper "SAAE-DNN: Deep Learning Method on Intrusion Detection"
by Chaofei Tang, Nurbol Luktarhan, and Yuxin Zhao (2020)

Key architecture:
- Stacked AutoEncoder (SAE) for feature extraction
- Attention Mechanism to highlight key features
- DNN classifier initialized with SAAE encoder weights
- Binary and multi-class classification support

Results from paper (NSL-KDD):
- Binary-classification: 87.74% (KDDTest+), 82.97% (KDDTest-21)
- Multi-classification: 82.14% (KDDTest+), 77.57% (KDDTest-21)
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

from dataset import load_data
from evaluate import plot_confusion_matrix, plot_training_curves


class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention Mechanism Layer
    
    Based on paper equations (5-7):
    - M = tanh(Wa * x' + ba)
    - α_i = softmax(M_i)
    - v = Σ(x' * α_i^T)
    
    This layer learns to weight features by importance.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.Wa = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.ba = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # M = tanh(Wa * x' + ba)
        M = tf.nn.tanh(tf.matmul(inputs, self.Wa) + self.ba)
        
        # α_i = softmax(M_i)
        alpha = tf.nn.softmax(M, axis=-1)
        
        # v = x' * α^T (element-wise multiplication)
        v = inputs * alpha
        
        return v
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


def build_attention_autoencoder(input_dim, latent_dim, activation='relu'):
    """
    Build a single Attention AutoEncoder (AAE)
    
    Architecture from paper:
    - Encoder: input -> latent layer
    - Attention: applied to latent layer
    - Decoder: latent layer -> output
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent layer dimension
        activation: Activation function (default: 'relu')
    
    Returns:
        encoder, decoder, autoencoder models
    """
    # Encoder
    encoder_input = tf.keras.Input(shape=(input_dim,), name='encoder_input')
    encoded = tf.keras.layers.Dense(
        latent_dim,
        activation=activation,
        kernel_initializer='he_normal',
        name='encoder_dense'
    )(encoder_input)
    
    # Attention mechanism
    attention = AttentionLayer(name='attention')(encoded)
    
    encoder = tf.keras.Model(encoder_input, attention, name='encoder')
    
    # Decoder
    decoder_input = tf.keras.Input(shape=(latent_dim,), name='decoder_input')
    decoded = tf.keras.layers.Dense(
        input_dim,
        activation=activation,
        kernel_initializer='he_normal',
        name='decoder_dense'
    )(decoder_input)
    
    decoder = tf.keras.Model(decoder_input, decoded, name='decoder')
    
    # Full autoencoder
    autoencoder_input = tf.keras.Input(shape=(input_dim,), name='ae_input')
    encoded_out = encoder(autoencoder_input)
    decoded_out = decoder(encoded_out)
    
    autoencoder = tf.keras.Model(
        autoencoder_input, 
        decoded_out, 
        name='attention_autoencoder'
    )
    
    return encoder, decoder, autoencoder


def build_stacked_attention_autoencoder(input_dim, latent_dims=[90, 80], activation='relu'):
    """
    Build Stacked Attention AutoEncoder (SAAE)
    
    From paper Section 4.2:
    - Network structure: 102-90-80-90-102
    - Two AAEs stacked together
    - First AAE: 102 -> 90
    - Second AAE: 90 -> 80
    
    Args:
        input_dim: Input feature dimension (102 after preprocessing)
        latent_dims: List of latent dimensions [90, 80]
        activation: Activation function
    
    Returns:
        encoders, decoders, autoencoders for each layer
    """
    encoders = []
    decoders = []
    autoencoders = []
    
    current_dim = input_dim
    
    for i, latent_dim in enumerate(latent_dims):
        encoder, decoder, autoencoder = build_attention_autoencoder(
            current_dim,
            latent_dim,
            activation=activation
        )
        
        encoders.append(encoder)
        decoders.append(decoder)
        autoencoders.append(autoencoder)
        
        current_dim = latent_dim
    
    return encoders, decoders, autoencoders


def build_saae_dnn_classifier(input_dim, num_classes, latent_dims=[90, 80], 
                               hidden_dims=[50, 25, 10], activation='relu',
                               l1=1e-5, l2=1e-4, dropout=0.3):
    """
    Build SAAE-DNN classifier
    
    From paper Section 4.3:
    - Uses trained SAAE encoder weights to initialize DNN
    - DNN hidden layers: [50, 25, 10]
    - Output: softmax for classification
    
    Args:
        input_dim: Input dimension (102)
        num_classes: Number of output classes (2 for binary, 5 for multi)
        latent_dims: SAAE latent dimensions [90, 80]
        hidden_dims: DNN hidden layer dimensions [50, 25, 10]
        activation: Activation function
        l1, l2: Regularization parameters
        dropout: Dropout rate
    
    Returns:
        Full SAAE-DNN model
    """
    inputs = tf.keras.Input(shape=(input_dim,), name='saae_dnn_input')
    x = inputs
    
    # SAAE encoder layers (will be pre-trained)
    # Layer 1: 102 -> 90
    x = tf.keras.layers.Dense(
        latent_dims[0],
        activation=activation,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        name='saae_encoder_1'
    )(x)
    x = AttentionLayer(name='saae_attention_1')(x)
    
    # Layer 2: 90 -> 80
    x = tf.keras.layers.Dense(
        latent_dims[1],
        activation=activation,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        name='saae_encoder_2'
    )(x)
    x = AttentionLayer(name='saae_attention_2')(x)
    
    # DNN classification layers
    for i, hidden_dim in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(
            hidden_dim,
            activation=activation,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            name=f'dnn_hidden_{i+1}'
        )(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dnn_dropout_{i+1}')(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification: 1 unit with sigmoid
        outputs = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1*0.1, l2=l2*0.1),
            name='output'
        )(x)
    else:
        # Multi-class: num_classes units with softmax
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1*0.1, l2=l2*0.1),
            name='output'
        )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='saae_dnn')
    
    return model


def pretrain_autoencoder(autoencoder, X_train, epochs=100, batch_size=256, 
                         learning_rate=0.05, verbose=1):
    """
    Pre-train a single autoencoder
    
    From paper Section 5.4:
    - Learning rate: 0.05 for SAAE
    - Epochs: 100
    - Optimizer: Adam
    - Loss: MSE (reconstruction error)
    
    Args:
        autoencoder: Autoencoder model to train
        X_train: Training data
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        verbose: Verbosity level
    """
    print(f"\nPre-training {autoencoder.name}...")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse',  # Reconstruction error (equation 4 in paper)
        metrics=['mae']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=verbose
    )
    
    history = autoencoder.fit(
        X_train, X_train,  # Autoencoder reconstructs input
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    return history


def pretrain_saae(encoders, decoders, autoencoders, X_train, epochs=100, 
                  batch_size=256, learning_rate=0.05, verbose=1):
    """
    Pre-train Stacked Attention AutoEncoder layer by layer
    
    From paper Section 3.2:
    "AEs are stacked to achieve greedy hierarchical learning, where the lth 
    latent layer is used as input to the l + 1th latent layer in the stack."
    
    This is greedy layer-wise pre-training.
    """
    print("\n" + "="*80)
    print("SAAE Pre-training (Greedy Layer-wise)")
    print("="*80)
    
    X_current = X_train
    histories = []
    
    for i, (encoder, decoder, autoencoder) in enumerate(zip(encoders, decoders, autoencoders)):
        print(f"\nLayer {i+1}/{len(autoencoders)}")
        print(f"  Input dim: {X_current.shape[1]}")
        print(f"  Latent dim: {encoder.output_shape[1]}")
        
        # Train this autoencoder
        history = pretrain_autoencoder(
            autoencoder,
            X_current,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
        )
        histories.append(history)
        
        # Use encoder output as input to next layer
        X_current = encoder.predict(X_current, verbose=0)
        
        print(f"  ✓ Layer {i+1} pre-training complete")
        print(f"  Final loss: {history.history['loss'][-1]:.6f}")
    
    print("\n" + "="*80)
    print("✓ SAAE Pre-training Complete")
    print("="*80)
    
    return histories


def transfer_saae_weights(saae_dnn_model, encoders):
    """
    Transfer pre-trained SAAE encoder weights to SAAE-DNN model
    
    From paper Section 4.3:
    "The weights of the trained SAAE encoder are used to initialize 
    the weights of DNN hidden layer"
    """
    print("\nTransferring pre-trained SAAE weights to DNN...")
    
    # Transfer first encoder weights
    saae_encoder_1_weights = encoders[0].get_layer('encoder_dense').get_weights()
    saae_dnn_model.get_layer('saae_encoder_1').set_weights(saae_encoder_1_weights)
    
    # Transfer first attention weights
    saae_attention_1_weights = encoders[0].get_layer('attention').get_weights()
    saae_dnn_model.get_layer('saae_attention_1').set_weights(saae_attention_1_weights)
    
    # Transfer second encoder weights
    saae_encoder_2_weights = encoders[1].get_layer('encoder_dense').get_weights()
    saae_dnn_model.get_layer('saae_encoder_2').set_weights(saae_encoder_2_weights)
    
    # Transfer second attention weights
    saae_attention_2_weights = encoders[1].get_layer('attention').get_weights()
    saae_dnn_model.get_layer('saae_attention_2').set_weights(saae_attention_2_weights)
    
    print("✓ Weight transfer complete")


def train_saae_dnn(model, X_train, y_train, X_val, y_val, epochs=100, 
                   batch_size=256, learning_rate=0.006, patience=10, verbose=1, 
                   run_dir=None):
    """
    Train SAAE-DNN classifier
    
    From paper Section 5.4:
    - Learning rate: 0.006 for DNN
    - Epochs: 100
    - Optimizer: Adam
    """
    print("\n" + "="*80)
    print("SAAE-DNN Training")
    print("="*80)
    
    num_classes = y_train.shape[1] if len(y_train.shape) > 1 else 2
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if num_classes == 2 or len(y_train.shape) == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'),
                   tf.keras.metrics.AUC(name='auc')]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy',
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall')]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
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
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\n✓ SAAE-DNN training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, run_dir, dataset_name='Test'):
    """Evaluate model and save results"""
    print(f"\n{'='*80}")
    print(f"Evaluation on {dataset_name}")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    
    # Handle binary vs multi-class
    if len(y_pred_probs.shape) == 1 or y_pred_probs.shape[1] == 1:
        # Binary classification
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Calculate metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, classification_report, confusion_matrix)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
    recall = recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
    f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
    
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
    target_names = ['Normal', 'Attack'] if len(np.unique(y_true)) == 2 else \
                   ['Normal', 'Probe', 'DoS', 'U2R', 'R2L']
    
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
        description='Train SAAE-DNN model (paper implementation)'
    )
    
    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data file')
    parser.add_argument('--test_21_path', type=str, default=None,
                       help='Path to KDDTest-21 file')
    
    # Model architecture (from paper)
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[90, 80],
                       help='SAAE latent dimensions (default: 90 80)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[50, 25, 10],
                       help='DNN hidden dimensions (default: 50 25 10)')
    
    # Classification type
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification (Normal vs Attack)')
    parser.add_argument('--multiclass', action='store_true',
                       help='Multi-class classification (5 classes)')
    
    # Training parameters (from paper Section 5.4)
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='SAAE pre-training epochs (default: 100)')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='SAAE-DNN training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--pretrain_lr', type=float, default=0.05,
                       help='SAAE pre-training learning rate (default: 0.05)')
    parser.add_argument('--train_lr', type=float, default=0.006,
                       help='SAAE-DNN training learning rate (default: 0.006)')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--l1', type=float, default=1e-5,
                       help='L1 regularization (default: 1e-5)')
    parser.add_argument('--l2', type=float, default=1e-4,
                       help='L2 regularization (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/saae_runs',
                       help='Output directory')
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip SAAE pre-training (use random initialization)')
    
    args = parser.parse_args()
    
    # Default to binary if neither specified
    if not args.binary and not args.multiclass:
        args.binary = True
    
    print("\n" + "="*80)
    print("SAAE-DNN: Intrusion Detection System")
    print("Paper: Tang et al., 2020 - Symmetry")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✓ GPU enabled: {len(gpus)} device(s)")
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Create output directory
    run_num = 1
    while (Path(args.output_dir) / f'run_saae_{run_num:03d}').exists():
        run_num += 1
    
    run_dir = Path(args.output_dir) / f'run_saae_{run_num:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {run_dir}\n")
    
    # Load data
    print("="*80)
    print("Loading Data")
    print("="*80)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
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
    
    def calculate_zero_percentage(df, numeric_columns):
        """
        Calculate percentage of zeros for each numeric column.
        Used for SAAE-DNN statistical filtering (Paper Section 4.1.3).
        """
        zero_percentages = {}
        for col in numeric_columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / len(df)) * 100
            zero_percentages[col] = zero_pct
        return zero_percentages
    
    def preprocess_nsl_kdd(df, scaler=None, feature_columns=None, features_to_keep=None):
        """
        Preprocess NSL-KDD data following SAAE-DNN paper methodology.
        
        Paper Section 4.1.3: Statistical Filtering
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
            print(f"  Step 4: Statistical Filtering (SAAE-DNN Paper Section 4.1.3)...")
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
            print(f"      Expected: 102 features (18 numeric + 84 one-hot)")
        else:
            # Align with training columns
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]
        
        X = X.values.astype(np.float32)
        
        # Step 5: StandardScaler normalization
        print(f"  Step 5: MinMaxScaler normalization [0, 1]...")
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        
        return X, y, scaler, feature_columns, features_to_keep
    
    # Load training data
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
    if args.test_path:
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
    
    # Determine dimensions
    input_dim = X_train.shape[1]
    
    if args.binary:
        print("\nClassification type: Binary (Normal vs Attack)")
        num_classes = 2
    else:
        print("\nClassification type: Multi-class (Normal, Probe, DoS, U2R, R2L)")
        num_classes = 5
        # Convert to categorical if multi-class
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Feature dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    
    # Step 1: Build and pre-train SAAE
    if not args.skip_pretrain:
        print("\n" + "="*80)
        print("Step 1: SAAE Pre-training")
        print("="*80)
        print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, args.latent_dims))} -> ... -> {input_dim}")
        print(f"  Pre-training epochs: {args.pretrain_epochs}")
        print(f"  Learning rate: {args.pretrain_lr}")
        
        encoders, decoders, autoencoders = build_stacked_attention_autoencoder(
            input_dim=input_dim,
            latent_dims=args.latent_dims,
            activation='relu'
        )
        
        pretrain_histories = pretrain_saae(
            encoders, decoders, autoencoders,
            X_train,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.pretrain_lr,
            verbose=1
        )
    else:
        print("\n⚠ Skipping SAAE pre-training (using random initialization)")
        encoders = None
    
    # Step 2: Build SAAE-DNN classifier
    print("\n" + "="*80)
    print("Step 2: Building SAAE-DNN Classifier")
    print("="*80)
    print(f"  SAAE layers: {input_dim} -> {' -> '.join(map(str, args.latent_dims))}")
    print(f"  DNN layers: {args.latent_dims[-1]} -> {' -> '.join(map(str, args.hidden_dims))} -> {num_classes}")
    
    saae_dnn_model = build_saae_dnn_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dims=args.latent_dims,
        hidden_dims=args.hidden_dims,
        activation='relu',
        l1=args.l1,
        l2=args.l2,
        dropout=args.dropout
    )
    
    print(f"  Total parameters: {saae_dnn_model.count_params():,}")
    
    # Step 3: Transfer pre-trained weights
    if not args.skip_pretrain:
        transfer_saae_weights(saae_dnn_model, encoders)
    
    # Step 4: Train SAAE-DNN
    print("\n" + "="*80)
    print("Step 3: Training SAAE-DNN Classifier")
    print("="*80)
    print(f"  Training epochs: {args.train_epochs}")
    print(f"  Learning rate: {args.train_lr}")
    print(f"  Batch size: {args.batch_size}")
    
    history = train_saae_dnn(
        saae_dnn_model,
        X_train, y_train,
        X_val, y_val,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.train_lr,
        verbose=1,
        run_dir=run_dir
    )
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    plot_training_curves(history, run_dir / 'training_curves.png')
    
    # Step 5: Evaluate on test sets
    print("\n" + "="*80)
    print("Step 4: Evaluation")
    print("="*80)
    
    # Evaluate on KDDTest+
    metrics_test = evaluate_model(
        saae_dnn_model, X_test, y_test, run_dir, 'KDDTest+'
    )
    
    # Evaluate on KDDTest-21 if available
    if args.test_21_path:
        print("\nEvaluating on KDDTest-21...")
        # Load KDDTest-21 data
        # X_test_21, y_test_21 = load_test_21_data(args.test_21_path)
        # metrics_test_21 = evaluate_model(
        #     saae_dnn_model, X_test_21, y_test_21, run_dir, 'KDDTest-21'
        # )
    
    # Save final model
    saae_dnn_model.save(run_dir / 'saae_dnn_final.keras')
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()