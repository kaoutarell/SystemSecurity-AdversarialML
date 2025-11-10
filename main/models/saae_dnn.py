#!/usr/bin/env python3
"""
SAAE-DNN: Stacked Attention AutoEncoder with Deep Neural Network

Implementation based on the paper:
"SAAE-DNN: Deep Learning Method on Intrusion Detection"
by Chaofei Tang, Nurbol Luktarhan, and Yuxin Zhao (2020)

Key architecture:
- Stacked AutoEncoder (SAE) for feature extraction
- Attention Mechanism to highlight key features
- DNN classifier initialized with SAAE encoder weights
- Two-stage training: SAAE pretraining + DNN fine-tuning

Paper results on NSL-KDD:
- Binary: 87.74% (KDDTest+), 82.97% (KDDTest-21)
- Multi-class: 82.14% (KDDTest+), 77.57% (KDDTest-21)
"""

import tensorflow as tf
import numpy as np


class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention Mechanism Layer.
    
    Based on paper equations (5-7):
    - M = tanh(Wa * x' + ba)
    - α_i = softmax(M_i)
    - v = Σ(x' * α_i^T)
    
    Learns to weight features by importance.
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


class SAAEDNNModel:
    """
    Stacked Attention AutoEncoder with DNN Classifier.
    
    Architecture from paper:
    - SAAE: Two-layer stacked attention autoencoder [input -> 90 -> 80]
    - DNN: Three-layer classifier [80 -> 50 -> 25 -> 10 -> output]
    - Two-stage training: Greedy layer-wise pretraining + supervised fine-tuning
    
    Best hyperparameters (80.70% accuracy):
    - batch_size: 256
    - pretrain_lr: 0.05 (for autoencoder)
    - train_lr: 0.006 (for classifier)
    - dropout: 0.3
    - l1: 1e-5
    - l2: 1e-4
    - latent_dims: [90, 80]
    - hidden_dims: [50, 25, 10]
    """
    
    def __init__(self, input_dim, num_classes=2, latent_dims=[90, 80],
                 hidden_dims=[50, 25, 10], activation='relu',
                 l1=1e-5, l2=1e-4, dropout=0.3):
        """
        Initialize SAAE-DNN model parameters.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes (2 for binary)
            latent_dims: SAAE latent dimensions [90, 80]
            hidden_dims: DNN hidden layer dimensions [50, 25, 10]
            activation: Activation function
            l1: L1 regularization coefficient
            l2: L2 regularization coefficient
            dropout: Dropout rate
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        
        # Will store pretrained components
        self.encoders = []
        self.decoders = []
        self.autoencoders = []
    
    def build_autoencoder(self, input_dim, latent_dim, layer_num):
        """
        Build a single Attention AutoEncoder (AAE).
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            layer_num: Layer number for naming
        
        Returns:
            encoder, decoder, autoencoder models
        """
        # Encoder
        encoder_input = tf.keras.Input(shape=(input_dim,), name=f'encoder_input_{layer_num}')
        encoded = tf.keras.layers.Dense(
            latent_dim,
            activation=self.activation,
            kernel_initializer='he_normal',
            name=f'encoder_dense_{layer_num}'
        )(encoder_input)
        attention = AttentionLayer(name=f'attention_{layer_num}')(encoded)
        encoder = tf.keras.Model(encoder_input, attention, name=f'encoder_{layer_num}')
        
        # Decoder
        decoder_input = tf.keras.Input(shape=(latent_dim,), name=f'decoder_input_{layer_num}')
        decoded = tf.keras.layers.Dense(
            input_dim,
            activation=self.activation,
            kernel_initializer='he_normal',
            name=f'decoder_dense_{layer_num}'
        )(decoder_input)
        decoder = tf.keras.Model(decoder_input, decoded, name=f'decoder_{layer_num}')
        
        # Full autoencoder
        autoencoder_input = tf.keras.Input(shape=(input_dim,), name=f'ae_input_{layer_num}')
        encoded_out = encoder(autoencoder_input)
        decoded_out = decoder(encoded_out)
        autoencoder = tf.keras.Model(
            autoencoder_input,
            decoded_out,
            name=f'autoencoder_{layer_num}'
        )
        
        return encoder, decoder, autoencoder
    
    def build_stacked_autoencoder(self):
        """
        Build Stacked Attention AutoEncoder (SAAE).
        
        Greedy layer-wise pretraining:
        - Layer 1: input_dim -> latent_dims[0]
        - Layer 2: latent_dims[0] -> latent_dims[1]
        
        Returns:
            Lists of encoders, decoders, autoencoders
        """
        encoders = []
        decoders = []
        autoencoders = []
        
        current_dim = self.input_dim
        
        for i, latent_dim in enumerate(self.latent_dims):
            encoder, decoder, autoencoder = self.build_autoencoder(
                current_dim, latent_dim, i+1
            )
            encoders.append(encoder)
            decoders.append(decoder)
            autoencoders.append(autoencoder)
            current_dim = latent_dim
        
        self.encoders = encoders
        self.decoders = decoders
        self.autoencoders = autoencoders
        
        return encoders, decoders, autoencoders
    
    def build_classifier(self):
        """
        Build SAAE-DNN classifier.
        
        Architecture:
        - SAAE encoder layers (will use pretrained weights)
        - DNN hidden layers with dropout
        - Output layer (sigmoid for binary, softmax for multi-class)
        
        Returns:
            Uncompiled Keras model
        """
        inputs = tf.keras.Input(shape=(self.input_dim,), name='saae_dnn_input')
        x = inputs
        
        # SAAE encoder layers (will be initialized with pretrained weights)
        for i, latent_dim in enumerate(self.latent_dims):
            x = tf.keras.layers.Dense(
                latent_dim,
                activation=self.activation,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                name=f'saae_encoder_{i+1}'
            )(x)
            x = AttentionLayer(name=f'saae_attention_{i+1}')(x)
        
        # DNN classification layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = tf.keras.layers.Dense(
                hidden_dim,
                activation=self.activation,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                name=f'dnn_hidden_{i+1}'
            )(x)
            x = tf.keras.layers.Dropout(self.dropout, name=f'dnn_dropout_{i+1}')(x)
        
        # Output layer
        if self.num_classes == 2:
            # Binary classification
            outputs = tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1*0.1, l2=self.l2*0.1),
                name='output'
            )(x)
        else:
            # Multi-class classification
            outputs = tf.keras.layers.Dense(
                self.num_classes,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1*0.1, l2=self.l2*0.1),
                name='output'
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='saae_dnn')
        
        return model
    
    def pretrain_autoencoder(self, autoencoder, X_train, epochs=100,
                            batch_size=256, learning_rate=0.05, verbose=1):
        """
        Pretrain a single autoencoder.
        
        Args:
            autoencoder: Autoencoder model
            X_train: Training data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        print(f"\nPretraining {autoencoder.name}...")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        autoencoder.compile(
            optimizer=optimizer,
            loss='mse',  # Reconstruction error
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
    
    def pretrain_saae(self, X_train, epochs=100, batch_size=256,
                      learning_rate=0.05, verbose=1):
        """
        Pretrain SAAE layer by layer (greedy layer-wise pretraining).
        
        Args:
            X_train: Training data
            epochs: Number of epochs per layer
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
        
        Returns:
            List of training histories
        """
        print("\n" + "="*80)
        print("SAAE Pretraining (Greedy Layer-wise)")
        print("="*80)
        
        if not self.autoencoders:
            self.build_stacked_autoencoder()
        
        X_current = X_train
        histories = []
        
        for i, (encoder, decoder, autoencoder) in enumerate(
            zip(self.encoders, self.decoders, self.autoencoders)
        ):
            print(f"\nLayer {i+1}/{len(self.autoencoders)}")
            print(f"  Input dim: {X_current.shape[1]}")
            print(f"  Latent dim: {encoder.output_shape[1]}")
            
            # Train this autoencoder
            history = self.pretrain_autoencoder(
                autoencoder, X_current, epochs, batch_size, learning_rate, verbose
            )
            histories.append(history)
            
            # Use encoder output as input to next layer
            X_current = encoder.predict(X_current, verbose=0)
            
            print(f"  ✓ Layer {i+1} pretraining complete")
            print(f"  Final loss: {history.history['loss'][-1]:.6f}")
        
        print("\n" + "="*80)
        print("✓ SAAE Pretraining Complete")
        print("="*80)
        
        return histories
    
    def transfer_weights(self, classifier_model):
        """
        Transfer pretrained SAAE weights to classifier.
        
        Args:
            classifier_model: SAAE-DNN classifier model
        """
        print("\nTransferring pretrained SAAE weights to classifier...")
        
        if not self.encoders:
            print("⚠ No pretrained weights available. Skipping transfer.")
            return
        
        # Transfer encoder weights for each layer
        for i, encoder in enumerate(self.encoders):
            layer_num = i + 1
            
            # Transfer encoder dense weights
            encoder_weights = encoder.get_layer(f'encoder_dense_{layer_num}').get_weights()
            classifier_model.get_layer(f'saae_encoder_{layer_num}').set_weights(encoder_weights)
            
            # Transfer attention weights
            attention_weights = encoder.get_layer(f'attention_{layer_num}').get_weights()
            classifier_model.get_layer(f'saae_attention_{layer_num}').set_weights(attention_weights)
        
        print("✓ Weight transfer complete")
    
    def compile_classifier(self, model, learning_rate=0.006):
        """
        Compile the classifier model.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate
        
        Returns:
            Compiled model
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        else:
            loss = 'categorical_crossentropy'
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def build_and_compile(self, learning_rate=0.006):
        """
        Build and compile the classifier (without pretraining).
        
        Args:
            learning_rate: Learning rate
        
        Returns:
            Compiled classifier model
        """
        # Build stacked autoencoder structure (for weight initialization)
        # Only build if not already built (to preserve pretrained weights)
        if not self.autoencoders:
            self.build_stacked_autoencoder()
        
        # Build classifier
        classifier = self.build_classifier()
        
        # Compile
        classifier = self.compile_classifier(classifier, learning_rate)
        
        return classifier


def create_saae_dnn_model(input_dim, num_classes=2, latent_dims=[90, 80],
                          hidden_dims=[50, 25, 10], l1=1e-5, l2=1e-4,
                          dropout=0.3, learning_rate=0.006,
                          pretrain=False, X_train=None, pretrain_epochs=100,
                          pretrain_lr=0.05, pretrain_batch_size=256):
    """
    Factory function to create a compiled SAAE-DNN model.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        latent_dims: SAAE latent dimensions
        hidden_dims: DNN hidden dimensions
        l1: L1 regularization
        l2: L2 regularization
        dropout: Dropout rate
        learning_rate: Learning rate for classifier training
        pretrain: Whether to pretrain SAAE
        X_train: Training data for pretraining (required if pretrain=True)
        pretrain_epochs: Number of pretraining epochs
        pretrain_lr: Learning rate for pretraining
        pretrain_batch_size: Batch size for pretraining
    
    Returns:
        Compiled SAAE-DNN model (and pretrain histories if pretrain=True)
    """
    saae_dnn = SAAEDNNModel(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dims=latent_dims,
        hidden_dims=hidden_dims,
        l1=l1,
        l2=l2,
        dropout=dropout
    )
    
    # Optionally pretrain SAAE
    pretrain_histories = None
    if pretrain:
        if X_train is None:
            raise ValueError("X_train is required for SAAE pretraining")
        
        pretrain_histories = saae_dnn.pretrain_saae(
            X_train,
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            learning_rate=pretrain_lr,
            verbose=1
        )
    
    # Build and compile classifier
    classifier = saae_dnn.build_and_compile(learning_rate=learning_rate)
    
    # Transfer pretrained weights if available
    if pretrain:
        saae_dnn.transfer_weights(classifier)
    
    if pretrain:
        return classifier, pretrain_histories
    else:
        return classifier
