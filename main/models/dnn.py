#!/usr/bin/env python3
"""
Dense Neural Network (DNN) IDS Model

A standard feedforward neural network for intrusion detection.
Uses statistical feature filtering and proven regularization techniques.
"""

import tensorflow as tf


class DNNModel:
    """
    Dense Neural Network for Intrusion Detection.
    
    Architecture:
    - Input layer
    - Multiple dense hidden layers with ReLU activation
    - Optional batch normalization
    - Dropout regularization
    - L1/L2 weight regularization
    - Binary classification output (sigmoid)
    
    Best hyperparameters (77.92% accuracy):
    - batch_size: 256
    - learning_rate: 0.006
    - dropout: 0.3
    - l1: 1e-5
    - l2: 1e-4
    """
    
    def __init__(self, input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4, 
                 use_batchnorm=False):
        """
        Initialize DNN model parameters.
        
        Args:
            input_dim: Number of input features
            dropout_rate: Dropout rate for regularization
            l1: L1 regularization coefficient
            l2: L2 regularization coefficient
            use_batchnorm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.use_batchnorm = use_batchnorm
    
    def build(self):
        """
        Build the DNN model.
        
        Returns:
            Uncompiled Keras model
        """
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input')
        x = inputs
        
        # Hidden layers with progressively smaller dimensions
        hidden_dims = [128, 64, 32]
        
        for i, units in enumerate(hidden_dims):
            x = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                kernel_initializer='he_normal',
                name=f'dense_{i+1}'
            )(x)
            
            if self.use_batchnorm:
                x = tf.keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            x = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'dropout_{i+1}'
            )(x)
        
        # Output layer (binary classification)
        outputs = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1*0.1, l2=self.l2*0.1),
            name='output'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='dnn_ids')
        
        return model
    
    def compile(self, model, learning_rate=0.006):
        """
        Compile the model with optimizer and metrics.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for Adam optimizer
        
        Returns:
            Compiled Keras model
        """
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
    
    def build_and_compile(self, learning_rate=0.006):
        """
        Build and compile the model in one step.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        
        Returns:
            Compiled Keras model ready for training
        """
        model = self.build()
        model = self.compile(model, learning_rate=learning_rate)
        return model


def create_dnn_model(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4,
                     use_batchnorm=False, learning_rate=0.006):
    """
    Factory function to create a compiled DNN model.
    
    Args:
        input_dim: Number of input features
        dropout_rate: Dropout rate for regularization
        l1: L1 regularization coefficient
        l2: L2 regularization coefficient
        use_batchnorm: Whether to use batch normalization
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model ready for training
    """
    dnn = DNNModel(
        input_dim=input_dim,
        dropout_rate=dropout_rate,
        l1=l1,
        l2=l2,
        use_batchnorm=use_batchnorm
    )
    return dnn.build_and_compile(learning_rate=learning_rate)
