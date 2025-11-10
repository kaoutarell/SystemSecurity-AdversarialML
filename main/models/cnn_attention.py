#!/usr/bin/env python3
"""
CNN with Efficient Channel Attention (ECA) IDS Model

Convolutional Neural Network with attention mechanism for intrusion detection.
Reshapes tabular data into 2D images and applies CNN spatial feature extraction
combined with ECA channel attention.

Based on: Alrayes et al., 2024 methodology
"""

import tensorflow as tf


class ECABlock(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) Block.
    
    Simplified attention without BatchNorm - better for NSL-KDD.
    Learns channel-wise importance weights through:
    1. Global Average Pooling
    2. 1D Convolution for attention
    3. Sigmoid activation for weights
    4. Element-wise multiplication
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


class CNNAttentionModel:
    """
    CNN with ECA Attention for Intrusion Detection.
    
    Architecture:
    - Input: (image_size, image_size, 1) 2D images
    - Conv2D + ECA Attention + MaxPooling (2 blocks)
    - Global Average Pooling
    - Dense classification head
    - NO BatchNormalization (NSL-KDD distribution mismatch)
    - L1/L2 regularization + Dropout
    
    Best hyperparameters (78.18% accuracy):
    - batch_size: 2048 (CRITICAL!)
    - learning_rate: 0.001
    - dropout: 0.4
    - l1: 1e-5
    - l2: 5e-4
    - image_size: 12 (12x12 = 144 features)
    """
    
    def __init__(self, image_size=12, dropout=0.4, l1=1e-5, l2=5e-4):
        """
        Initialize CNN Attention model parameters.
        
        Args:
            image_size: Size of square image (12x12 = 144 features)
            dropout: Dropout rate for regularization
            l1: L1 regularization coefficient
            l2: L2 regularization coefficient
        """
        self.image_size = image_size
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2
    
    def build(self):
        """
        Build the CNN Attention model.
        
        Returns:
            Uncompiled Keras model
        """
        # Input: 2D images
        inputs = tf.keras.Input(
            shape=(self.image_size, self.image_size, 1),
            name='input_images'
        )
        x = inputs
        
        # First Conv Block + ECA (NO BatchNorm!)
        x = tf.keras.layers.Conv2D(
            32, (3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
            kernel_initializer='he_normal',
            name='conv1'
        )(x)
        x = ECABlock(kernel_size=3, name='eca1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = tf.keras.layers.Dropout(self.dropout * 0.5, name='dropout1')(x)
        
        # Second Conv Block + ECA
        x = tf.keras.layers.Conv2D(
            64, (3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
            kernel_initializer='he_normal',
            name='conv2'
        )(x)
        x = ECABlock(kernel_size=3, name='eca2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = tf.keras.layers.Dropout(self.dropout * 0.7, name='dropout2')(x)
        
        # Global Average Pooling (reduces overfitting)
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        
        # Dense classification head
        x = tf.keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
            kernel_initializer='he_normal',
            name='dense1'
        )(x)
        x = tf.keras.layers.Dropout(self.dropout, name='dropout3')(x)
        
        # Output (binary classification)
        outputs = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1*0.1, l2=self.l2*0.1),
            name='output'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_eca_ids')
        
        return model
    
    def compile(self, model, learning_rate=0.001):
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
    
    def build_and_compile(self, learning_rate=0.001):
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


def create_cnn_attention_model(image_size=12, dropout=0.4, l1=1e-5, l2=5e-4,
                               learning_rate=0.001):
    """
    Factory function to create a compiled CNN Attention model.
    
    Args:
        image_size: Size of square image (12x12 = 144 features)
        dropout: Dropout rate for regularization
        l1: L1 regularization coefficient
        l2: L2 regularization coefficient
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model ready for training
    """
    cnn = CNNAttentionModel(
        image_size=image_size,
        dropout=dropout,
        l1=l1,
        l2=l2
    )
    return cnn.build_and_compile(learning_rate=learning_rate)
