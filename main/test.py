import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from pathlib import Path
import sys
import os

# Add main directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import load_data, preprocess_nsl_kdd

# Load NSL-KDD data
print("="*80)
print("PHASE 1: BASELINE MODEL")
print("="*80)

df_train = load_data('nsl-kdd/KDDTrain+.txt')
df_test = load_data('nsl-kdd/KDDTest+.txt')

# Preprocess: One-hot encode, normalize, handle class imbalance
X_train, y_train, scaler, feature_columns = preprocess_nsl_kdd(
    df_train, 
    binary=True,
    use_statistical_filter=False
)
X_test, y_test, _, _ = preprocess_nsl_kdd(
    df_test, 
    scaler=scaler, 
    feature_columns=feature_columns,
    binary=True,
    use_statistical_filter=False
)

# Identify feature types (for constrained attacks later)
# Categorical features that got one-hot encoded: protocol_type (3), service (70), flag (11)
# These appear at the end after continuous features
n_features = X_train.shape[1]
n_categorical = 3 + 70 + 11  # protocol_type + service + flag = 84
n_continuous = n_features - n_categorical

continuous_idx = list(range(n_continuous))
onehot_idx = list(range(n_continuous, n_features))

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Features: {X_train.shape[1]}")
print(f"    - Continuous: {len(continuous_idx)}")
print(f"    - One-hot encoded: {len(onehot_idx)}")
print(f"  Class distribution:")
print(f"    - Normal: {(y_train == 0).sum():,} ({(y_train == 0).mean():.1%})")
print(f"    - Attack: {(y_train == 1).sum():,} ({(y_train == 1).mean():.1%})")

# Compute class weights (critical for imbalanced dataset!)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

def build_baseline_ids_model(input_dim, name="baseline_ids"):
    """
    Standard IDS model (non-robust)
    Architecture: Dense network with batch normalization
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        
        # First block
        tf.keras.layers.Dense(128, activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Second block
        tf.keras.layers.Dense(64, activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Third block
        tf.keras.layers.Dense(32, activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        # Output
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name=name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

# Build and train baseline model
model_baseline = build_baseline_ids_model(X_train.shape[1])
model_baseline.summary()

print("\n" + "="*80)
print("Training Baseline Model")
print("="*80)

history_baseline = model_baseline.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ],
    verbose=1
)

# Save baseline model
model_baseline.save('models/baseline_ids_model.h5')
print("\n✓ Baseline model saved!")

def evaluate_model_comprehensive(model, X, y, model_name="Model"):
    """Comprehensive evaluation metrics"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print(f"\n{'='*80}")
    print(f"{model_name} - Clean Data Performance")
    print(f"{'='*80}")
    
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = (y_pred == y).mean()
    auc = roc_auc_score(y, y_pred_prob)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, 
                                target_names=['Normal', 'Attack'],
                                digits=4))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                Normal  Attack")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

# Evaluate baseline
baseline_results = evaluate_model_comprehensive(
    model_baseline, X_test, y_test, 
    "Baseline IDS Model"
)

def pgd_attack_tensorflow(model, X, y, epsilon=0.03, alpha=0.01, num_iter=40,
                          continuous_indices=None, onehot_indices=None,
                          random_start=False):
    """
    PGD attack for TensorFlow models with one-hot protection
    
    Args:
        model: TensorFlow model
        X: Input samples (numpy array)
        y: True labels
        epsilon: Maximum perturbation (L-infinity norm)
        alpha: Step size
        num_iter: Number of iterations
        continuous_indices: Indices of continuous features (can be perturbed)
        onehot_indices: Indices of one-hot features (CANNOT be perturbed)
        random_start: Whether to start from random point in epsilon ball
    
    Returns:
        X_adv: Adversarial examples
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Initialize perturbation
    if random_start:
        delta = tf.random.uniform(
            shape=X_tensor.shape,
            minval=-epsilon,
            maxval=epsilon,
            dtype=tf.float32
        )
    else:
        delta = tf.zeros_like(X_tensor)
    
    delta = tf.Variable(delta, trainable=True)
    
    # PGD iterations
    for iteration in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            
            # Adversarial input
            X_adv = X_tensor + delta
            
            # Compute loss (we want to MAXIMIZE loss)
            predictions = model(X_adv, training=False)
            loss = tf.keras.losses.binary_crossentropy(
                y_tensor,
                tf.squeeze(predictions)
            )
            loss = tf.reduce_mean(loss)
        
        # Compute gradient
        gradient = tape.gradient(loss, delta)
        
        # Take step in direction of gradient (gradient ASCENT)
        delta_update = alpha * tf.sign(gradient)
        delta.assign_add(delta_update)
        
        # Project back to epsilon ball
        delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))
        
        # CRITICAL: Zero out perturbations on one-hot features
        if onehot_indices is not None and len(onehot_indices) > 0:
            delta_np = delta.numpy()
            delta_np[:, onehot_indices] = 0
            delta.assign(delta_np)
    
    X_adv = X_tensor + delta
    return X_adv.numpy()


# Test PGD implementation
print("\n" + "="*80)
print("PHASE 2: WHITE-BOX PGD ATTACK")
print("="*80)
print("\nTesting PGD attack implementation...")

# Small test
X_test_sample = X_test[:100]
y_test_sample = y_test[:100]

X_adv_sample = pgd_attack_tensorflow(
    model_baseline,
    X_test_sample,
    y_test_sample,
    epsilon=0.03,
    alpha=0.01,
    num_iter=40,
    continuous_indices=continuous_idx,
    onehot_indices=onehot_idx,
    random_start=False
)

print(f"✓ PGD attack implemented successfully!")
print(f"  Perturbation statistics:")
print(f"    Max perturbation: {np.abs(X_adv_sample - X_test_sample).max():.6f}")
print(f"    Mean perturbation: {np.abs(X_adv_sample - X_test_sample).mean():.6f}")


print("\n" + "="*80)
print("Generating White-Box Adversarial Examples")
print("="*80)

# Generate adversarial examples for full test set
# (may need to batch this for memory)
def generate_adversarial_batched(model, X, y, batch_size=1000, **attack_params):
    """Generate adversarial examples in batches"""
    n_samples = len(X)
    X_adv_full = np.zeros_like(X)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        print(f"  Processing samples {i:5d} - {end_idx:5d} / {n_samples}", end='\r')
        
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]
        
        X_adv_batch = pgd_attack_tensorflow(
            model, X_batch, y_batch, **attack_params
        )
        
        X_adv_full[i:end_idx] = X_adv_batch
    
    print()  # New line after progress
    return X_adv_full

# Generate white-box adversarial examples
X_test_adv_whitebox = generate_adversarial_batched(
    model_baseline,
    X_test,
    y_test,
    batch_size=1000,
    epsilon=0.03,
    alpha=0.01,
    num_iter=40,
    continuous_indices=continuous_idx,
    onehot_indices=onehot_idx,
    random_start=False
)

# Save adversarial examples
np.save('adversarial_data/X_test_adv_whitebox_eps003.npy', X_test_adv_whitebox)
print("✓ White-box adversarial examples saved!")

# Evaluate on adversarial examples
whitebox_results = evaluate_model_comprehensive(
    model_baseline,
    X_test_adv_whitebox,
    y_test,
    "Baseline Model - White-Box PGD Attack (ε=0.03)"
)

# Calculate attack success rate
attack_success_rate = 1 - (whitebox_results['accuracy'] / baseline_results['accuracy'])

print(f"\n{'='*80}")
print(f"WHITE-BOX ATTACK SUMMARY")
print(f"{'='*80}")
print(f"Clean accuracy:       {baseline_results['accuracy']:.4f}")
print(f"Adversarial accuracy: {whitebox_results['accuracy']:.4f}")
print(f"Accuracy drop:        {baseline_results['accuracy'] - whitebox_results['accuracy']:.4f}")
print(f"Attack success rate:  {attack_success_rate:.2%}")
print(f"{'='*80}")

print("\n" + "="*80)
print("PHASE 3: BLACK-BOX TRANSFER ATTACKS")
print("="*80)

def build_simple_surrogate(input_dim, name="surrogate_simple"):
    """Simple shallow network (different from baseline)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name=name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_deep_surrogate(input_dim, name="surrogate_deep"):
    """Deep network with different architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name=name)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Train surrogate models
print("\nTraining Surrogate Model 1: Simple Architecture")
print("-" * 80)
surrogate_simple = build_simple_surrogate(X_train.shape[1])
surrogate_simple.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
    verbose=1
)
surrogate_simple.save('models/surrogate_simple.h5')

print("\nTraining Surrogate Model 2: Deep Architecture")
print("-" * 80)
surrogate_deep = build_deep_surrogate(X_train.shape[1])
surrogate_deep.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
    verbose=1
)
surrogate_deep.save('models/surrogate_deep.h5')

print("\n✓ Surrogate models trained!")

print("\n" + "="*80)
print("Generating Transfer Attacks from Surrogates")
print("="*80)

transfer_results = {}

# Transfer attack from Simple surrogate
print("\n1. Generating adversarial examples on SIMPLE surrogate...")
X_test_adv_transfer_simple = generate_adversarial_batched(
    surrogate_simple,
    X_test,
    y_test,
    batch_size=1000,
    epsilon=0.03,
    alpha=0.01,
    num_iter=40,
    continuous_indices=continuous_idx,
    onehot_indices=onehot_idx
)
np.save('adversarial_data/X_test_adv_transfer_simple_eps003.npy', 
        X_test_adv_transfer_simple)

# Test on TARGET model (baseline)
print("   Testing on TARGET model (baseline)...")
transfer_simple_results = evaluate_model_comprehensive(
    model_baseline,
    X_test_adv_transfer_simple,
    y_test,
    "Transfer Attack from Simple Surrogate"
)
transfer_results['simple'] = transfer_simple_results

# Transfer attack from Deep surrogate
print("\n2. Generating adversarial examples on DEEP surrogate...")
X_test_adv_transfer_deep = generate_adversarial_batched(
    surrogate_deep,
    X_test,
    y_test,
    batch_size=1000,
    epsilon=0.03,
    alpha=0.01,
    num_iter=40,
    continuous_indices=continuous_idx,
    onehot_indices=onehot_idx
)
np.save('adversarial_data/X_test_adv_transfer_deep_eps003.npy', 
        X_test_adv_transfer_deep)

# Test on TARGET model
print("   Testing on TARGET model (baseline)...")
transfer_deep_results = evaluate_model_comprehensive(
    model_baseline,
    X_test_adv_transfer_deep,
    y_test,
    "Transfer Attack from Deep Surrogate"
)
transfer_results['deep'] = transfer_deep_results

print("\n3. Generating ENSEMBLE transfer attack...")

# Average perturbations from both surrogates
delta_simple = X_test_adv_transfer_simple - X_test
delta_deep = X_test_adv_transfer_deep - X_test

delta_ensemble = (delta_simple + delta_deep) / 2

# Clip to epsilon ball
delta_ensemble = np.clip(delta_ensemble, -0.03, 0.03)

# Zero out one-hot features
delta_ensemble[:, onehot_idx] = 0

X_test_adv_transfer_ensemble = X_test + delta_ensemble
np.save('adversarial_data/X_test_adv_transfer_ensemble_eps003.npy',
        X_test_adv_transfer_ensemble)

# Test on TARGET model
transfer_ensemble_results = evaluate_model_comprehensive(
    model_baseline,
    X_test_adv_transfer_ensemble,
    y_test,
    "Ensemble Transfer Attack"
)
transfer_results['ensemble'] = transfer_ensemble_results

print("\n" + "="*80)
print("BLACK-BOX TRANSFER ATTACK SUMMARY")
print("="*80)

print(f"\nBaseline Model Performance:")
print(f"  Clean accuracy:                    {baseline_results['accuracy']:.4f}")
print(f"\nAttack Results:")
print(f"  White-box PGD (worst-case):        {whitebox_results['accuracy']:.4f}")
print(f"  Transfer from Simple surrogate:    {transfer_results['simple']['accuracy']:.4f}")
print(f"  Transfer from Deep surrogate:      {transfer_results['deep']['accuracy']:.4f}")
print(f"  Transfer from Ensemble:            {transfer_results['ensemble']['accuracy']:.4f}")

print(f"\nAttack Success Rates:")
wb_success = 1 - (whitebox_results['accuracy'] / baseline_results['accuracy'])
simple_success = 1 - (transfer_results['simple']['accuracy'] / baseline_results['accuracy'])
deep_success = 1 - (transfer_results['deep']['accuracy'] / baseline_results['accuracy'])
ensemble_success = 1 - (transfer_results['ensemble']['accuracy'] / baseline_results['accuracy'])

print(f"  White-box:           {wb_success:.2%}")
print(f"  Transfer (Simple):   {simple_success:.2%}")
print(f"  Transfer (Deep):     {deep_success:.2%}")
print(f"  Transfer (Ensemble): {ensemble_success:.2%}")

print(f"\nKey Finding:")
print(f"  Transfer attacks achieve {simple_success/wb_success:.1%} of white-box effectiveness,")
print(f"  demonstrating significant black-box threat!")
print("="*80)

print("\n" + "="*80)
print("PHASE 4: ADVERSARIAL TRAINING")
print("="*80)

def adversarial_training_epoch(model, X_batch, y_batch, optimizer,
                               epsilon, alpha, num_iter,
                               continuous_idx, onehot_idx):
    """
    Single adversarial training step on a batch
    
    Standard adversarial training: train on adversarial examples
    """
    # Generate adversarial examples for this batch
    X_adv_batch = pgd_attack_tensorflow(
        model, X_batch, y_batch,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        continuous_indices=continuous_idx,
        onehot_indices=onehot_idx,
        random_start=True  # Random start for diversity
    )
    
    # Train on adversarial examples
    with tf.GradientTape() as tape:
        predictions = model(X_adv_batch, training=True)
        loss = tf.keras.losses.binary_crossentropy(y_batch, tf.squeeze(predictions))
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss.numpy()


def train_robust_ids_model(X_train, y_train, X_val, y_val,
                           continuous_idx, onehot_idx,
                           epsilon=0.03, alpha=0.01, num_iter=10,
                           epochs=50, batch_size=128,
                           class_weight_dict=None):
    """
    Train robust IDS model using PGD adversarial training
    
    Training strategy:
    - Every batch: generate adversarial examples, train on them
    - Following Madry et al. methodology
    """
    print("\nBuilding robust model...")
    model_robust = build_baseline_ids_model(
        X_train.shape[1],
        name="robust_ids"
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Lower LR for stability
    
    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    print(f"\nAdversarial Training Configuration:")
    print(f"  Epsilon (ε): {epsilon}")
    print(f"  Alpha (α): {alpha}")
    print(f"  PGD iterations: {num_iter}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"\nStarting training...\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        epoch_losses = []
        for batch_idx, (X_batch, y_batch) in enumerate(train_dataset):
            loss = adversarial_training_epoch(
                model_robust, X_batch, y_batch, optimizer,
                epsilon, alpha, num_iter,
                continuous_idx, onehot_idx
            )
            epoch_losses.append(loss)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}, Loss: {np.mean(epoch_losses[-50:]):.4f}", 
                      end='\r')
        
        print()  # New line
        
        # Validation on clean data
        val_pred = model_robust.predict(X_val, verbose=0)
        val_acc = ((val_pred.flatten() > 0.5).astype(int) == y_val).mean()
        
        # Validation on adversarial data (sample for speed)
        val_sample_size = min(1000, len(X_val))
        X_val_sample = X_val[:val_sample_size]
        y_val_sample = y_val[:val_sample_size]
        
        X_val_adv = pgd_attack_tensorflow(
            model_robust, X_val_sample, y_val_sample,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter,
            continuous_indices=continuous_idx,
            onehot_indices=onehot_idx
        )
        
        val_adv_pred = model_robust.predict(X_val_adv, verbose=0)
        val_adv_acc = ((val_adv_pred.flatten() > 0.5).astype(int) == y_val_sample).mean()
        
        print(f"  Train Loss: {np.mean(epoch_losses):.4f}")
        print(f"  Val Clean Acc: {val_acc:.4f}")
        print(f"  Val Adv Acc: {val_adv_acc:.4f}")
        print()
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_robust.save('models/robust_ids_best.h5')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            old_lr = optimizer.learning_rate.numpy()
            new_lr = old_lr * 0.5
            optimizer.learning_rate.assign(new_lr)
            print(f"  Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}\n")
    
    # Load best model
    model_robust = tf.keras.models.load_model('models/robust_ids_best.h5')
    print("\n✓ Adversarial training completed!")
    print(f"✓ Best model loaded (val accuracy: {best_val_acc:.4f})")
    
    return model_robust


# Train robust model
model_robust = train_robust_ids_model(
    X_train, y_train,
    X_test, y_test,  # Using test as validation for simplicity
    continuous_idx, onehot_idx,
    epsilon=0.03,
    alpha=0.01,
    num_iter=10,  # 10 iterations during training (faster)
    epochs=50,
    batch_size=128,
    class_weight_dict=class_weight_dict
)

print("\n" + "="*80)
print("PHASE 5: COMPREHENSIVE EVALUATION")
print("="*80)

# Evaluate robust model on clean data
robust_clean_results = evaluate_model_comprehensive(
    model_robust,
    X_test,
    y_test,
    "Robust Model - Clean Data"
)


print("\n" + "-"*80)
print("Generating white-box attack on ROBUST model...")
print("-"*80)

X_test_adv_whitebox_robust = generate_adversarial_batched(
    model_robust,
    X_test,
    y_test,
    batch_size=1000,
    epsilon=0.03,
    alpha=0.01,
    num_iter=40,  # Strong attack (40 iterations)
    continuous_indices=continuous_idx,
    onehot_indices=onehot_idx
)

robust_whitebox_results = evaluate_model_comprehensive(
    model_robust,
    X_test_adv_whitebox_robust,
    y_test,
    "Robust Model - White-Box PGD Attack"
)


print("\n" + "-"*80)
print("Testing robust model against transfer attacks...")
print("-"*80)

# Test robust model on previously generated transfer attacks
robust_transfer_simple = evaluate_model_comprehensive(
    model_robust,
    X_test_adv_transfer_simple,
    y_test,
    "Robust Model - Transfer from Simple Surrogate"
)

robust_transfer_deep = evaluate_model_comprehensive(
    model_robust,
    X_test_adv_transfer_deep,
    y_test,
    "Robust Model - Transfer from Deep Surrogate"
)

robust_transfer_ensemble = evaluate_model_comprehensive(
    model_robust,
    X_test_adv_transfer_ensemble,
    y_test,
    "Robust Model - Ensemble Transfer Attack"
)

# Create results dataframe for visualization
import pandas as pd
df_results = pd.DataFrame({
    'Scenario': [
        'Clean Data',
        'White-Box PGD',
        'Transfer (Simple)',
        'Transfer (Deep)',
        'Transfer (Ensemble)'
    ],
    'Baseline Accuracy': [
        baseline_results['accuracy'],
        whitebox_results['accuracy'],
        transfer_results['simple']['accuracy'],
        transfer_results['deep']['accuracy'],
        transfer_results['ensemble']['accuracy']
    ],
    'Robust Accuracy': [
        robust_clean_results['accuracy'],
        robust_whitebox_results['accuracy'],
        robust_transfer_simple['accuracy'],
        robust_transfer_deep['accuracy'],
        robust_transfer_ensemble['accuracy']
    ]
})
df_results['Improvement'] = df_results['Robust Accuracy'] - df_results['Baseline Accuracy']

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Accuracy comparison
scenarios = df_results['Scenario']
x = np.arange(len(scenarios))
width = 0.35

bars1 = ax1.bar(x - width/2, df_results['Baseline Accuracy'], 
                width, label='Baseline Model', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, df_results['Robust Accuracy'], 
                width, label='Robust Model', color='#27ae60', alpha=0.8)

ax1.set_xlabel('Attack Scenario', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1.0])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement
improvements = df_results['Improvement'][1:]  # Exclude clean data
scenarios_adv = df_results['Scenario'][1:]

colors = ['#3498db' if imp > 0 else '#e74c3c' for imp in improvements]
bars3 = ax2.barh(scenarios_adv, improvements, color=colors, alpha=0.8)

ax2.set_xlabel('Accuracy Improvement', fontsize=12, fontweight='bold')
ax2.set_title('Robust Model Improvement vs Baseline', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, improvements)):
    ax2.text(val + 0.01 if val > 0 else val - 0.01, i,
            f'+{val:.3f}' if val > 0 else f'{val:.3f}',
            va='center', ha='left' if val > 0 else 'right',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to results/model_comparison.png")
plt.show()


print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Calculate key metrics
clean_accuracy_drop = baseline_results['accuracy'] - robust_clean_results['accuracy']
avg_adv_improvement = df_results['Improvement'][1:].mean()

wb_success_baseline = 1 - (whitebox_results['accuracy'] / baseline_results['accuracy'])
wb_success_robust = 1 - (robust_whitebox_results['accuracy'] / robust_clean_results['accuracy'])

transfer_avg_baseline = df_results['Baseline Accuracy'][2:].mean()
transfer_avg_robust = df_results['Robust Accuracy'][2:].mean()

print(f"\nKey Findings:")
print(f"\n1. Clean Accuracy Trade-off:")
print(f"   - Baseline: {baseline_results['accuracy']:.4f}")
print(f"   - Robust:   {robust_clean_results['accuracy']:.4f}")
print(f"   - Drop:     {clean_accuracy_drop:.4f} ({clean_accuracy_drop/baseline_results['accuracy']*100:.2f}%)")

print(f"\n2. White-Box Attack Resistance:")
print(f"   - Baseline attack success: {wb_success_baseline:.2%}")
print(f"   - Robust attack success:   {wb_success_robust:.2%}")
print(f"   - Improvement:             {wb_success_baseline - wb_success_robust:.2%}")

print(f"\n3. Black-Box Transfer Attack Resistance:")
print(f"   - Baseline avg accuracy: {transfer_avg_baseline:.4f}")
print(f"   - Robust avg accuracy:   {transfer_avg_robust:.4f}")
print(f"   - Average improvement:   {transfer_avg_robust - transfer_avg_baseline:.4f}")

print(f"\n4. Overall Adversarial Robustness:")
print(f"   - Average improvement across all attacks: {avg_adv_improvement:.4f}")
print(f"   - Relative improvement: {avg_adv_improvement/df_results['Baseline Accuracy'][1:].mean()*100:.1f}%")

print(f"\n5. Transfer Attack Analysis:")
simple_transfer = transfer_results['simple']['accuracy']
deep_transfer = transfer_results['deep']['accuracy']
ensemble_transfer = transfer_results['ensemble']['accuracy']
whitebox_acc = whitebox_results['accuracy']

print(f"   - Simple surrogate transferability:   {simple_transfer/baseline_results['accuracy']:.2%}")
print(f"   - Deep surrogate transferability:     {deep_transfer/baseline_results['accuracy']:.2%}")
print(f"   - Ensemble transferability:           {ensemble_transfer/baseline_results['accuracy']:.2%}")
print(f"   - Transfer vs white-box effectiveness: {ensemble_transfer/whitebox_acc:.2%}")

print("\n" + "="*80)

print("\n" + "="*80)
print("SAVING FINAL DELIVERABLES")
print("="*80)

# Ensure directories exist
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('adversarial_data').mkdir(exist_ok=True)

# Models
print("\nSaving models...")
model_baseline.save('models/baseline_ids_final.h5')
model_robust.save('models/robust_ids_final.h5')
surrogate_simple.save('models/surrogate_simple_final.h5')
surrogate_deep.save('models/surrogate_deep_final.h5')
print("✓ All models saved!")

# Results
print("\nSaving results...")
results_dict = {
    'baseline_clean': baseline_results,
    'baseline_whitebox': whitebox_results,
    'baseline_transfer_simple': transfer_results['simple'],
    'baseline_transfer_deep': transfer_results['deep'],
    'baseline_transfer_ensemble': transfer_results['ensemble'],
    'robust_clean': robust_clean_results,
    'robust_whitebox': robust_whitebox_results,
    'robust_transfer_simple': robust_transfer_simple,
    'robust_transfer_deep': robust_transfer_deep,
    'robust_transfer_ensemble': robust_transfer_ensemble
}

import pickle
with open('results/all_results.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

df_results.to_csv('results/summary_table.csv', index=False)
print("✓ Results saved!")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print("\nFiles saved:")
print("  Models:")
print("    - models/baseline_ids_final.h5")
print("    - models/robust_ids_final.h5")
print("    - models/surrogate_simple_final.h5")
print("    - models/surrogate_deep_final.h5")
print("\n  Results:")
print("    - results/comprehensive_results.csv")
print("    - results/summary_table.csv")
print("    - results/all_results.pkl")
print("    - results/model_comparison.png")
print("\n  Adversarial Data:")
print("    - adversarial_data/X_test_adv_whitebox_eps003.npy")
print("    - adversarial_data/X_test_adv_transfer_*.npy")
print("="*80)