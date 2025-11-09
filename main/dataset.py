"""
Data Preprocessing for CNN Channel Attention IDS
Following the exact methodology from:
"CNN Channel Attention Intrusion Detection System Using NSL-KDD Dataset"
Alrayes et al., 2024

Key changes from standard preprocessing:
1. MinMaxScaler [0, 1] instead of RobustScaler
2. Padding to 144 features (12×12) or 784 features (28×28)
3. Reshaping to 2D images for CNN input
4. Separate test set (no train/test split from training data)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path


# NSL-KDD dataset column definitions
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
    """
    Load NSL-KDD dataset from file.
    
    Args:
        filepath: Path to KDDTrain+.txt or KDDTest+.txt
    
    Returns:
        DataFrame with NSL-KDD data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None)

    # Some NSL-KDD files include a difficulty column at the end
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        print("  Detected extra column(s) (e.g. difficulty). Removing trailing columns...")
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]

    df.columns = NSL_KDD_COLUMNS
    
    print(f"  Loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    print(f"\n  Class distribution:")
    print(df[TARGET_COLUMN].value_counts())
    
    return df


def pad_features(X, target_features=144):
    """
    Pad features to target number (144 for 12×12, 784 for 28×28).
    
    Paper methodology (Section 4.2):
    - Uses zero padding (constant mode)
    - Ensures exact square matrix dimensions
    
    Args:
        X: Feature matrix (samples, features)
        target_features: Target number of features (12×12=144 or 28×28=784)
    
    Returns:
        Padded feature matrix (samples, target_features)
    """
    current_features = X.shape[1]
    
    if current_features < target_features:
        padding = target_features - current_features
        print(f"  Padding: {current_features} → {target_features} features (adding {padding} zeros)")
        X_padded = np.pad(X, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    elif current_features > target_features:
        print(f"  Truncating: {current_features} → {target_features} features")
        X_padded = X[:, :target_features]
    else:
        print(f"  No padding needed: {current_features} features")
        X_padded = X
    
    return X_padded


def reshape_to_images(X, image_size=12):
    """
    Reshape 1D features to 2D images for CNN.
    
    Paper methodology (Section 4.2):
    - Reshapes to square matrices (12×12 or 28×28)
    - Adds channel dimension (grayscale: 1 channel)
    
    Args:
        X: Feature matrix (samples, features) where features = image_size²
        image_size: Height/width of square image (12 or 28)
    
    Returns:
        Reshaped data (samples, height, width, 1) for CNN input
    """
    n_samples = X.shape[0]
    expected_features = image_size * image_size
    
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Feature count ({X.shape[1]}) doesn't match image_size² ({expected_features}). "
            f"Use pad_features() first."
        )
    
    X_images = X.reshape(n_samples, image_size, image_size, 1)
    print(f"  Reshaped to images: {X_images.shape}")
    
    return X_images


def preprocess_data(df, scaler=None, feature_columns=None, image_size=12):
    """
    Preprocess NSL-KDD data following the paper's methodology.
    
    Paper's preprocessing steps (Section 4.1-4.2):
    1. Handle missing values
    2. Create binary labels (0=normal, 1=attack)
    3. One-hot encode categorical features
    4. Pad to square number of features
    5. MinMax normalize to [0, 1]
    6. Reshape to 2D images
    
    Args:
        df: Raw NSL-KDD DataFrame
        scaler: Fitted MinMaxScaler (None for training data)
        feature_columns: Expected feature columns (None for training data)
        image_size: Image dimensions (12 for 12×12, 28 for 28×28)
    
    Returns:
        X_images: Preprocessed images (samples, height, width, 1)
        y: Binary labels (samples,)
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature column names
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    print("  Step 1: Handling missing values...")
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 2: Create binary labels
    print("  Step 2: Creating binary labels (0=normal, 1=attack)...")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
    y = df[TARGET_COLUMN].values
    
    # Step 3: One-hot encode categorical features
    print(f"  Step 3: One-hot encoding {CATEGORICAL_COLUMNS}...")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    # Align columns with training set (for test data)
    if feature_columns is not None:
        print("  Aligning features with training set...")
        df = df.reindex(columns=list(feature_columns) + DROP_COLUMNS, fill_value=0)
    else:
        feature_columns = df.drop(DROP_COLUMNS, axis=1).columns.tolist()
    
    # Extract features
    X = df.drop(DROP_COLUMNS, axis=1).values
    print(f"  Features after one-hot encoding: {X.shape[1]}")
    
    # Step 4: Pad to square number
    print(f"  Step 4: Padding to {image_size}×{image_size} = {image_size**2} features...")
    target_features = image_size * image_size
    X_padded = pad_features(X, target_features)
    
    # Step 5: MinMax normalize [0, 1] - CRITICAL for paper's method
    print("  Step 5: MinMax normalization [0, 1]...")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = scaler.fit_transform(X_padded)
        print("    Fitted new MinMaxScaler")
    else:
        X_normalized = scaler.transform(X_padded)
        print("    Used existing MinMaxScaler")
    
    # Verify normalization range
    print(f"    Data range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    
    # Step 6: Reshape to 2D images
    print(f"  Step 6: Reshaping to {image_size}×{image_size} images...")
    X_images = reshape_to_images(X_normalized, image_size)
    
    return X_images, y, scaler, feature_columns


def preprocess_data_standard(df, scaler=None, feature_columns=None, binary=True):
    """
    Preprocess NSL-KDD data with Standard (Z-score) normalization.
    
    For models that benefit from zero-centered, unit-variance features.
    Better for deep neural networks with gradient-based optimization.
    
    Args:
        df: Raw NSL-KDD DataFrame
        scaler: Fitted StandardScaler (None for training data)
        feature_columns: Expected feature columns (None for training data)
        binary: Binary (True) or multi-class (False) classification
    
    Returns:
        X: Preprocessed features (samples, features)
        y: Labels (samples,)
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 2: Create labels
    if binary:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
        y = df[TARGET_COLUMN].values
    else:
        # Multi-class: map to 5 classes (Normal, Probe, DoS, U2R, R2L)
        label_map = {
            'normal': 0,
            'probe': 1, 'portsweep': 1, 'ipsweep': 1, 'nmap': 1, 'satan': 1,
            'dos': 2, 'back': 2, 'land': 2, 'neptune': 2, 'pod': 2, 'smurf': 2, 'teardrop': 2,
            'u2r': 3, 'buffer_overflow': 3, 'loadmodule': 3, 'perl': 3, 'rootkit': 3,
            'r2l': 4, 'ftp_write': 4, 'guess_passwd': 4, 'imap': 4, 'multihop': 4,
            'phf': 4, 'spy': 4, 'warezclient': 4, 'warezmaster': 4
        }
        y = df[TARGET_COLUMN].map(label_map).fillna(0).astype(int).values
    
    # Step 3: One-hot encode categorical features
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    # Align columns with training set (for test data)
    if feature_columns is not None:
        df = df.reindex(columns=list(feature_columns) + DROP_COLUMNS, fill_value=0)
    else:
        feature_columns = df.drop(DROP_COLUMNS, axis=1).columns.tolist()
    
    # Extract features
    X = df.drop(DROP_COLUMNS, axis=1).values.astype(np.float32)
    
    # Step 4: Standard scale (Z-score normalization: mean=0, std=1)
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler, feature_columns


def preprocess_data_minmax(df, scaler=None, feature_columns=None, binary=True):
    """
    Preprocess NSL-KDD data with MinMax normalization (no image reshaping).
    
    For SAAE-DNN and similar models that work with flat feature vectors.
    
    Args:
        df: Raw NSL-KDD DataFrame
        scaler: Fitted MinMaxScaler (None for training data)
        feature_columns: Expected feature columns (None for training data)
        binary: Binary (True) or multi-class (False) classification
    
    Returns:
        X: Preprocessed features (samples, features)
        y: Labels (samples,)
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature column names
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    # Step 2: Create labels
    if binary:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
        y = df[TARGET_COLUMN].values
    else:
        # Multi-class: map to 5 classes (Normal, Probe, DoS, U2R, R2L)
        label_map = {
            'normal': 0,
            'probe': 1, 'portsweep': 1, 'ipsweep': 1, 'nmap': 1, 'satan': 1,
            'dos': 2, 'back': 2, 'land': 2, 'neptune': 2, 'pod': 2, 'smurf': 2, 'teardrop': 2,
            'u2r': 3, 'buffer_overflow': 3, 'loadmodule': 3, 'perl': 3, 'rootkit': 3,
            'r2l': 4, 'ftp_write': 4, 'guess_passwd': 4, 'imap': 4, 'multihop': 4,
            'phf': 4, 'spy': 4, 'warezclient': 4, 'warezmaster': 4
        }
        y = df[TARGET_COLUMN].map(label_map).fillna(0).astype(int).values
    
    # Step 3: One-hot encode categorical features
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    # Align columns with training set (for test data)
    if feature_columns is not None:
        df = df.reindex(columns=list(feature_columns) + DROP_COLUMNS, fill_value=0)
    else:
        feature_columns = df.drop(DROP_COLUMNS, axis=1).columns.tolist()
    
    # Extract features
    X = df.drop(DROP_COLUMNS, axis=1).values.astype(np.float32)
    
    # Step 4: MinMax normalize [0, 1]
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler, feature_columns


def prepare_datasets(X_train_full, y_train_full, val_split=0.1, random_state=42):
    """
    Split training data into train/validation sets.
    
    Paper's methodology:
    - 90% training, 10% validation (from KDDTrain+.txt)
    - Test set is completely separate (KDDTest+.txt)
    
    Args:
        X_train_full: Full training images (samples, height, width, 1)
        y_train_full: Full training labels (samples,)
        val_split: Validation set proportion (default: 0.1 = 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train: Training images
        X_val: Validation images
        y_train: Training labels
        y_val: Validation labels
    """
    print(f"\nSplitting training data: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% validation...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_split, 
        random_state=random_state, 
        stratify=y_train_full
    )
    
    # Convert to float32 for TensorFlow
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Validation set: {X_val.shape[0]:,} samples")
    
    print(f"\n  Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\n  Validation set distribution:")
    unique, counts = np.unique(y_val, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_val)*100:.1f}%)")
    
    return X_train, X_val, y_train, y_val


def save_artifacts(scaler, feature_columns, image_size, output_path):
    """
    Save preprocessing artifacts for later use.
    
    Args:
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature column names
        image_size: Image dimensions used
        output_path: Path to save artifacts
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'image_size': image_size,
        'scaler_type': 'MinMaxScaler',
        'feature_range': (0, 1)
    }
    
    joblib.dump(artifacts, output_path)
    print(f"\nArtifacts saved to {output_path}")
    print(f"  - Scaler: MinMaxScaler [0, 1]")
    print(f"  - Features: {len(feature_columns)}")
    print(f"  - Image size: {image_size}×{image_size}")


def load_artifacts(artifacts_path):
    """
    Load preprocessing artifacts.
    
    Args:
        artifacts_path: Path to saved artifacts
    
    Returns:
        Dictionary with scaler, feature_columns, and image_size
    """
    artifacts_path = Path(artifacts_path)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts not found at: {artifacts_path}")
    
    artifacts = joblib.load(artifacts_path)
    
    # Validate artifacts
    required_keys = ['scaler', 'feature_columns', 'image_size']
    for key in required_keys:
        if key not in artifacts:
            raise ValueError(f"Artifacts missing required key: {key}")
    
    print(f"Loaded artifacts from {artifacts_path}")
    print(f"  - Scaler: {artifacts.get('scaler_type', 'Unknown')}")
    print(f"  - Features: {len(artifacts['feature_columns'])}")
    print(f"  - Image size: {artifacts['image_size']}×{artifacts['image_size']}")
    
    return artifacts


def load_and_preprocess_file(filepath, artifacts_path=None, scaler=None, 
                             feature_columns=None, image_size=12, n_samples=None):
    """
    Load and preprocess a NSL-KDD file (for testing/inference).
    
    Args:
        filepath: Path to data file
        artifacts_path: Path to saved artifacts (alternative to passing scaler/features)
        scaler: Fitted MinMaxScaler (if not using artifacts_path)
        feature_columns: Expected features (if not using artifacts_path)
        image_size: Image dimensions (if not using artifacts_path)
        n_samples: Limit number of samples (None for all)
    
    Returns:
        X_images: Preprocessed images (samples, height, width, 1)
        y: Binary labels (samples,)
        df: Original DataFrame
    """
    # Load artifacts if path provided
    if artifacts_path is not None:
        artifacts = load_artifacts(artifacts_path)
        scaler = artifacts['scaler']
        feature_columns = artifacts['feature_columns']
        image_size = artifacts['image_size']
    
    # Validate inputs
    if scaler is None or feature_columns is None:
        raise ValueError("Must provide either artifacts_path or (scaler + feature_columns)")
    
    # Load data
    df = pd.read_csv(filepath, header=None)
    
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]
    df.columns = NSL_KDD_COLUMNS

    if n_samples is not None:
        print(f"Limiting to first {n_samples} samples...")
        df = df.head(n_samples)

    # Preprocess
    print(f"\nPreprocessing {filepath}...")
    X_images, y, _, _ = preprocess_data(
        df, 
        scaler=scaler, 
        feature_columns=feature_columns,
        image_size=image_size
    )
    
    X_images = X_images.astype('float32')
    
    return X_images, y, df


def process_train_test_files(train_path, test_path, image_size=12, 
                             val_split=0.1, artifacts_path='artifacts/preprocessing.pkl'):
    """
    Complete preprocessing pipeline for training.
    
    Paper's methodology:
    1. Load KDDTrain+.txt and KDDTest+.txt separately
    2. Preprocess training data (fit scaler)
    3. Split training into train/validation (90/10)
    4. Preprocess test data (use fitted scaler)
    5. Save artifacts
    
    Args:
        train_path: Path to KDDTrain+.txt
        test_path: Path to KDDTest+.txt
        image_size: Image dimensions (12 or 28)
        val_split: Validation split proportion (default: 0.1)
        artifacts_path: Path to save preprocessing artifacts
    
    Returns:
        X_train: Training images
        X_val: Validation images
        X_test: Test images
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
    """
    print("="*80)
    print("CNN Channel Attention IDS - Data Preprocessing")
    print("Following paper methodology: Alrayes et al., 2024")
    print("="*80)
    
    # Load and preprocess training data
    print(f"\n{'='*80}")
    print("STEP 1: Loading and preprocessing training data")
    print(f"{'='*80}")
    df_train = load_data(train_path)
    X_train_full, y_train_full, scaler, feature_columns = preprocess_data(
        df_train, 
        scaler=None, 
        feature_columns=None,
        image_size=image_size
    )
    
    # Split into train/validation
    print(f"\n{'='*80}")
    print("STEP 2: Creating train/validation split")
    print(f"{'='*80}")
    X_train, X_val, y_train, y_val = prepare_datasets(
        X_train_full, y_train_full,
        val_split=val_split
    )
    
    # Load and preprocess test data
    print(f"\n{'='*80}")
    print("STEP 3: Loading and preprocessing test data")
    print(f"{'='*80}")
    df_test = load_data(test_path)
    X_test, y_test, _, _ = preprocess_data(
        df_test,
        scaler=scaler,
        feature_columns=feature_columns,
        image_size=image_size
    )
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"\n  Test set distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_test)*100:.1f}%)")
    
    # Save artifacts
    print(f"\n{'='*80}")
    print("STEP 4: Saving preprocessing artifacts")
    print(f"{'='*80}")
    save_artifacts(scaler, feature_columns, image_size, artifacts_path)
    
    # Final summary
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Training set:   {X_train.shape} → {X_train.shape[0]:,} images of {image_size}×{image_size}")
    print(f"Validation set: {X_val.shape} → {X_val.shape[0]:,} images of {image_size}×{image_size}")
    print(f"Test set:       {X_test.shape} → {X_test.shape[0]:,} images of {image_size}×{image_size}")
    print(f"\nData ready for CNN training!")
    print(f"Expected input shape: (batch_size, {image_size}, {image_size}, 1)")
    print("="*80)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage
if __name__ == "__main__":
    # Process training and test files
    X_train, X_val, X_test, y_train, y_val, y_test = process_train_test_files(
        train_path='nsl-kdd/KDDTrain+.txt',
        test_path='nsl-kdd/KDDTest+.txt',
        image_size=12,
        val_split=0.1,
        artifacts_path='artifacts/preprocessing.pkl'
    )
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES FROM STANDARD PREPROCESSING:")
    print("="*80)
    print("✓ MinMaxScaler [0, 1] instead of RobustScaler")
    print("✓ Padding to 144 features (12×12)")
    print("✓ Reshaping to 2D images: (samples, 12, 12, 1)")
    print("✓ Separate test file (not split from training)")
    print("✓ 90/10 train/validation split")
    print("="*80)