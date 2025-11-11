"""
NSL-KDD Data Preprocessing for Adversarial ML Experiments

Simple preprocessing pipeline with StandardScaler normalization.
Designed for dense neural network models with gradient-based attacks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def preprocess_nsl_kdd(df, scaler=None, feature_columns=None, binary=True):
    """
    Preprocess NSL-KDD data with StandardScaler normalization.
    
    Pipeline:
    1. Handle missing values
    2. Create binary labels (normal=0, attack=1)
    3. One-hot encode categorical features
    4. StandardScaler normalization (mean=0, std=1)
    
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
    
    # Step 4: StandardScaler normalization (mean=0, std=1)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler, feature_columns