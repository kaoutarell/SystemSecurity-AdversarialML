import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
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


def preprocess_data(df, scaler=None, feature_columns=None):
    df = df.copy()
    
    for col in ['duration', 'wrong_fragment']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    num_cols = df.drop(DROP_COLUMNS + CATEGORICAL_COLUMNS, axis=1).columns.tolist()

    if scaler is None:
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "normal").astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    if feature_columns is not None:
        df = df.reindex(columns=list(feature_columns) + DROP_COLUMNS, fill_value=0)
    else:
        feature_columns = df.drop(DROP_COLUMNS, axis=1).columns.tolist()
    
    return df, scaler, feature_columns


def prepare_datasets(df, test_size=0.2, random_state=42):
    X = df.drop(DROP_COLUMNS, axis=1).values
    y = df[TARGET_COLUMN].values.astype(int)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    print(f"\n  Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Normal' if label == 0 else 'Attack'
        print(f"    {label_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def save_artifacts(scaler, feature_columns, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    joblib.dump(artifacts, output_path)
    print(f"Artifacts saved to {output_path}")


def load_artifacts(artifacts_path):
    artifacts_path = Path(artifacts_path)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts not found at: {artifacts_path}")
    
    artifacts = joblib.load(artifacts_path)
    return artifacts


def load_and_preprocess_file(filepath, artifacts_path, n_samples=None):
    artifacts = load_artifacts(artifacts_path)
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    df = pd.read_csv(filepath, header=None)
    
    if df.shape[1] > len(NSL_KDD_COLUMNS):
        df = df.iloc[:, :len(NSL_KDD_COLUMNS)]
    df.columns = NSL_KDD_COLUMNS

    if n_samples is not None:
        df = df.head(n_samples)

    y = None
    if TARGET_COLUMN in df.columns:
        y = (df[TARGET_COLUMN] != 'normal').astype(int).values
    
    df_processed, _, _ = preprocess_data(df, scaler=scaler, feature_columns=feature_columns)

    X = df_processed.drop(DROP_COLUMNS, axis=1).values.astype('float32')
    
    return X, y, df
