import argparse
import os
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Utilities / reproducibility
# ----------------------------------------------------------------
RND_SEED = 42
def set_seed(seed=RND_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ----------------------------------------------------------------
# Simple Feedforward Model
# ----------------------------------------------------------------
class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)  # 2 classes: normal / attack
        )
    
    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------
# Data loading + preprocessing
# ----------------------------------------------------------------
def load_nsl_kdd(path):
    """Load NSL-KDD dataset and handle the difficulty column."""
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None, sep=r'\s+', engine='python')
    
    print(f"Loaded {path}: shape = {df.shape}")
    print(f"Last 3 columns sample:\n{df.iloc[:5, -3:]}")
    
    # NSL-KDD format: [41 features, label, difficulty] | txt file
    # check if last column is numeric (difficulty score) -- bug fix | columns were 
    if df.shape[1] > 42:
        print("Detected difficulty column, removing it...")
        df = df.iloc[:, :-1]  # remove last column (difficulty) | wasn't well evaluated before
    
    return df

def preprocess(train_df, test_df, binary_label=True):
    """
    Preprocess NSL-KDD data with proper label handling.
    Label should now be at index -1 after removing difficulty column.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # trimming whitespace in string columns
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].str.strip()
            test_df[col] = test_df[col].str.strip()
    
    # extracting labels (at index -1)
    y_train_raw = train_df.iloc[:, -1].astype(str)
    y_test_raw = test_df.iloc[:, -1].astype(str)
    
    print(f"\nLabel distribution (train):")
    print(y_train_raw.value_counts().head(10))
    print(f"\nLabel distribution (test):")
    print(y_test_raw.value_counts().head(10))
    
    # drop label column
    X_train_df = train_df.iloc[:, :-1]
    X_test_df = test_df.iloc[:, :-1]
    
    print(f"\nFeature matrix shape: Train={X_train_df.shape}, Test={X_test_df.shape}")
    
    # identify categorical columns
    cat_cols = X_train_df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    # encode them 
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_df[col], X_test_df[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_df[col] = le.transform(X_train_df[col].astype(str))
        X_test_df[col] = le.transform(X_test_df[col].astype(str))
        encoders[col] = le
    
    # convert to numeric
    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
    X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce')
    
    # fill NaNs
    X_train_df = X_train_df.fillna(0.0)
    X_test_df = X_test_df.fillna(0.0)
    
    # scale 
    scaler = StandardScaler()
    scaler.fit(X_train_df)
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    
    # convert labels to binary
    if binary_label:
        y_train = (y_train_raw.str.lower() != "normal").astype(int).values
        y_test = (y_test_raw.str.lower() != "normal").astype(int).values
        
        print(f"\nBinary label distribution:")
        print(f"Train - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
        print(f"Test  - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    else:
        le_label = LabelEncoder()
        le_label.fit(pd.concat([y_train_raw, y_test_raw], axis=0))
        y_train = le_label.transform(y_train_raw)
        y_test = le_label.transform(y_test_raw)
        encoders['label_encoder'] = le_label
    
    artifacts = {
        "scaler": scaler,
        "encoders": encoders,
        "cat_cols": cat_cols,
        "feature_count": X_train.shape[1]
    }
    
    return X_train, X_test, y_train, y_test, artifacts

# ----------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += Xb.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += Xb.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch}/{epochs}  | Train loss: {train_loss:.4f} acc: {train_acc:.4f}  | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
    
    return model, history

def evaluate_model(model, X, y, device='cpu'):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, digits=4, labels=[0, 1], target_names=['Normal', 'Attack'])
    cm = confusion_matrix(y, preds, labels=[0, 1])
    
    return acc, report, cm, preds

# ----------------------------------------------------------------
#                            **Main**
# ----------------------------------------------------------------
def main(args):
    set_seed()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*60)
    print("Loading data...")
    print("="*60)
    train_df = load_nsl_kdd(args.train)
    test_df = load_nsl_kdd(args.test)
    
    print("\n" + "="*60)
    print("Preprocessing...")
    print("="*60)
    X_train_all, X_test, y_train_all, y_test, artifacts = preprocess(
        train_df, test_df, binary_label=True
    )
    
    input_dim = artifacts["feature_count"]
    
    # split training into train & val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_all, 
        test_size=0.15, 
        random_state=RND_SEED, 
        stratify=y_train_all
    )
    
    # convert to tensors and dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = SimpleFFNN(input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    print(model)
    
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    model, history = train_model(
        model, train_loader, val_loader, 
        epochs=args.epochs, lr=args.lr, device=device
    )
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_acc, test_report, test_cm, preds = evaluate_model(model, X_test, y_test, device=device)
    
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nClassification report:\n", test_report)
    print("\nConfusion matrix:\n", test_cm)
    
    # save model & artifacts
    model_path = Path(args.out_dir) / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    artifacts_save = {
        "scaler": artifacts["scaler"],
        "encoders": artifacts["encoders"],
        "cat_cols": artifacts["cat_cols"],
        "feature_count": artifacts["feature_count"]
    }
    joblib.dump(artifacts_save, Path(args.out_dir) / "artifacts.joblib")
    print("Saved preprocessing artifacts.")
    
    # save metrics
    with open(Path(args.out_dir)/"metrics.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write("\nClassification report:\n")
        f.write(test_report)
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(test_cm))
    
    #------------------------------------------------------------------------------------------
    # very optional but just for debugging purposes on my end : plot training curves 
    #------------------------------------------------------------------------------------------
  
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Loss Curve")
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", marker='o')
    plt.plot(history["val_acc"], label="Val Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Accuracy Curve")
    
    plt.tight_layout()
    plt.savefig(Path(args.out_dir)/"training_curves.png", dpi=150)
    plt.close()
    
    print(f"\nAll results saved to {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to training file")
    p.add_argument("--test", required=True, help="Path to test file")
    p.add_argument("--out_dir", default="results", help="Output directory")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    
    args = p.parse_args()
    main(args)