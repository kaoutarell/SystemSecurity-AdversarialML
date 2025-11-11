import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from pathlib import Path


def evaluate_model(model, X_test, y_test, num_classes=2, binary=True, verbose=1):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        num_classes: Number of classes (2 for binary)
        binary: Whether it's binary classification
        verbose: Verbosity level for model.evaluate()
    
    Returns:
        dict: Dictionary containing all metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    
    if binary:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = y_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    
    # Try to get AUC if applicable
    try:
        if binary:
            auc = roc_auc_score(y_true, y_pred_probs)
        else:
            auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')
    except:
        auc = None
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get loss from model if available
    try:
        eval_results = model.evaluate(X_test, y_test, verbose=verbose)
        loss = eval_results[0] if isinstance(eval_results, list) else eval_results
    except:
        loss = None
    
    # Compile metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    if auc is not None:
        metrics['auc'] = float(auc)
    if loss is not None:
        metrics['loss'] = float(loss)
    
    # Print metrics
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    if auc is not None:
        print(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print("="*80)
    
    return metrics


def evaluate_model_simple(model, X_test, y_test, verbose=1):
    """
    Simple evaluation using model's built-in evaluate method.
    Legacy function for backward compatibility.
    
    Returns:
        tuple: (metrics_dict, y_pred)
    """
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    print("\nEvaluating model...")
    
    # Model now returns 5 metrics: loss, accuracy, precision, recall, auc
    eval_results = model.evaluate(X_test, y_test, verbose=verbose)
    loss = eval_results[0]
    accuracy = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    auc = eval_results[4] if len(eval_results) > 4 else None

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'loss': float(loss)
    }
    
    if auc is not None:
        metrics['auc'] = float(auc)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    if auc is not None:
        print(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Loss:      {loss:.4f}")
    print("="*60)
    
    return metrics, y_pred


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print classification report with precision, recall, f1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names (default: ['Normal', 'Attack'])
    """
    if target_names is None:
        target_names = ['Normal', 'Attack']
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("="*80)


def plot_confusion_matrix(cm_or_y_true, y_pred=None, output_path=None, labels=None):
    """
    Plot confusion matrix.
    
    Args:
        cm_or_y_true: Either confusion matrix (2D array) or y_true labels
        y_pred: Predicted labels (required if cm_or_y_true is y_true)
        output_path: Path to save plot
        labels: Display labels (default: ['Normal', 'Attack'])
    """
    if labels is None:
        labels = ['Normal', 'Attack']
    
    # Handle both confusion matrix and y_true/y_pred inputs
    if y_pred is not None:
        # cm_or_y_true is actually y_true
        cm = confusion_matrix(cm_or_y_true, y_pred)
    else:
        # cm_or_y_true is the confusion matrix
        cm = np.array(cm_or_y_true) if not isinstance(cm_or_y_true, np.ndarray) else cm_or_y_true
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    plt.close()
    return fig


def plot_training_curves(history, output_path=None):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    
    plt.close()
    return fig


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


def save_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {output_path}")
