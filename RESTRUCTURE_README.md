# IDS Neural Network - Restructured Codebase

## 📁 Project Structure

```
SystemSecurity-AdversarialML/
├── model.py              # Model architecture and training logic
├── dataset.py            # Data loading and preprocessing
├── evaluate.py           # Model evaluation and metrics
├── utils.py              # Utility functions (directory management)
├── train_ids.py          # Main training script
├── attacks/              # Adversarial attack implementations
│   └── fgsm.py          # FGSM attack
├── results/              # All outputs saved here
│   ├── models/          # Latest trained models and artifacts
│   ├── runs/            # Individual training run outputs
│   │   ├── run_001/
│   │   ├── run_002/
│   │   └── ...
│   └── attacks/         # Attack evaluation results
├── nsl-kdd/             # NSL-KDD dataset
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Training a Model

```bash
# Basic training (auto-incremented run directory)
python train_ids.py --train_path nsl-kdd/KDDTrain+.txt --epochs 15 --batch_size 2048

# With custom hyperparameters
python train_ids.py \
    --train_path nsl-kdd/KDDTrain+.txt \
    --epochs 20 \
    --batch_size 2048 \
    --dropout 0.1 \
    --l1 1e-6 \
    --l2 1e-5 \
    --learning_rate 0.001 \
    --optimizer adam
```

**Training outputs saved to:**
- `results/runs/run_XXX/` - Individual run results (models, metrics, plots)
- `results/models/` - Latest model for inference

### 2. Running FGSM Attack

```bash
# Test on all samples with multiple epsilon values
python attacks/fgsm.py \
    --model_path results/models/nn_ids_model.keras \
    --artifacts results/models/artifacts.joblib \
    --test_path nsl-kdd/KDDTest+.txt \
    --n_samples 22544 \
    --epsilons 0.01 0.02 0.05 0.1

# Quick test on subset
python attacks/fgsm.py --n_samples 1000 --epsilons 0.01 0.05 0.1
```

## 📚 Module Documentation

### `model.py`
**Model architecture and training logic**

Key functions:
- `build_model(input_dim, dropout_rate=0.3, l1=1e-5, l2=1e-4)` - Construct neural network
- `compile_model(model, learning_rate=0.001, optimizer_name='adam')` - Compile with optimizer
- `create_callbacks(output_dir, ...)` - Create training callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping)
- `train_model(model, X_train, y_train, ...)` - Train the model
- `load_model(model_path)` - Load saved model

**Architecture:**
- 4 hidden layers: 64 → 128 → 256 → 64
- ReLU activation, Dropout, L1/L2 regularization
- Sigmoid output for binary classification

### `dataset.py`
**Data loading and preprocessing**

Key functions:
- `load_data(filepath)` - Load NSL-KDD file
- `preprocess_data(df, scaler=None, feature_columns=None)` - Preprocess features
  - Numeric scaling (RobustScaler)
  - One-hot encoding for categorical features
  - Binary label conversion (normal=0, attack=1)
- `prepare_datasets(df, test_size=0.2)` - Train/test split
- `save_artifacts(scaler, feature_columns, output_path)` - Save preprocessing artifacts
- `load_and_preprocess_file(filepath, artifacts_path, n_samples)` - Load and preprocess for inference

**Features:**
- 43 original features → 122 features after preprocessing
- ~35 numeric features (scaled)
- ~87 one-hot encoded categorical features

### `evaluate.py`
**Model evaluation and visualization**

Key functions:
- `evaluate_model(model, X_test, y_test)` - Compute metrics
- `print_classification_report(y_true, y_pred)` - Detailed per-class metrics
- `plot_confusion_matrix(y_true, y_pred, output_path)` - Confusion matrix visualization
- `plot_training_curves(history, output_path)` - Training/validation curves
- `save_metrics(metrics, output_path)` - Save metrics to JSON
- `save_training_history(history, output_path)` - Save training history

### `attacks/fgsm.py`
**FGSM adversarial attack**

Key functions:
- `fgsm_attack(model, x, y, epsilon)` - Generate adversarial examples
  - Formula: `adv_x = x + ε × sign(∇_x J(θ, x, y))`
- `evaluate_robustness(model, X_clean, y_clean, epsilons)` - Test multiple epsilon values
- `print_summary(results)` - Display attack results

**Usage:**
Tests model robustness by perturbing the 122 network features with varying epsilon magnitudes.

## 📊 Results Directory Structure

```
results/
├── models/                           # Latest model (used for inference)
│   ├── nn_ids_model.keras           # Trained model
│   └── artifacts.joblib             # Scaler + feature columns
├── runs/                             # Training run outputs
│   ├── run_001/
│   │   ├── nn_ids_model.keras       # Model snapshot
│   │   ├── artifacts.joblib         # Artifacts
│   │   ├── metrics.json             # Evaluation metrics
│   │   ├── training_history.json    # Training history
│   │   ├── confusion_matrix.png     # Confusion matrix plot
│   │   └── training_curves.png      # Training curves plot
│   └── run_002/
│       └── ...
└── attacks/                          # Attack evaluation outputs
```

## 🔬 Example Workflow

### Complete Training and Attack Pipeline

```bash
# 1. Train model (saves to results/runs/run_XXX/ and results/models/)
python train_ids.py \
    --train_path nsl-kdd/KDDTrain+.txt \
    --epochs 15 \
    --batch_size 2048 \
    --dropout 0.1

# 2. Evaluate robustness with FGSM attack
python attacks/fgsm.py \
    --model_path results/models/nn_ids_model.keras \
    --artifacts results/models/artifacts.joblib \
    --test_path nsl-kdd/KDDTest+.txt \
    --epsilons 0.01 0.02 0.05 0.1

# 3. Check results
cat results/runs/run_001/metrics.json
```

## 🎯 Key Improvements

1. **Modular Architecture**: Clean separation of concerns
   - Model logic in `model.py`
   - Data handling in `dataset.py`
   - Evaluation in `evaluate.py`
   - Attacks in `attacks/` directory

2. **Centralized Results**: All outputs in `results/` directory
   - Auto-incremented run directories
   - Latest model in `results/models/`
   - Easy to track experiments

3. **Reusable Components**: Import functions across scripts
   - No code duplication
   - Easy to extend with new attacks
   - Clean main training script

4. **Attack Organization**: Separate directory for attack implementations
   - Easy to add new attacks (PGD, C&W, etc.)
   - Consistent interface across attacks

## 🔧 Legacy Files

The following files are from the old structure and can be ignored:
- `train_nn.py` - Replaced by modular components
- `fgsm_attack.py` - Now in `attacks/fgsm.py`
- `adversarial_attack.py` - Old inference script
- `results_nn_run_*/` - Old result directories
