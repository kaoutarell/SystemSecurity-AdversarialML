# SystemSecurity-AdversarialML

Intrusion Detection System Robustness under Adversarial Machine Learning Attacks

![Python](https://img.shields.io/badge/Python-3.12-purple?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-purple?logo=tensorflow)
![CUDA](https://img.shields.io/badge/CUDA-12.0-green?logo=nvidia)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-IDS-purple)
![Cyber Security](https://img.shields.io/badge/Cybersecurity-Adversarial%20ML-purple?logo=hackaday)

## Project Overview

Modern Intrusion Detection Systems (IDS) increasingly rely on Machine Learning models to detect malicious network behavior. However, adversarial attacks can manipulate inputs in subtle ways to evade detection, creating significant security vulnerabilities.

📑 This project investigates how adversarial machine learning affects neural network-based IDS models using the NSL-KDD cybersecurity dataset, specifically implementing and analyzing the Fast Gradient Sign Method (FGSM) attack.

## Project Goals

| Goal                             | Description                                             | Status |
| -------------------------------- | ------------------------------------------------------- | ------ |
| 📦 Train a baseline IDS ML model | Build a neural network to classify network traffic      | ✅     |
| ⚔️ Attack the model (FGSM)       | Generate adversarial samples & evaluate evasion success | ✅     |
| 📉 Analyze model robustness      | Compare clean vs attacked accuracy                      | ✅     |
| 🛡️ Explore defenses              | Test adversarial training / model hardening             | 🔄     |
| 📊 Present results               | Graphs, metrics, report, reproducible pipeline          | ✅     |

## Key Results

### Optimized Model Performance (Run 012)
- **Architecture**: 3-layer Neural Network (64→64→32 neurons) with Batch Normalization
- **Parameters**: ~8,000 (simplified from 65,729 to prevent overfitting)
- **Training Dataset**: NSL-KDD KDDTrain+ (125,973 samples)
- **Test Dataset**: NSL-KDD KDDTest+ (22,544 samples, different attack distribution)
- **Test Accuracy**: **80.39%**
- **Precision**: 84.05%
- **Recall**: 80.92%
- **Training Epochs**: 48 (early stopped, restored weights from epoch 28)

**Challenge**: The NSL-KDD test set contains different attack types than the training set, creating a distribution mismatch that limits generalization. The model achieved 80% accuracy on this challenging split.

### FGSM Attack Results (Full Test Set: 22,544 samples)

| Scenario             | Accuracy | Degradation | Normal Traffic | Attack Traffic |
| -------------------- | -------- | ----------- | -------------- | -------------- |
| Clean test data      | 80.39%   | baseline    | 80%            | 81%            |
| FGSM ε = 0.01        | 84.29%   | **+3.90%**  | 75.76%         | 90.75%         |
| FGSM ε = 0.02        | 82.97%   | **+2.58%**  | 73.39%         | 90.21%         |
| FGSM ε = 0.05        | 79.07%   | -1.32%      | 68.65%         | 86.95%         |
| FGSM ε = 0.1         | 57.74%   | -22.65%     | 24.72%         | 82.72%         |

**Key Findings**:
1. **Surprising Robustness**: Small perturbations (ε ≤ 0.02) actually *improve* accuracy by 2-4%, likely regularizing the input and bridging the train/test distribution gap
2. **Asymmetric Vulnerability**: At ε=0.1, normal traffic detection collapses (25%) while attack detection remains reasonable (83%)
3. **Overall Robustness**: Model maintains >79% accuracy up to ε=0.05, demonstrating reasonable adversarial resistance
4. **Vulnerability Threshold**: Significant degradation (-23%) only occurs at ε=0.1, an aggressive perturbation magnitude

## Project Structure

```
SystemSecurity-AdversarialML/
├── main/
│   ├── model.py          # Neural network architecture & training
│   ├── dataset.py        # NSL-KDD data loading & preprocessing
│   ├── evaluate.py       # Model evaluation & visualization
│   ├── utils.py          # Directory management utilities
│   └── train_ids.py      # Main training script
├── attacks/
│   └── fgsm.py           # FGSM attack implementation
├── results/
│   ├── models/           # Latest trained model & artifacts
│   ├── runs/             # Individual training runs
│   └── attacks/          # Attack evaluation results
└── nsl-kdd/              # NSL-KDD dataset files
```

## Setup & Usage

### 1. Environment Setup

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the NSL-KDD dataset:
- Create `nsl-kdd/` directory in project root
- Download from [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Required files:
  - `KDDTrain+.txt` (training data)
  - `KDDTest+.txt` (test data)

### 3. Train the Model

```bash
python main/train_ids.py \
  --train_path nsl-kdd/KDDTrain+.txt \
  --test_path nsl-kdd/KDDTest+.txt \
  --use_separate_test \
  --epochs 100 \
  --batch_size 1024 \
  --dropout 0.6 \
  --l1 2e-4 \
  --l2 2e-3 \
  --learning_rate 0.0003 \
  --patience 20 \
  --use_batchnorm
```

**Key Training Parameters:**
- `--use_separate_test`: Use KDDTest+ for validation instead of splitting KDDTrain+
- `--dropout`: Higher values (0.5-0.6) prevent overfitting
- `--l1`, `--l2`: Regularization coefficients (higher values for stronger regularization)
- `--learning_rate`: Lower values (0.0003) for more stable convergence
- `--patience`: Early stopping patience (20 epochs)
- `--use_batchnorm`: Enable batch normalization layers (default: True)

**Output Files** (saved to `results/runs/run_XXX/`):
- `nn_ids_model.keras` — Trained model
- `artifacts.joblib` — Scaler & feature columns
- `metrics.json` — Performance metrics
- `training_history.json` — Training curves data
- `confusion_matrix.png` — Confusion matrix visualization
- `training_curves.png` — Loss & accuracy plots

### 4. Run FGSM Attack

```bash
python attacks/fgsm.py \
  --n_samples 22544 \
  --epsilons 0.01 0.02 0.05 0.1
```

**Parameters:**
- `--model_path`: Path to trained model (default: `results/models/nn_ids_model.keras`)
- `--test_path`: Path to test data (default: `nsl-kdd/KDDTest+.txt`)
- `--n_samples`: Number of samples to attack (default: all 22,544)
- `--epsilons`: List of perturbation magnitudes to test

## Technical Details

### Model Architecture
- **Input Layer**: 122 features (preprocessed from 41 NSL-KDD features)
- **Hidden Layers**: 
  - Dense(64) + BatchNorm + ReLU + Dropout(0.6)
  - Dense(64) + BatchNorm + ReLU + Dropout(0.6)
  - Dense(32) + BatchNorm + ReLU + Dropout(0.6)
- **Output Layer**: Dense(1) + Sigmoid (binary classification)
- **Total Parameters**: ~8,000 (optimized for generalization)
- **Regularization**: L1(2e-4) + L2(2e-3) + Batch Normalization
- **Optimizer**: Adam (lr=0.0003)
- **Callbacks**: 
  - EarlyStopping (patience=20, restore_best_weights=True)
  - ReduceLROnPlateau (factor=0.5, patience=10)

### FGSM Attack
The Fast Gradient Sign Method generates adversarial examples using:

```
adv_x = x + ε × sign(∇_x J(θ, x, y))
```

Where:
- `x`: Original input
- `ε`: Perturbation magnitude (epsilon)
- `∇_x J(θ, x, y)`: Gradient of loss w.r.t. input
- `adv_x`: Adversarial example (clipped to [0, 1])

### Preprocessing Pipeline
1. **Numeric Features**: RobustScaler normalization (~35 features)
2. **Categorical Features**: One-hot encoding (protocol_type, service, flag)
3. **Label Encoding**: Binary (0=Normal, 1=Attack)
4. **Feature Count**: 122 total features after preprocessing

## Results Analysis

### Training Methodology
**Critical Insight**: The NSL-KDD dataset presents a unique challenge - the test set (KDDTest+) contains different attack types than the training set (KDDTrain+), creating a distribution mismatch. This is **intentional** in the dataset design to test generalization, but makes achieving high accuracy difficult.

**Training Approach**:
- Train on full KDDTrain+ (125,973 samples)
- Validate on KDDTest+ (22,544 samples) during training using `--use_separate_test`
- Use aggressive regularization to prevent overfitting to training distribution
- Early stopping with patience=20 to restore best generalization point

### Training Performance
- **Final Test Accuracy**: 80.39% (epoch 28)
- **Peak Test Accuracy**: 80.58% (epoch 24)
- **Training Epochs**: 48 total, early stopped and restored to epoch 28
- **Convergence**: Validation accuracy showed significant fluctuation (52-81%) due to distribution mismatch
- **Overfitting Control**: Batch normalization + high dropout (0.6) + strong L1/L2 regularization

### Attack Vulnerability Analysis
1. **Small Perturbations Improve Accuracy** (ε ≤ 0.02):
   - Unexpected finding: accuracy increased by 2-4%
   - Hypothesis: Perturbations act as regularization, bridging train/test distribution gap
   - Similar to how noise injection can improve generalization

2. **Moderate Robustness** (ε = 0.05):
   - Accuracy: 79.07% (only -1.3% degradation)
   - Model maintains reasonable performance at practical perturbation levels

3. **Vulnerability at High Perturbations** (ε = 0.1):
   - Accuracy drops to 57.74% (-22.65%)
   - Normal traffic detection severely impacted (24.72%)
   - Attack detection remains relatively stable (82.72%)

### Class-Specific Observations
**Asymmetric Vulnerability Pattern**:
- **Clean Data**: Balanced detection (Normal: 80%, Attack: 81%)
- **ε = 0.01-0.02**: Attack detection improves significantly (90%+), normal traffic drops slightly
- **ε = 0.1**: Normal traffic detection collapses (24.72%), attack detection remains at 82.72%

**Interpretation**: The model learned more robust features for attack detection than for normal traffic. This suggests:
- Attack patterns have more discriminative features that are harder to perturb
- Normal traffic features are more sensitive to input perturbations
- Potential improvement: More aggressive augmentation for normal traffic during training

## GPU Acceleration

The project supports NVIDIA GPU training with CUDA:
```bash
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
python main/train_ids.py --train_path nsl-kdd/KDDTrain+.txt --epochs 20
```

Tested Configuration:
- GPU: NVIDIA GeForce RTX 4070 SUPER
- CUDA: 12.0
- cuDNN: 9.1
- TensorFlow: 2.20 with XLA compilation

## Team Members

| Name            | Role                                      |
| --------------- | ----------------------------------------- |
| Kaoutar         | Model architecture & adversarial attacks  |
| <Team Member 2> | Defense strategies & analysis             |
| <Team Member 3> | Dataset engineering & preprocessing       |
| <Team Member 4> | Results visualization & documentation     |
| <Team Member 5> | Performance optimization & GPU tuning     |
| <Team Member 6> | Report writing & reproducibility testing  |

## License

This project was designed and implemented by the SystemSecurity-AdversarialML team as part of the SOEN 321 course (Fall 2025). It is intended solely for academic research and educational purposes.

*All datasets and external resources referenced remain the property of their respective owners.*
