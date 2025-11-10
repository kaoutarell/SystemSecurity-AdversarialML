# Reproducibility Guide

This document explains how to reproduce the training results for all three models.

## Random Seed Control

All training scripts now support the `--seed` parameter to ensure reproducible results:

- **Default seed**: 42
- **Custom seed**: Use `--seed <number>` to set a different seed

### What is Controlled by the Seed

1. **NumPy random operations** (data shuffling, etc.)
2. **TensorFlow random operations** (weight initialization, dropout, etc.)
3. **Train/validation split** (always uses `random_state=42`)

### What is NOT Controlled

1. **GPU non-determinism**: GPU operations may have slight variations due to floating-point precision
2. **Multi-threading**: Some operations may vary slightly in multi-threaded environments
3. **Hardware differences**: Different GPUs/CPUs may produce slightly different results

## Reproduction Commands

### 1. IDS (Dense Neural Network) - 78.40% Accuracy

```bash
cd /mnt/data/Projects/SystemSecurity-AdversarialML/main

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
../.venv/bin/python train_ids.py \
  --train_path ../nsl-kdd/KDDTrain+.txt \
  --test_path ../nsl-kdd/KDDTest+.txt \
  --use_separate_test \
  --epochs 100 \
  --batch_size 256 \
  --dropout 0.3 \
  --l1 1e-5 \
  --l2 1e-4 \
  --learning_rate 0.006 \
  --patience 10 \
  --validation_split 0.2 \
  --seed 42
```

**Expected Results:**
- Accuracy: ~78.40%
- Precision: ~96.40%
- Recall: ~64.46%
- F1-Score: ~77.26%

---

### 2. CNN + ECA Attention - 75.86% Accuracy

```bash
cd /mnt/data/Projects/SystemSecurity-AdversarialML/main

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
../.venv/bin/python train_cnn_attention.py \
  --train_path ../nsl-kdd/KDDTrain+.txt \
  --test_path ../nsl-kdd/KDDTest+.txt \
  --use_separate_test \
  --epochs 100 \
  --batch_size 2048 \
  --dropout 0.4 \
  --l1 1e-5 \
  --l2 5e-4 \
  --learning_rate 0.001 \
  --patience 15 \
  --image_size 12 \
  --seed 42
```

**Expected Results:**
- Accuracy: ~75.86%
- Precision: ~91.86%
- Recall: ~63.19%
- F1-Score: ~74.87%

---

### 3. SAAE-DNN - 80.95% Accuracy ⭐ BEST

```bash
cd /mnt/data/Projects/SystemSecurity-AdversarialML/main

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
../.venv/bin/python train_saae_dnn.py \
  --train_path ../nsl-kdd/KDDTrain+.txt \
  --test_path ../nsl-kdd/KDDTest+.txt \
  --binary \
  --latent_dims 90 80 \
  --hidden_dims 50 25 10 \
  --pretrain_epochs 100 \
  --train_epochs 100 \
  --batch_size 256 \
  --pretrain_lr 0.05 \
  --train_lr 0.006 \
  --dropout 0.3 \
  --l1 1e-5 \
  --l2 1e-4 \
  --patience 10 \
  --validation_split 0.2 \
  --use_separate_test \
  --seed 42
```

**Expected Results:**
- Accuracy: ~80.70-81.00% (varies slightly due to GPU non-determinism)
- Precision: ~92-96%
- Recall: ~69-72%
- F1-Score: ~80-81%

**Note**: SAAE-DNN results may vary ±0.5% between runs even with the same seed due to:
- GPU floating-point operations
- Pretraining convergence variations
- Early stopping timing

---

## Improving Reproducibility

For even more deterministic results, you can:

1. **Disable GPU parallelism** (slower but more deterministic):
   ```bash
   export TF_DETERMINISTIC_OPS=1
   ```

2. **Force single-threaded execution**:
   ```bash
   export OMP_NUM_THREADS=1
   export TF_NUM_INTRAOP_THREADS=1
   export TF_NUM_INTEROP_THREADS=1
   ```

3. **Use CPU instead of GPU** (fully deterministic but much slower):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

## Results Variance

Based on multiple runs, here is the expected variance:

| Model | Expected Accuracy | Typical Range | Seed Impact |
|-------|------------------|---------------|-------------|
| IDS | 78.40% | ±0.2% | Low |
| CNN | 75.86% | ±0.3% | Low |
| SAAE-DNN | 80.70-81.00% | ±0.5% | Medium |

The SAAE-DNN model has higher variance because:
- Two-stage training (pretraining + fine-tuning)
- Greedy layer-wise pretraining can converge differently
- More complex architecture with multiple autoencoders

## Best Practices

1. **Always specify `--seed`** for reproducibility
2. **Run multiple times** (3-5 runs) and report mean ± std
3. **Document environment**:
   - TensorFlow version
   - CUDA version
   - GPU model
   - Python version
4. **Save training logs** for debugging variance issues

## Environment Information

```bash
# Check versions
python --version
pip show tensorflow
nvcc --version
nvidia-smi
```

## Contact

For questions about reproducibility, please refer to the main README.md or check the training scripts.
