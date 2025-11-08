#!/bin/bash
# End-to-end pipeline test

echo "========================================="
echo "Testing Complete Pipeline"
echo "========================================="

echo ""
echo "1. Training model (2 epochs)..."
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
.venv/bin/python main/train.py \
    --train_path nsl-kdd/KDDTrain+.txt \
    --epochs 2 \
    --batch_size 2048 \
    --dropout 0.1 2>&1 | grep -E "Accuracy:|saved to:"

echo ""
echo "2. Running FGSM attack (500 samples)..."
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit \
.venv/bin/python attacks/fgsm.py \
    --n_samples 500 \
    --epsilons 0.01 0.05 0.1 2>&1 | grep -E "Clean Accuracy:|ε ="

echo ""
echo "========================================="
echo "✅ Pipeline test complete!"
echo "========================================="
