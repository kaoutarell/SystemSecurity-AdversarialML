#!/usr/bin/env python3
"""
Run PGD (Projected Gradient Descent) Attack on Trained DNN Models

This script performs PGD adversarial attacks to evaluate model robustness.

Usage:
    python run_pgd_attack.py --model_dir results/runs/run_dnn_test_001 --epsilon 0.1 --num_iter 40
    python run_pgd_attack.py --model_dir results/runs/run_dnn_test_001 --compare_epsilons
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import json
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from attacks.pgd import (
    pgd_attack_batch,
    evaluate_pgd_robustness, 
    print_pgd_results,
    compare_epsilon_values
)
from dataset import load_data, preprocess_nsl_kdd


def load_model_and_artifacts(model_dir):
    """
    Load trained model and preprocessing artifacts.
    
    Args:
        model_dir: Path to model directory
    
    Returns:
        model: Loaded Keras model
        artifacts: Dictionary with scaler and feature_columns
    """
    model_dir = Path(model_dir)
    
    # Load model
    model_path = model_dir / 'best_model.keras'
    if not model_path.exists():
        model_path = model_dir / 'final_model.keras'
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Architecture: {len(model.layers)} layers")
    print(f"  Parameters: {model.count_params():,}")
    
    # Load artifacts
    artifacts_path = model_dir / 'artifacts.joblib'
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path}")
    
    print(f"\nLoading artifacts from: {artifacts_path}")
    artifacts = joblib.load(artifacts_path)
    print(f"✓ Artifacts loaded")
    print(f"  Features: {len(artifacts['feature_columns'])}")
    
    return model, artifacts


def main():
    parser = argparse.ArgumentParser(
        description='PGD Attack on Trained DNN IDS Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model and artifacts')
    parser.add_argument('--test_path', type=str, default='nsl-kdd/KDDTest+.txt',
                        help='Path to test data')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    
    # PGD attack parameters
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Maximum perturbation (L-infinity norm)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Step size (default: epsilon/4)')
    parser.add_argument('--num_iter', type=int, default=40,
                        help='Number of PGD iterations')
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='Use random initialization')
    parser.add_argument('--no_random_start', dest='random_start', action='store_false',
                        help='Disable random initialization')
    
    # Epsilon comparison mode
    parser.add_argument('--compare_epsilons', action='store_true',
                        help='Compare multiple epsilon values')
    parser.add_argument('--epsilon_values', type=float, nargs='+',
                        default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
                        help='Epsilon values to compare')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/attacks',
                        help='Output directory for results')
    parser.add_argument('--save_adversarial', action='store_true',
                        help='Save adversarial examples to disk')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=9281,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("="*80)
    print("PGD ADVERSARIAL ATTACK - DNN IDS MODEL")
    print("="*80)
    print(f"\nModel directory: {args.model_dir}")
    print(f"Test data: {args.test_path}")
    print(f"Random seed: {args.seed}")
    
    # Load model and artifacts
    print(f"\n{'='*80}")
    print("LOADING MODEL AND ARTIFACTS")
    print(f"{'='*80}")
    
    model, artifacts = load_model_and_artifacts(args.model_dir)
    
    # Load and preprocess test data
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    
    print(f"Loading from: {args.test_path}")
    test_df = load_data(args.test_path)
    
    # Limit samples if specified
    if args.n_samples:
        test_df = test_df.iloc[:args.n_samples]
        print(f"Limited to {args.n_samples} samples")
    
    X_test, y_test, _, _ = preprocess_nsl_kdd(
        test_df,
        scaler=artifacts['scaler'],
        feature_columns=artifacts['feature_columns'],
        use_statistical_filter=True
    )
    
    print(f"\n✓ Loaded {len(X_test)} test samples")
    print(f"  Normal samples (0): {np.sum(y_test == 0):,}")
    print(f"  Attack samples (1): {np.sum(y_test == 1):,}")
    print(f"  Features: {X_test.shape[1]}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set alpha if not specified
    if args.alpha is None:
        args.alpha = args.epsilon / 4.0
    
    if args.compare_epsilons:
        # Compare multiple epsilon values
        print(f"\n{'='*80}")
        print("EPSILON COMPARISON MODE")
        print(f"{'='*80}")
        
        all_results = compare_epsilon_values(
            model=model,
            X_clean=X_test,
            y_clean=y_test,
            epsilons=args.epsilon_values,
            alpha_ratio=0.25,
            num_iter=args.num_iter,
            batch_size=args.batch_size,
            random_start=args.random_start
        )
        
        # Save comparison results
        comparison_file = output_dir / f'pgd_epsilon_comparison_{Path(args.model_dir).name}.json'
        comparison_data = {
            'model_dir': str(args.model_dir),
            'test_path': args.test_path,
            'n_samples': len(X_test),
            'num_iter': args.num_iter,
            'random_start': args.random_start,
            'seed': args.seed,
            'results': {}
        }
        
        for epsilon, results in all_results.items():
            comparison_data['results'][str(epsilon)] = {
                'clean_accuracy': results['clean_accuracy'],
                'adversarial_accuracy': results['adversarial_accuracy'],
                'accuracy_degradation': results['accuracy_degradation'],
                'degradation_percentage': results['degradation_percentage'],
                'attack_success_rate': results['attack_success_rate'],
                'perturbation_stats': results['perturbation_stats'],
                'class_metrics': results['class_metrics']
            }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n✓ Comparison results saved to: {comparison_file}")
        
    else:
        # Single epsilon attack
        print(f"\n{'='*80}")
        print("SINGLE EPSILON ATTACK MODE")
        print(f"{'='*80}")
        
        results = evaluate_pgd_robustness(
            model=model,
            X_clean=X_test,
            y_clean=y_test,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_iter=args.num_iter,
            batch_size=args.batch_size,
            random_start=args.random_start
        )
        
        print_pgd_results(results, args.epsilon, args.alpha, args.num_iter)
        
        # Save results
        results_file = output_dir / f'pgd_attack_{Path(args.model_dir).name}_eps{args.epsilon:.3f}.json'
        results_data = {
            'model_dir': str(args.model_dir),
            'test_path': args.test_path,
            'n_samples': len(X_test),
            'attack_params': {
                'epsilon': args.epsilon,
                'alpha': args.alpha,
                'num_iter': args.num_iter,
                'random_start': args.random_start,
                'seed': args.seed
            },
            'results': {
                'clean_accuracy': results['clean_accuracy'],
                'adversarial_accuracy': results['adversarial_accuracy'],
                'accuracy_degradation': results['accuracy_degradation'],
                'degradation_percentage': results['degradation_percentage'],
                'attack_success_rate': results['attack_success_rate'],
                'perturbation_stats': results['perturbation_stats'],
                'class_metrics': results['class_metrics']
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # Save adversarial examples if requested
        if args.save_adversarial:
            adv_file = output_dir / f'pgd_adversarial_{Path(args.model_dir).name}_eps{args.epsilon:.3f}.npz'
            np.savez_compressed(
                adv_file,
                adversarial_examples=results['adversarial_examples'],
                perturbations=results['perturbations'],
                clean_predictions=results['predictions_clean'],
                adversarial_predictions=results['predictions_adversarial'],
                true_labels=y_test
            )
            print(f"✓ Adversarial examples saved to: {adv_file}")
    
    print(f"\n{'='*80}")
    print("PGD ATTACK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
