import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from main.model import load_model
from main.dataset import load_and_preprocess_file


def fgsm_attack(model, x, y, epsilon):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=False)
        logits = tf.squeeze(logits)
        loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=False)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, x)
    
    adv_x = x + epsilon * tf.sign(gradients)
    
    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
    
    return adv_x.numpy(), gradients.numpy()


def evaluate_robustness(model, X_clean, y_clean, epsilons):
    results = {
        'epsilons': epsilons,
        'clean_accuracy': None,
        'adversarial_accuracies': [],
        'accuracy_degradation': [],
        'predictions_clean': None,
        'predictions_adversarial': {}
    }
    
    X_clean = tf.cast(X_clean, tf.float32).numpy()
    y_clean_np = np.array(y_clean, dtype=np.float32)
    
    print("\n" + "="*70)
    print("CLEAN SAMPLES EVALUATION")
    print("="*70)
    probs_clean = model.predict(X_clean, verbose=0)
    preds_clean = (probs_clean >= 0.5).astype(int).flatten()
    clean_accuracy = np.mean(preds_clean == y_clean_np)
    results['clean_accuracy'] = clean_accuracy
    results['predictions_clean'] = preds_clean
    
    print(f"\nClean Accuracy: {clean_accuracy:.4f} ({clean_accuracy*100:.2f}%)")
    print(f"Correct predictions: {int(np.sum(preds_clean == y_clean_np))}/{len(y_clean_np)}")
    
    print("\n" + "="*70)
    print("ADVERSARIAL ATTACK EVALUATION (FGSM)")
    print("="*70)
    
    for epsilon in epsilons:
        print(f"\nEpsilon: {epsilon}")
        print("-" * 70)
        
        adv_x, grads = fgsm_attack(model, X_clean, y_clean_np, epsilon)
        
        probs_adv = model.predict(adv_x, verbose=0)
        preds_adv = (probs_adv >= 0.5).astype(int).flatten()
        adv_accuracy = np.mean(preds_adv == y_clean_np)
        
        degradation = clean_accuracy - adv_accuracy
        degradation_pct = (degradation / clean_accuracy * 100) if clean_accuracy > 0 else 0
        
        results['adversarial_accuracies'].append(adv_accuracy)
        results['predictions_adversarial'][epsilon] = preds_adv
        results['accuracy_degradation'].append(degradation)
        
        print(f"  Adversarial Accuracy:     {adv_accuracy:.4f} ({adv_accuracy*100:.2f}%)")
        print(f"  Accuracy Degradation:     {degradation:.4f} ({degradation_pct:.2f}%)")
        print(f"  Correct predictions:      {int(np.sum(preds_adv == y_clean_np))}/{len(y_clean_np)}")
        
        for label in [0, 1]:
            mask = y_clean_np == label
            if np.sum(mask) > 0:
                class_acc = np.mean(preds_adv[mask] == y_clean_np[mask])
                class_name = "Normal" if label == 0 else "Attack"
                print(f"    {class_name:8s} ({np.sum(mask):3.0f} samples): {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        print(f"  Gradient Statistics (w.r.t. perturbation):")
        print(f"    Mean:     {np.mean(grads):.6f}")
        print(f"    Std Dev:  {np.std(grads):.6f}")
        print(f"    Min:      {np.min(grads):.6f}")
        print(f"    Max:      {np.max(grads):.6f}")
    
    return results


def print_summary(results):
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70)
    
    print(f"\nClean Accuracy:              {results['clean_accuracy']:.4f} ({results['clean_accuracy']*100:.2f}%)")
    print(f"\nAdversarial Accuracies:")
    
    for eps, acc, deg in zip(results['epsilons'], results['adversarial_accuracies'], results['accuracy_degradation']):
        deg_pct = (deg / results['clean_accuracy'] * 100) if results['clean_accuracy'] > 0 else 0
        print(f"  ε = {eps:5.3f}:  {acc:.4f} ({acc*100:6.2f}%)  │  Degradation: {deg:.4f} ({deg_pct:6.2f}%)")
    
    print(f"\nKey Observations:")
    
    if results['accuracy_degradation']:
        max_deg_idx = np.argmax(results['accuracy_degradation'])
        worst_eps = results['epsilons'][max_deg_idx]
        worst_acc = results['adversarial_accuracies'][max_deg_idx]
        worst_deg = results['accuracy_degradation'][max_deg_idx]
        print(f"  - Most vulnerable to attacks at ε = {worst_eps} (accuracy drops to {worst_acc:.4f})")
    
    initial_acc = results['clean_accuracy']
    final_acc = results['adversarial_accuracies'][-1]
    if final_acc < initial_acc * 0.5:
        print(f"  - Model exhibits significant vulnerability (>50% accuracy drop at max ε)")
    else:
        print(f"  - Model shows reasonable robustness (accuracy drop < 50% at max ε)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FGSM Adversarial Attack on IDS Model')
    parser.add_argument('--model_path', type=str, default='results/models/nn_ids_model.keras',
                        help='Path to trained model')
    parser.add_argument('--artifacts', type=str, default='results/models/artifacts.joblib',
                        help='Path to artifacts (scaler, features)')
    parser.add_argument('--test_path', type=str, default='nsl-kdd/KDDTest+.txt',
                        help='Path to test data')
    parser.add_argument('--n_samples', type=int, default=22544,
                        help='Number of samples to test (default: all)')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1],
                        help='Epsilon values to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("="*70)
    print("FGSM ADVERSARIAL ATTACK - IDS Model Robustness Evaluation")
    print("="*70)
    
    print(f"\nLoading model from {args.model_path}")
    model = load_model(args.model_path)
    print('✓ Model loaded successfully.')
    
    print(f'\nLoading test data from {args.test_path}')
    X_test, y_test, _ = load_and_preprocess_file(
        args.test_path, 
        args.artifacts, 
        n_samples=args.n_samples
    )
    
    if y_test is None:
        raise ValueError("Could not extract labels from test data. Ensure outcome column is present.")
    
    y_test = np.array(y_test, dtype=np.int32)
    
    print(f'✓ Loaded {len(X_test)} test samples')
    print(f'  Normal samples:  {np.sum(y_test == 0)}')
    print(f'  Attack samples:  {np.sum(y_test == 1)}')
    
    print(f'\nTesting FGSM attack with epsilon values: {args.epsilons}')
    results = evaluate_robustness(model, X_test, y_test, args.epsilons)
    
    print_summary(results)
    
    print("\n" + "="*70)
    print("Done.")
    print("="*70)
