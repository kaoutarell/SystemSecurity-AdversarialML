#!/usr/bin/env python3
"""
PGD (Projected Gradient Descent) Attack Implementation

PGD is an iterative variant of FGSM that applies multiple small perturbations
and projects back to the epsilon ball after each step. It's considered one of
the strongest first-order adversarial attacks.

Reference: Madry et al. (2017) "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""

import numpy as np
import tensorflow as tf


def pgd_attack(model, x, y, epsilon, alpha, num_iter, random_start=True, clip_min=0.0, clip_max=1.0):
    """
    Perform PGD attack on a batch of samples.
    
    Args:
        model: Trained Keras model
        x: Input samples (batch_size, features)
        y: True labels (batch_size,)
        epsilon: Maximum perturbation magnitude (L_inf norm)
        alpha: Step size for each iteration
        num_iter: Number of iterations
        random_start: Whether to start from random point in epsilon ball
        clip_min: Minimum value for clipping
        clip_max: Maximum value for clipping
    
    Returns:
        adv_x: Adversarial examples
        perturbation: The actual perturbation added
        loss_history: Loss at each iteration
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    # Initialize adversarial example
    if random_start:
        # Start from random point in epsilon ball
        random_noise = tf.random.uniform(
            shape=x.shape,
            minval=-epsilon,
            maxval=epsilon,
            dtype=tf.float32
        )
        adv_x = x + random_noise
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    else:
        adv_x = tf.identity(x)
    
    loss_history = []
    
    # Iteratively perturb
    for i in range(num_iter):
        adv_x = tf.Variable(adv_x)
        
        with tf.GradientTape() as tape:
            tape.watch(adv_x)
            logits = model(adv_x, training=False)
            logits = tf.squeeze(logits)
            
            # Binary cross-entropy loss
            loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=False)
            loss = tf.reduce_mean(loss)
        
        loss_history.append(float(loss.numpy()))
        
        # Compute gradient
        gradients = tape.gradient(loss, adv_x)
        
        # Take step in direction of gradient (to maximize loss)
        adv_x = adv_x + alpha * tf.sign(gradients)
        
        # Project back to epsilon ball around original input
        perturbation = adv_x - x
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        adv_x = x + perturbation
        
        # Clip to valid range
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    
    perturbation = adv_x - x
    
    return adv_x.numpy(), perturbation.numpy(), loss_history


def pgd_attack_batch(model, X, y, epsilon, alpha, num_iter, batch_size=256, random_start=True):
    """
    Perform PGD attack on entire dataset with batching.
    
    Args:
        model: Trained Keras model
        X: All input samples
        y: All true labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size per iteration
        num_iter: Number of iterations
        batch_size: Batch size for processing
        random_start: Whether to use random initialization
    
    Returns:
        adv_X: All adversarial examples
        perturbations: All perturbations
        loss_histories: Loss history for each batch
    """
    n_samples = len(X)
    adv_X = np.zeros_like(X)
    perturbations = np.zeros_like(X)
    loss_histories = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]
        
        adv_batch, pert_batch, loss_hist = pgd_attack(
            model, X_batch, y_batch, 
            epsilon, alpha, num_iter, 
            random_start=random_start
        )
        
        adv_X[i:end_idx] = adv_batch
        perturbations[i:end_idx] = pert_batch
        loss_histories.append(loss_hist)
    
    return adv_X, perturbations, loss_histories


def evaluate_pgd_robustness(model, X_clean, y_clean, epsilon, alpha, num_iter, 
                            batch_size=256, random_start=True):
    """
    Evaluate model robustness against PGD attack.
    
    Args:
        model: Trained Keras model
        X_clean: Clean input samples
        y_clean: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of PGD iterations
        batch_size: Batch size for processing
        random_start: Whether to use random initialization
    
    Returns:
        Dictionary with evaluation metrics
    """
    X_clean = np.array(X_clean, dtype=np.float32)
    y_clean = np.array(y_clean, dtype=np.float32)
    
    # Evaluate on clean samples
    print(f"\n{'='*80}")
    print(f"Evaluating clean samples...")
    print(f"{'='*80}")
    
    probs_clean = model.predict(X_clean, batch_size=batch_size, verbose=0)
    preds_clean = (probs_clean >= 0.5).astype(int).flatten()
    clean_accuracy = np.mean(preds_clean == y_clean)
    
    print(f"Clean Accuracy: {clean_accuracy:.4f} ({clean_accuracy*100:.2f}%)")
    print(f"Correct predictions: {int(np.sum(preds_clean == y_clean))}/{len(y_clean)}")
    
    # Generate adversarial examples
    print(f"\n{'='*80}")
    print(f"Generating PGD adversarial examples...")
    print(f"  Epsilon: {epsilon}")
    print(f"  Alpha (step size): {alpha}")
    print(f"  Iterations: {num_iter}")
    print(f"  Random start: {random_start}")
    print(f"{'='*80}")
    
    adv_X, perturbations, loss_histories = pgd_attack_batch(
        model, X_clean, y_clean, 
        epsilon, alpha, num_iter,
        batch_size=batch_size,
        random_start=random_start
    )
    
    # Evaluate on adversarial samples
    print(f"\nEvaluating adversarial samples...")
    probs_adv = model.predict(adv_X, batch_size=batch_size, verbose=0)
    preds_adv = (probs_adv >= 0.5).astype(int).flatten()
    adv_accuracy = np.mean(preds_adv == y_clean)
    
    # Calculate metrics
    degradation = clean_accuracy - adv_accuracy
    degradation_pct = (degradation / clean_accuracy * 100) if clean_accuracy > 0 else 0
    
    # Perturbation statistics
    pert_l2 = np.linalg.norm(perturbations, axis=1)
    pert_linf = np.max(np.abs(perturbations), axis=1)
    
    # Success rate (samples that changed prediction)
    prediction_changed = (preds_clean != preds_adv)
    attack_success_rate = np.mean(prediction_changed)
    
    # Per-class analysis
    class_metrics = {}
    for label in [0, 1]:
        mask = y_clean == label
        if np.sum(mask) > 0:
            class_clean_acc = np.mean(preds_clean[mask] == y_clean[mask])
            class_adv_acc = np.mean(preds_adv[mask] == y_clean[mask])
            class_success = np.mean(prediction_changed[mask])
            
            class_name = "Normal" if label == 0 else "Attack"
            class_metrics[class_name] = {
                'samples': int(np.sum(mask)),
                'clean_accuracy': float(class_clean_acc),
                'adv_accuracy': float(class_adv_acc),
                'attack_success': float(class_success)
            }
    
    results = {
        'clean_accuracy': float(clean_accuracy),
        'adversarial_accuracy': float(adv_accuracy),
        'accuracy_degradation': float(degradation),
        'degradation_percentage': float(degradation_pct),
        'attack_success_rate': float(attack_success_rate),
        'perturbation_stats': {
            'l2_mean': float(np.mean(pert_l2)),
            'l2_std': float(np.std(pert_l2)),
            'l2_max': float(np.max(pert_l2)),
            'linf_mean': float(np.mean(pert_linf)),
            'linf_std': float(np.std(pert_linf)),
            'linf_max': float(np.max(pert_linf))
        },
        'class_metrics': class_metrics,
        'predictions_clean': preds_clean,
        'predictions_adversarial': preds_adv,
        'adversarial_examples': adv_X,
        'perturbations': perturbations,
        'loss_histories': loss_histories
    }
    
    return results


def print_pgd_results(results, epsilon, alpha, num_iter):
    """
    Pretty print PGD attack results.
    
    Args:
        results: Dictionary from evaluate_pgd_robustness()
        epsilon: Epsilon value used
        alpha: Alpha value used
        num_iter: Number of iterations used
    """
    print(f"\n{'='*80}")
    print(f"PGD ATTACK RESULTS")
    print(f"{'='*80}")
    
    print(f"\nAttack Configuration:")
    print(f"  Epsilon (ε):        {epsilon}")
    print(f"  Step size (α):      {alpha}")
    print(f"  Iterations:         {num_iter}")
    print(f"  Attack type:        L∞ (infinity norm)")
    
    print(f"\nOverall Performance:")
    print(f"  Clean Accuracy:              {results['clean_accuracy']:.4f} ({results['clean_accuracy']*100:.2f}%)")
    print(f"  Adversarial Accuracy:        {results['adversarial_accuracy']:.4f} ({results['adversarial_accuracy']*100:.2f}%)")
    print(f"  Accuracy Degradation:        {results['accuracy_degradation']:.4f} ({results['degradation_percentage']:.2f}%)")
    print(f"  Attack Success Rate:         {results['attack_success_rate']:.4f} ({results['attack_success_rate']*100:.2f}%)")
    
    print(f"\nPerturbation Statistics:")
    pert = results['perturbation_stats']
    print(f"  L2 Norm:    mean={pert['l2_mean']:.6f}, std={pert['l2_std']:.6f}, max={pert['l2_max']:.6f}")
    print(f"  L∞ Norm:    mean={pert['linf_mean']:.6f}, std={pert['linf_std']:.6f}, max={pert['linf_max']:.6f}")
    
    print(f"\nPer-Class Analysis:")
    for class_name, metrics in results['class_metrics'].items():
        print(f"\n  {class_name} ({metrics['samples']} samples):")
        print(f"    Clean Accuracy:        {metrics['clean_accuracy']:.4f} ({metrics['clean_accuracy']*100:.2f}%)")
        print(f"    Adversarial Accuracy:  {metrics['adv_accuracy']:.4f} ({metrics['adv_accuracy']*100:.2f}%)")
        print(f"    Attack Success Rate:   {metrics['attack_success']:.4f} ({metrics['attack_success']*100:.2f}%)")
    
    print(f"\n{'='*80}")


def compare_epsilon_values(model, X_clean, y_clean, epsilons, alpha_ratio=0.25, 
                          num_iter=40, batch_size=256, random_start=True):
    """
    Compare PGD attack effectiveness across different epsilon values.
    
    Args:
        model: Trained model
        X_clean: Clean samples
        y_clean: True labels
        epsilons: List of epsilon values to test
        alpha_ratio: Ratio of alpha to epsilon (default: 0.25)
        num_iter: Number of iterations
        batch_size: Batch size
        random_start: Whether to use random initialization
    
    Returns:
        Dictionary with results for each epsilon
    """
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"PGD ATTACK - EPSILON COMPARISON")
    print(f"{'='*80}")
    print(f"\nTesting epsilon values: {epsilons}")
    print(f"Alpha ratio: {alpha_ratio} (α = {alpha_ratio} × ε)")
    print(f"Iterations: {num_iter}")
    print(f"Random start: {random_start}")
    
    for epsilon in epsilons:
        alpha = epsilon * alpha_ratio
        
        print(f"\n{'='*80}")
        print(f"Testing ε = {epsilon:.4f}, α = {alpha:.4f}")
        print(f"{'='*80}")
        
        results = evaluate_pgd_robustness(
            model, X_clean, y_clean,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            batch_size=batch_size,
            random_start=random_start
        )
        
        print_pgd_results(results, epsilon, alpha, num_iter)
        
        all_results[epsilon] = results
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"EPSILON COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Epsilon':>10} | {'Clean Acc':>10} | {'Adv Acc':>10} | {'Degradation':>12} | {'Success Rate':>12}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    
    for epsilon in epsilons:
        res = all_results[epsilon]
        print(f"{epsilon:>10.4f} | {res['clean_accuracy']:>10.4f} | "
              f"{res['adversarial_accuracy']:>10.4f} | "
              f"{res['degradation_percentage']:>11.2f}% | "
              f"{res['attack_success_rate']*100:>11.2f}%")
    
    return all_results
