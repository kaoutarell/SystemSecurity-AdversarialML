# SystemSecurity-AdversarialML

Intrusion Detection under Adversarial Machine Learning Attacks

![Python](https://img.shields.io/badge/Python-3.9+-purple?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Model-purple?logo=pytorch)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-IDS-purple)
![Cyber Security](https://img.shields.io/badge/Cybersecurity-Adversarial%20ML-purple?logo=hackaday)

## Project Overview

Modern Intrusion Detection Systems (IDS) increasingly rely on Machine Learning models to detect malicious network behavior.
However, adversarial attacks can manipulate inputs in subtle ways to evade detection, even when changes are invisible to humans.

📑 This project investigates the adversarial robustness of neural network-based Intrusion Detection Systems (IDS) using the NSL-KDD dataset. We demonstrate that models achieving 99.67% accuracy on clean data catastrophically fail under adversarial attacks (dropping to 49.92%), and show that semantically-constrained adversarial training restores robustness to 92-95% across all attack scenarios.

## Project Main Questions

1. **How vulnerable are ML-based IDS to adversarial attacks?**
   - White-box attacks (full model access)
   - Black-box transfer attacks (surrogate models)
   - Query-based attacks (minimal knowledge)

2. **Can adversarial training defend against these attacks?**
   - While maintaining acceptable clean performance
   - Using semantically-valid network traffic constraints

3. **What are the practical security implications?**
   - Gap between reported accuracy and actual robustness
   - Deployment recommendations for security-critical ML

## Methodology
### Attack Framework
| Attack Type | Attacker Knowledge | Success Rate (Baseline) |
|-------------|-------------------|------------------------|
| **White-box PGD** | Complete (architecture, parameters, gradients) | 50.08% |
| **Transfer (Simple)** | Black-box (trained surrogate) | 29.20% |
| **Transfer (Deep)** | Black-box (deep surrogate) | 31.36% |
| **Transfer (Ensemble)** | Black-box (ensemble surrogates) | 34.34% |
| **Query-based** | Minimal (5% seed data + 10k queries) | 41.93% |

### Semantic Constraints
All attacks maintain **deployment-viable network traffic**:
- **Count features** → Integer values only
- **Binary flags** → {0, 1}
- **Non-negative features** → ≥ 0 (byte counts, durations)
- **Rate features** → [0, 1] (error rates, connection rates)
- **Categorical features** → Valid one-hot encodings (protocol, service, flag)

### Defense Strategy
**Adversarial Training** (Madry et al.):
- Generate semantically-constrained PGD adversarial examples during training
- Mix adversarial + clean data in each batch
- Force model to learn robust decision boundaries
- Parameters: ε=0.5, α=0.008, 20-40 PGD iterations

## Code - Google Colab
> Run Experiments on Google Colab

**🔗 [Open in Google Colab](https://colab.research.google.com/drive/18pOhNwcn8_JBMaP6womffT9MH4U8_UI_?usp=sharing)**

(no local installation needed)

## Results Summary

| Scenario | Baseline Accuracy | Robust Accuracy | Improvement |
|----------|------------------|-----------------|-------------|
| **Clean Data** | 99.67% | 95.82% | -3.85pp |
| **White-box PGD** | 49.92% | 92.61% | **+42.69pp** |
| **Transfer (Simple)** | 70.80% | 94.40% | **+23.60pp** |
| **Transfer (Deep)** | 68.64% | 95.07% | **+26.43pp** |
| **Transfer (Ensemble)** | 65.66% | 95.07% | **+29.41pp** |
| **Query-based** | 58.07% | 95.43% | **+37.36pp** |

**Attack Success Rate Reduction:** 5.2× to 9.2× across all scenarios  

## Experimental Findings

### Finding 1: Clean Accuracy ≠ Adversarial Robustness
> **99.67% clean accuracy → 49.92% under attack**
>
> Standard training provides **zero security guarantees** against adaptive adversaries.

### Finding 2: Black-box Attacks Are Highly Effective
> **Transfer success: 29-34% | Query-based success: 42%**
>
> Attackers need minimal knowledge to evade detection—dramatically lowering the attack barrier.

### Finding 3: Adversarial Training Works
> **Robust model: 92-95% accuracy across all attack scenarios**
>
> Semantically-constrained adversarial training provides **practical robustness** with only 3.85pp clean accuracy cost.

### Finding 4: Semantic Constraints Are Essential
> **All adversarial examples pass deployment validity checks**
>
> Without constraints, defenses learn to detect nonsensical inputs rather than realistic threats.

---
**Course:** SOEN 321 - Information System Security  
**Institution:** Concordia University, Montreal, Quebec, Canada  

## License

This project was designed and implemented by the SystemSecurity-AdversarialML team as part of the SOEN 321 course (Fall 2025). It is intended solely for academic research and educational purposes.

_All datasets and external resources referenced remain the property of their respective owners._
