# SystemSecurity-AdversarialML

Intrusion Detection under Adversarial Machine Learning Attacks

![Python](https://img.shields.io/badge/Python-3.9+-purple?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Model-purple?logo=pytorch)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-IDS-purple)
![Cyber Security](https://img.shields.io/badge/Cybersecurity-Adversarial%20ML-purple?logo=hackaday)

## Project Overview

Modern Intrusion Detection Systems (IDS) increasingly rely on Machine Learning models to detect malicious network behavior.
However, adversarial attacks can manipulate inputs in subtle ways to evade detection, even when changes are invisible to humans.

📑 This project investigates how adversarial machine learning affects IDS models, specifically using the NSL-KDD cyber-security dataset.

## Our Goals

| Goal                             | Description                                             |
| -------------------------------- | ------------------------------------------------------- |
| 📦 Train a baseline IDS ML model | Build a neural network to classify network traffic      |
| ⚔️ Attack the model (FGSM)       | Generate adversarial samples & evaluate evasion success |
| 📉 Analyze model robustness      | Compare clean vs attacked accuracy                      |
| 🛡️ Explore defenses              | Test adversarial training / model hardening             |
| 📊 Present results               | Graphs, metrics, report, reproducible pipeline          |

## Expected Results

We expect to observe the following

- High baseline accuracy (not perfect)
- Significant performance drop under adversarial attack
- IDS becomes vulnerable even to small perturbations
- Defense training improves resilience, but at cost of accuracy

Expectations :
| Scenario | Accuracy |
| -------------------------- | ----------------------------- |
| Clean test data | ~75–90% |
| FGSM ε = 0.01 | noticeable drop |
| FGSM ε = 0.1 | large accuracy collapse |
| After adversarial training | higher adversarial robustness |

## Setup & Usage

After cloning the repo :

### 1. Set up the env

*Mac & Linux OSs*

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*Windows*

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Datasets (download locally)

> **⚠️ VERY IMPORTANT ⚠️** : since the datasets are HUGE, they're not pushed to the repo.

- Create a folder on root lvl of the project called : data
- Download datasets (NSL-KDD) in /data (created above):
  [KDD Test+ Txt](https://www.kaggle.com/datasets/hassan06/nslkdd?select=KDDTest%2B.txt)
  [KDD Train+ Txt](https://www.kaggle.com/datasets/hassan06/nslkdd?select=KDDTrain%2B.txt)

3. Train the model (commands)

```
python train.py --train data/KDDTrain+.txt --test data/KDDTest+.txt --out_dir results --epochs 15 --batch_size 256
```

> 🚨 Watch out for the format of the datasets, it should be (.txt) not .arff or any other extension. The code relies on it.

**Expected Output** :
In results/:

- `model.pth` — trained model
- `artifacts.joblib` — scaler + encoders
- `metrics.txt` — performance
- `training curves` (\*.png)

## Current Status

| Stage               | Status                               |
| ------------------- | ------------------------------------ |
| Data Preprocessing  | ✅ Completed                         |
| Model Training      | ✅ Neural network trained on NSL-KDD |
| Baseline Accuracy   | ✅ ~78% (Normal vs Attack)           |
| FGSM Implementation | 🔄 In Progress                       |
| Defense Training    | 🔄 To-Do                             |
| Final Report        | 🔄 Pending                           |

**Files produced so far**

| File                  | Purpose                              |
| --------------------- | ------------------------------------ |
| `train.py`            | Data prep + NN training pipeline     |
| `results/model.pth`   | Saved model weights                  |
| `results/metrics.txt` | Baseline accuracy & confusion matrix |
| `results/*.png`       | Loss & accuracy curves               |

## Team Members

| Name            | Roles                                 |
| --------------- | ------------------------------------- |
| Kaoutar         | Model training & pipeline development |
| <Team Member 2> | Adversarial attack implementation     |
| <Team Member 3> | Defense strategies & analysis         |
| <Team Member 4> | Report & evaluation                   |
| <Team Member 5> | Dataset engineering                   |
| <Team Member 6> | Documentation & reproducibility       |

## License

This project was designed and implemented by the SystemSecurity-AdversarialML team as part of the SOEN 321 course (Fall 2025). It is intended solely for academic research and educational purposes.

_All datasets and external resources referenced remain the property of their respective owners._
