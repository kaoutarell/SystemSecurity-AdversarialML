### Models

Each test will be performed on 

DNN

CNN + Attention

Attention + DNN


### Dataset

issue with test dataset and online available results, give examples

some scrap the test dataset others don't massive impact on accruacy

OOD generalization for test dataset

given our experiments are not trying to evaluate the performance of the model on unseen attack types, but more so the exact effect and scale of different attacks and attack prevention, it would be better to use only the training dataset split between train, validation and test to achieve ~99% accruacy. By doing so, it will be easier to measure the impact of our experiments, as we won't be getting confusion between our model simply failing as a result of poor model accuracy

### Attacks

fgsm

### Defense


Using nsl kdd dataset for network traffic intrusion detection

Are you familiar with this dataset

Searching for state of the art model to be able to get best accuracy and I ran into a lot of misleading info online, moslty having to do with the use of train and test dataset. 

So I found that either models gave 99% accurcay or around 80% accuracy depending on wihch was used

In theory, should be using the test dataset for proper results, but it will be harder to properly measure the effects of attacks and defenses since the amount of false positives 

if follow paper attacks etc, what expecting on notable findings and contribution

Chose one of the approaches,



Show pertubations fall within normal distributions, could not be detected as anomalies



Adversarial training - generalizes across threat models
Model robustness curves - useful for evaluation

PGD Attack

Find epsilon such that modifications stray least from normal distribution

Research using white-box attacks (PGD, FGSM) is valuable because:

1. **Worst-Case Analysis**: "If we defend against the strongest possible attack (white-box), we're more likely to be robust to weaker attacks (black-box)"

2. **Transferability**: "Models defended against white-box attacks on themselves also become more robust to black-box attacks from other models"

 Substitute Mode Training using Jacobian based dataset augmentation.
 
Transfer attack

PGD Adversarial Training