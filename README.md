# Prototype-Based Joint Embedding Method (PB&J)

This code repository implements the Prototype-Based Joint Embedding method in the paper "Improving Explainability of Softmax Classifiers Using a Prototype-Based Joint Embedding Method" by Hilarie Sit, Brendan Keith, and Karianne Bergen. PB&J is an interpretable-by-design, explainable AI approach that provides instance-based explanations for softmax classifiers using similarity of the model input to examples within the training dataset.

**To train the model**:
1. Specify dataset, model, and training hyperparameters in utils.py
2. Run train.py

**To generate prototypes using the stochastic sampling approach**:
1. Specify path to trained models in prototypes.py
2. Run prototypes.py

**To evaluate OOD detection performance using the centroid-based approach**:
1. Specify path to trained models in ood_detection.py
2. Run ood_detection.py
