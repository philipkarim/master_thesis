# Master Thesis - Quantum Computing and Quantum Machine Learning
## Title: Quantum computing: Machine learning with emphasize on Variational Quantum Boltzmann machines
## Philip K. Sørli Niane
### University of Oslo - Department of physics

## Main folders
[`results/`](https://github.com/philipkarim/master_thesis/tree/main/src/results) contains all results regarding discriminative- and generative learning, in addition to simulation of quantum systems using VarITE.

[`rbm/`](https://github.com/philipkarim/master_thesis/tree/main/src/rbm) contains all code regarding the Gaussian-binary RBM utilized to compute quantum mechanical ground state computations.

## Main code files
### General files
[`main`](https://github.com/philipkarim/master_thesis/tree/main/src/main.py) starts off the different functions and is used to define parameters.

[`simulation`](https://github.com/philipkarim/master_thesis/tree/main/src/simulation.py) is used as a script dealing with fork-paralellization of multiple runs in paralell.

[`varQITE`](https://github.com/philipkarim/master_thesis/tree/main/src/varQITE.py) is a class taking care of the VarQITE part of the VarQBM.

[`utils`](https://github.com/philipkarim/master_thesis/tree/main/src/utils.py) handles utility functions and extra functions used around the repository.

[`analyzer`](https://github.com/philipkarim/master_thesis/tree/main/src/analyzer.py) is used as a plotter script of results.

[`BM`](https://github.com/philipkarim/master_thesis/tree/main/src/BM.py) handles the training regarding the classical RBM utilizing scikit-learn.

[`ml_methods_class`](https://github.com/philipkarim/master_thesis/tree/main/src/ml_methods_class.py) handles the machine learning methods utilized through scikit-learn.

### Discriminative learning
[`train_supervised`](https://github.com/philipkarim/master_thesis/tree/main/src/train_supervised.py) is used for discriminative learning, doing the training part by sending datasets and parameters chosen

[`fraud_classification`](https://github.com/philipkarim/master_thesis/tree/main/src/fraud_classification.py) handles fraud classification data before the dataset is sent into [`train_supervised`](https://github.com/philipkarim/master_thesis/tree/main/src/train_supervised.py).

[`quantum_mnist`](https://github.com/philipkarim/master_thesis/tree/main/src/quantum_mnist.py) handles handwritten digit recognition before the dataset is sent into [`train_supervised`](https://github.com/philipkarim/master_thesis/tree/main/src/train_supervised.py).

[`franke`](https://github.com/philipkarim/master_thesis/tree/main/src/franke.py) handles Franke's function before the data is sent into [`train_supervised`](https://github.com/philipkarim/master_thesis/tree/main/src/train_supervised.py).

[`NN_class`](https://github.com/philipkarim/master_thesis/tree/main/src/NN_class.py) A neural network class used to encode features into the VarQBM utilizing pytorch.

### Quantum mechanical computations
[`quantum_simulation`](https://github.com/philipkarim/master_thesis/tree/main/src/quantum_simulation.py) is used to simulate quantum systems using VarITE.