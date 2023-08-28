# wain23-mmd-regularization

A refactored version of the code which may or may not work.

### Setup
1. Install requirements listed in `./requirements.txt`
2. Run sample experiment via `python -m main`. It should train two agents in under a minute and save logs to `./logs/mnist.pkl`
3. Run hyperparameter search on MNIST via `python -m main -c mnist_search`, it will probably have to run overnight
4. Run hyperparameter search on CIFAR10 via `python -m main -c cifar10_search`, it will take forever
