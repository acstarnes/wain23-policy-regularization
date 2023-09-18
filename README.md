# Policy Entropy Regularization

Source code for the numerical experiments presented in the paper "Increasing Entropy to Boost Policy Gradient Performance on Personalization Tasks".

### Setup
1. Install requirements listed in `./requirements.txt`
2. Run an experiment via `python -m main -c {config}`, where `config.yml` is the config file in `./configs/`
3. Visualize results of an experiment via `python -m main -l {logs}`, where `logs.pkl` is the log file in `./logs/`
4. Images will be saved to `./images/`

### Results
![results](https://github.com/acstarnes/wain23-policy-regularization/assets/38059493/a63a7ffc-a64b-409e-bd00-596122c40be6)
![histograms](https://github.com/acstarnes/wain23-policy-regularization/assets/38059493/5c6e80ae-5edb-4139-9fa8-e0fdeff3511a)

