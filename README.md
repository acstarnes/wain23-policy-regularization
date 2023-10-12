# Policy Entropy Regularization

Source code for the numerical experiments presented in the paper "[Increasing Entropy to Boost Policy Gradient Performance on Personalization Tasks](https://arxiv.org/abs/2310.05324)".

### Setup
1. Install requirements listed in `./requirements.txt`
2. Run an experiment via `python -m main -c {config}`, where `config.yml` is the config file in `./configs/`
3. Visualize results of an experiment via `python -m main -l {logs}`, where `logs.pkl` is the log file in `./logs/`
4. Images will be saved to `./images/`

### Results
![performance](https://github.com/acstarnes/wain23-policy-regularization/assets/38059493/6c3b0d23-2bac-428f-b5f3-de902dcaa39a)
![histograms](https://github.com/acstarnes/wain23-policy-regularization/assets/38059493/f6c3fbac-5a69-47e3-a733-f2e359d26d04)
