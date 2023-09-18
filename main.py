import os
import yaml
import pickle
import argparse
import numpy as np
from collections import defaultdict

from environments import *
from agent import *


class Experiment:

    def __init__(self, config_file):
        """Set up experiment from a given config file."""
        try:
            configs = yaml.safe_load(open(f'./configs/{config_file}.yml'))
            self.name = config_file
            self.__dict__.update(configs)
        except:
            raise OSError(f'Could not load "./configs/{config_file}.yml" file...')
        self.configure_experiment()

    def configure_experiment(self):
        """Initialize variables."""
        self.env = self.setup_environment()
        self.eval_steps = np.linspace(0, self.num_steps, self.num_eval, dtype=int)
        self.logs = {'exp_name': self.name,
                     'steps': self.eval_steps,
                     'loss': defaultdict(list),
                     'accuracy': defaultdict(list),
                     'entropy': defaultdict(list),
                     'histogram': defaultdict(list),
                     'reward': defaultdict(list)}

    def setup_environment(self):
        """Set up contextual bandit environment."""
        if self.environment_name == 'mnist':
            return MNISTEnv(self.random_seed)
        elif self.environment_name == 'cifar10':
            return CIFAR10Env(self.random_seed)
        elif self.environment_name == 'cifar100':
            return CIFAR100Env(self.random_seed)
        elif self.environment_name == 'spotify':
            return SpotifyEnv(self.random_seed)
        else:
            raise NameError(f'Environment "{self.environment_name}" is not defined...')

    def train_agents(self):
        """Train agents with the provided parameters."""
        self.agents = {}
        for name, params in self.agent_params.items():
            agent = Agent(name, params, self.random_seed)
            agent.train(self.env, self.logs, self.eval_steps.copy())
            self.agents[agent.name] = agent
        self.save_logs()

    def save_logs(self):
        """Save experiment data."""
        os.makedirs('./logs/', exist_ok=True)
        with open(f'./logs/{self.name}.pkl', 'wb') as logfile:
            pickle.dump(self.logs, logfile)


if __name__ == '__main__':

    # parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='mnist',
                        help='Name of the config file in "./configs/"')
    args = parser.parse_args()

    # set up experiment and train the agents
    exp = Experiment(args.config)
    exp.train_agents()

