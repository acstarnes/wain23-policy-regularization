"""
Setup various contextual bandit environments.
There is a lot of code overlap between the environments but I'm going to
leave it like that for the simplicity of future modifications.
"""

import gym
import numpy as np
import tensorflow as tf


class MNISTEnv(gym.Env):
    """Set up MNIST classification task as a contextual bandit environment."""

    def __init__(self, random_seed=0):
        super().__init__()
        self.name = 'mnist'
        self.random_seed = random_seed
        self.setup()

    def fix_random_seed(self):
        """Fix random seed for reproducibility."""
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def setup(self):
        """Setup the environment."""
        self.load_data()
        self.setup_state_space()
        self.setup_action_space()
        self.setup_reward_space()

    def load_data(self):
        """Load MNIST dataset."""
        self.labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        (x_tr, y_tr), (x_ts, y_ts) = tf.keras.datasets.mnist.load_data()
        self.x_tr = (x_tr / 255).astype('float32')
        self.x_ts = (x_ts / 255).astype('float32')
        self.y_tr = y_tr.flatten()
        self.y_ts = y_ts.flatten()

    def setup_state_space(self):
        """Setup state space."""
        self.num_states = self.y_tr.size
        self.observation_space =\
            gym.spaces.Box(low=0., high=1., shape=self.x_tr[0].shape, dtype=np.float32)

    def setup_action_space(self):
        """Setup action space."""
        self.num_classes = np.unique(self.y_tr).size
        self.action_space = gym.spaces.Discrete(self.num_classes)

    def setup_reward_space(self):
        """Compute reward matrix."""
        # maximum reward 1, average reward 0
        self.R = (self.num_classes * np.eye(self.num_classes) - 1) / (self.num_classes - 1)

    def compute_reward(self, state_index, action_index):
        """Compute the reward value for a given state and an action index."""
        r = self.R[self.y_tr[state_index], action_index.flatten()]
        return r

    def reset(self, num=1):
        """Randomly sample a state."""
        self.state_index = np.random.randint(self.num_states, size=num)
        self.state = self.x_tr[self.state_index]
        return self.state

    def step(self, action_index):
        """Take an action for an observed state and compute the reward."""
        reward = self.compute_reward(self.state_index, action_index)
        done = True
        info = {}
        return self.state_index, reward, done, info


class CIFAR10Env(gym.Env):
    """Set up CIFAR10 classification task as a contextual bandit environment."""

    def __init__(self, random_seed=0):
        super().__init__()
        self.name = 'cifar10'
        self.random_seed = random_seed
        self.setup()

    def fix_random_seed(self):
        """Fix random seed for reproducibility."""
        self.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def setup(self):
        """Setup the environment."""
        self.load_data()
        self.setup_state_space()
        self.setup_action_space()
        self.setup_reward_space()

    def load_data(self):
        """Load CIFAR10 dataset."""
        self.labels = ['plane', 'car', 'bird', 'cat', 'deer',\
                       'dog', 'frog', 'horse', 'ship', 'truck']
        (x_tr, y_tr), (x_ts, y_ts) = tf.keras.datasets.cifar10.load_data()
        self.x_tr = (x_tr / 255).astype('float32')
        self.x_ts = (x_ts / 255).astype('float32')
        self.y_tr = y_tr.flatten()
        self.y_ts = y_ts.flatten()

    def setup_state_space(self):
        """Setup state space."""
        self.num_states = self.y_tr.size
        self.observation_space =\
            gym.spaces.Box(low=0., high=1., shape=self.x_tr[0].shape, dtype=np.float32)

    def setup_action_space(self):
        """Setup action space."""
        self.num_classes = np.unique(self.y_tr).size
        self.action_space = gym.spaces.Discrete(self.num_classes)

    def setup_reward_space(self):
        """Compute reward matrix."""
        # maximum reward 1, average reward 0
        self.R = (self.num_classes * np.eye(self.num_classes) - 1) / (self.num_classes - 1)

    def compute_reward(self, state_index, action_index):
        """Compute the reward value for a given state and an action index."""
        r = self.R[self.y_tr[state_index], action_index.flatten()]
        return r

    def reset(self, num=1):
        """Randomly sample a state."""
        self.state_index = np.random.randint(self.num_states, size=num)
        self.state = self.x_tr[self.state_index]
        return self.state

    def step(self, action_index):
        """Take an action for an observed state and compute the reward."""
        reward = self.compute_reward(self.state_index, action_index)
        done = True
        info = {}
        return self.state_index, reward, done, info


if __name__ == '__main__':

    env = MNISTEnv()
    env = CIFAR10Env()
