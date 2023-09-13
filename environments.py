"""
Setup various contextual bandit environments.
There is a lot of code overlap between the environments but I'm going to
leave it like that for the simplicity of future modifications.
"""

import gym
import numpy as np
import pandas as pd
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
        self.num_classes = len(self.labels)
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

    def evaluate_predictions(self, actions):
        """Compute the average reward on the test set."""
        r = self.R[self.y_ts, actions.flatten()]
        return r.mean()


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
        self.num_classes = len(self.labels)
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

    def evaluate_predictions(self, actions):
        """Compute the average reward on the test set."""
        r = self.R[self.y_ts, actions.flatten()]
        return r.mean()


class CIFAR100Env(gym.Env):
    """Set up CIFAR100 classification task as a contextual bandit environment."""

    def __init__(self, random_seed=0):
        super().__init__()
        self.name = 'cifar100'
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
        """Load CIFAR100 dataset."""
        self.labels = list(map(str, range(100)))
        (x_tr, y_tr), (x_ts, y_ts) = tf.keras.datasets.cifar100.load_data()
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
        self.num_classes = len(self.labels)
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

    def evaluate_predictions(self, actions):
        """Compute the average reward on the test set."""
        r = self.R[self.y_ts, actions.flatten()]
        return r.mean()


class SpotifyEnv(gym.Env):
    '''Set up a contextual bandit environment based on Spotify data.'''

    def __init__(self, seed=0):
        super().__init__()
        self.name = 'spotify'
        self.random_seed = seed
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
        # generate test set
        np.random.seed(0)
        self.x_ts = self.reset(num=10000)
        ##self.y_ts = np.matmul(self.x_ts, self.R).argmax(axis=1)
        self.y_ts = np.matmul(self.x_ts, self.R)

    def load_data(self):
        """Load Spotify data."""
        self.genres = self.load_spotify_data_train()
        self.tracks = self.load_spotify_data_test()
        self.labels = list(map(str, range(len(self.tracks))))

    def load_spotify_data_train(self):
        '''Compute average feature vector for each music genre.'''
        self.info_train = pd.read_csv('./spotify/spotify_genres.csv', index_col=0)
        genres = {}
        for genre in self.info_train['genre']:
            features = pd.read_csv(f'./spotify/data/{genre}.csv', index_col=0)
            genres[genre] = dict(features.select_dtypes(include=np.number).mean())
        genres = pd.DataFrame(genres).transpose().sort_index()
        return genres

    def load_spotify_data_test(self):
        '''Load test tracks from each test data file.'''
        self.info_test = pd.read_csv('./spotify/spotify_actions.csv', index_col=0)
        tracks = []
        for playlist in self.info_test['genre']:
            tracks.append(pd.read_csv(f'./spotify/data/{playlist}.csv', index_col=0))
        tracks = pd.concat(tracks, ignore_index=True)
        return tracks

    def setup_state_space(self):
        """Setup state space."""
        self.state_dim = len(self.genres)
        self.observation_space =\
            gym.spaces.Box(low=0., high=1., shape=(self.state_dim,), dtype=np.float32)

    def setup_action_space(self):
        """Setup action space."""
        self.num_classes = len(self.labels)
        self.action_space = gym.spaces.Discrete(self.num_classes)

    def setup_reward_space(self):
        """Compute reward matrix."""
        genre_vals = np.array(self.genres.values.tolist())
        track_vals = np.array(self.tracks.drop('id', axis=1).values.tolist())
        self.R = np.matmul(genre_vals - genre_vals.mean(axis=0),\
                           (track_vals - track_vals.mean(axis=0)).T)
        self.neg, self.pos = -.1, .1

    def compute_reward(self, state, action):
        """Compute the reward value for a given state and an action index."""
        r = np.matmul(state, self.R)[range(len(action)), action.flatten()]
        r = (1*(r > self.pos) - 1*(r < self.neg)).astype(float)
        return r

    def reset(self, num=1):
        """Randomly sample a state."""
        sparsity = .05 + .2 * np.random.rand(num)
        num_prefs = (sparsity * self.state_dim).astype(int)
        inds = [np.random.choice(np.arange(self.state_dim), replace=False, size=n) for n in num_prefs]
        self.state = np.zeros((num,self.state_dim))
        for i, ind in enumerate(inds):
            self.state[i][ind] = 1.
        return self.state

    def step(self, action):
        """Take an action for an observed state and compute the reward."""
        reward = self.compute_reward(self.state, action)
        done = True
        info = {}
        return self.state, reward, done, info

    def evaluate_predictions(self, actions):
        """Compute the average reward on the test set."""
        r = self.compute_reward(self.x_ts, actions.flatten())
        return r.mean()



if __name__ == '__main__':

    env = MNISTEnv()
    env = CIFAR10Env()
    env = CIFAR100Env()
    env = SpotifyEnv()

