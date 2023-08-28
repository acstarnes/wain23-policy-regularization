import os
import yaml
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import deque, defaultdict

from environments import *


def get_kernel_coefs():
    """No idea what this is, treating as a blackbox for now."""
    gamma = 1.414

    def indicator(x,y):
        if x == y:
            return 0.
        else:
            return 2.

    def kernel(x,y):
        return np.exp((-1.) * np.power(gamma,-2) * indicator(x,y))

    kernel_coefs = np.array([[[kernel(a,a_prime) - kernel(a,a_star)\
        for a_prime in range(10)] for a_star in range(10)] for a in range(10)])

    return kernel_coefs


class Experiment:

    def __init__(self, config_file):
        """Set up experiment from a given config file."""
        try:
            configs = yaml.safe_load(open(f'./configs/{config_file}.yml'))
            self.name = config_file
            self.__dict__.update(configs)
        except:
            raise OSError(f'Could not load "./configs/{config_file}.yml" file...')

    def train_agents(self):
        """Train agents with the provided parameters."""
        self.logs = {'loss': defaultdict(list),
                     'accuracy': defaultdict(list),
                     'entropy': defaultdict(list),
                     'histogram': defaultdict(list)}
        self.models = {}
        self.eval_steps = np.linspace(0, self.num_steps, self.num_eval)
        for agent in self.agents:
            self.env = self.setup_environment()
            self.models[agent] = self.train(agent)
        self.save_logs()

    def setup_environment(self):
        """Set up contextual bandit environment."""
        if self.environment_name == 'mnist':
            return MNISTEnv(self.random_seed)
        elif self.environment_name == 'cifar10':
            return CIFAR10Env(self.random_seed)
        else:
            raise NameError(f'Environment "{self.environment_name}" is not defined...')

    def train(self, agent):
        """Configure and train the agent."""
        loss_type = self.agents[agent]['loss_type']
        batch_size = self.agents[agent]['batch_size']
        temp = self.agents[agent]['temperature']

        # set up replay buffer and loss function
        if loss_type == 'q':
            buffer = deque(maxlen=10000)
            compute_loss = self.compute_loss_q
        elif loss_type == 'pg':
            buffer = deque(maxlen=batch_size)
            compute_loss = self.compute_loss_pg
        else:
            raise NotImplementedError(f'Loss type "{loss_type}" is not implemented...')

        # build and train the model
        model = self.build_model(agent)
        for step in tqdm(range(self.num_steps), desc=f'Training {agent}-agent'):

            # make a prediction
            state = np.expand_dims(self.env.reset(), axis=0)
            logits = model.predict(state) / temp
            action = tf.random.categorical(logits, num_samples=1).numpy().item()
            _, reward, _, _ = self.env.step(action)
            buffer.append((state, action, reward))

            # compute loss and adjust model weights
            if (step + 1) % batch_size == 0:
                with tf.GradientTape() as tape:
                    loss = compute_loss(model, agent, buffer, batch_size, temp)

                # backpropagate the loss
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # evaluate the model
            if step + 1 in self.eval_steps:
                self.evaluate_model(model, agent)

        return model

    def compute_loss_q(self, model, agent, buffer, batch_size, temp):
        """Compute loss value for the q-agent."""
        # sample a batch of transitions
        inds = np.random.choice(len(buffer), size=batch_size, replace=False)
        S, A, R = zip(*[buffer[i] for i in inds])
        batch = tf.squeeze(tf.stack(S))

        # compute mse-loss
        Q = tf.gather_nd(model(batch), list(zip(range(len(A)), A)))
        loss = tf.keras.metrics.mean_squared_error(R, Q)

        # add regularization
        for reg, coef in self.agents[agent]['regularization'].items():
            if coef != 0:
                loss += coef * self.compute_regularization(reg, model, batch, temp)

        return loss

    def compute_loss_pg(self, model, agent, buffer, batch_size, temp):
        """Compute loss value for the pg-agent."""
        # sample a batch of transitions
        S, A, R = zip(*buffer)
        batch = tf.squeeze(tf.stack(S))

        # compute pg-loss
        probs = tf.nn.softmax(model(batch)/temp, axis=1)
        logprobs = tf.gather_nd(tf.math.log(probs), list(zip(range(len(A)), A)))
        loss = -tf.reduce_mean(R * logprobs)

        # add regularization
        for reg, coef in self.agents[agent]['regularization'].items():
            if coef != 0:
                loss += coef * self.compute_regularization(reg, model, batch, temp)

        return loss

    def compute_regularization(self, reg, model, batch, temp):
        """Compute a given regularization term for the model."""
        # entropy regularization
        if reg == 'entropy':
            logits = model(batch) / temp
            probs = tf.nn.softmax(logits, axis=1)
            entropy = tf.reduce_mean(tf.reduce_sum(-probs * tf.math.log(probs + 1e-8), axis=1))
            return -entropy

        # l1 regularization
        if reg == 'l1':
            l1 = tf.reduce_sum([tf.norm(weight, ord=1) for weight in model.trainable_weights[::2]])
            return l1

        # l2 regularization
        if reg == 'l2':
            l2 = tf.reduce_sum([tf.norm(weight, ord=2) for weight in model.trainable_weights[::2]])
            return l2

        # mmd regularization
        if reg == 'mmd':
            logits = model(batch) / temp
            probs = tf.nn.softmax(logits, axis=1)
            prob_products = probs / self.env.num_classes
            mmd_coefs = np.array([[tf.reduce_sum(kernel_coef * prob_prod)
                                   for kernel_coef in kernel_coefs]
                                   for prob_prod in prob_products])
            mmd_loss = tf.reduce_sum(mmd_coefs * probs)
            return mmd_loss

        if reg not in ['entropy', 'l1', 'l2', 'mmd']:
            raise NotImplementedError(f'Regularization "{reg}" is not defined...')

    def build_model(self, agent):
        """Set up feed-forward neural network for a given agent."""
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # construct the model
        model = tf.keras.Sequential(name=agent)
        model.add(tf.keras.layers.Flatten(input_shape=self.env.reset().shape))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.num_classes, activation='linear'))

        # compile and evaluate the model
        opt_params = self.agents[agent]['optimizer']
        model.compile(optimizer=getattr(tf.keras.optimizers, opt_params['name'])(**opt_params),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.evaluate_model(model, agent)

        return model

    def evaluate_model(self, model, agent):
        """Evaluate the model and record metrics."""
        # compute model loss and accuracy
        loss, acc = model.evaluate(self.env.x_ts, self.env.y_ts, verbose=0)
        self.logs['loss'][agent].append(loss)
        self.logs['accuracy'][agent].append(acc)

        # compute policy entropy
        logits = model(self.env.x_ts) / self.agents[agent]['temperature']
        probs = tf.nn.softmax(logits, axis=1)
        entropy = tf.reduce_mean(tf.reduce_sum(-probs * tf.math.log(probs + 1e-8), axis=1))
        self.logs['entropy'][agent].append(entropy)

        # compute action histogram
        actions = tf.random.categorical(logits, num_samples=1)
        hist = np.histogram(actions, bins=np.arange(self.env.num_classes+1), density=True)[0]
        self.logs['histogram'][agent].append(hist)

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
    kernel_coefs = get_kernel_coefs()
    exp.train_agents()

