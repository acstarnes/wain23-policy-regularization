import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import deque

np.set_printoptions(precision=3, suppress=True)


class Agent:

    def __init__(self, name, params, random_seed):
        self.name = name
        self.__dict__.update(params)
        self.random_seed = random_seed

        # set up replay buffer and loss function
        if self.loss_type == 'q':
            self.buffer = deque(maxlen=10000)
        elif self.loss_type == 'pg':
            self.buffer = deque(maxlen=self.batch_size)
        else:
            raise NotImplementedError(f'Loss type "{self.loss_type}" is not implemented...')

    def train(self, env, logs, steps):
        """Configure and train the agent."""
        self.env = env
        self.logs = logs
        eval_steps = deque(steps[1:])
        self.build_model()
        self.get_kernel_coefs()

        # train the model
        self.env.fix_random_seed()
        pbar = tqdm(total=steps[-1], ascii=True, desc=f'Training {self.name}-agent')
        for step in range(0, steps[-1], self.batch_size):
            # compute loss and adjust model weights
            with tf.GradientTape() as tape:

                # observe a batch of transitions
                states = self.env.reset(num=self.batch_size)
                logits = self.model(states) / self.temperature
                actions = tf.random.categorical(logits, num_samples=1).numpy()
                _, rewards, _, _ = self.env.step(actions)
                self.buffer.extend([*zip(states, actions, rewards)])

                if self.loss_type == 'pg':
                    # compute pg-loss
                    probs = tf.nn.softmax(logits, axis=1)
                    logprobs = tf.math.log(tf.gather_nd(probs + 1e-8, actions, batch_dims=1))
                    loss = -tf.reduce_mean(rewards * logprobs)

                elif self.loss_type == 'q':
                    # compute mse-loss
                    inds = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
                    states, actions, rewards = zip(*[self.buffer[i] for i in inds])
                    S = np.stack(states)
                    A = np.stack(actions)
                    R = np.stack(rewards)
                    Q = tf.gather_nd(self.model(S), A, batch_dims=1)
                    loss = tf.keras.metrics.mean_squared_error(R, Q)
                    # compute batch probabilities for regularization
                    probs = tf.nn.softmax(self.model(S) / self.temperature, axis=1)

                # add regularization
                for reg, coef in self.regularization.items():
                    if coef != 0:
                        loss += coef * self.compute_regularization(reg, probs)

            # backpropagate the loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # evaluate the model
            while eval_steps and (step + self.batch_size >= eval_steps[0]):
                eval_steps.popleft()
                self.evaluate_model()

            pbar.update(self.batch_size)
        pbar.close()

    def compute_regularization(self, reg, probs):
        """Compute a given regularization term for the model."""
        n = self.env.num_classes

        # entropy regularization
        if reg == 'ent':
            ent = tf.reduce_mean(tf.reduce_sum(-probs * tf.math.log(probs + 1e-8), axis=1))
            return -ent

        # jensen-shannon regularization
        if reg == 'js':
            mid = (probs + 1/n) / 2
            js = tf.reduce_mean(\
                tf.reduce_sum(-probs * tf.math.log(probs/mid + 1e-8), axis=1)\
                + tf.reduce_sum(-1/n * tf.math.log((1/n)/mid), axis=1))
            return -js

        # mmd regularization
        if reg == 'mmd':
            mmd_coefs = np.matmul(probs/n, self.kernel_coefs.transpose([1,2,0])).sum(axis=0)
            mmd_loss = tf.reduce_mean(tf.reduce_sum(mmd_coefs * probs, axis=1))
            return mmd_loss

        # hellinger regularization
        if reg == 'hl':
            hl = tf.reduce_mean(\
                tf.reduce_sum(tf.square(tf.sqrt(probs + 1e-8) - np.sqrt(1/n)), axis=1))
            return hl

        # total variation regularization
        if reg == 'tv':
            tv = tf.reduce_mean(tf.reduce_max(tf.abs(probs - 1/n), axis=1))
            return tv

        # l1 regularization
        if reg == 'l1':
            l1 = tf.reduce_sum([tf.norm(weight, ord=1)
                                for weight in self.model.trainable_weights[::2]])
            return l1

        # l2 regularization
        if reg == 'l2':
            l2 = tf.reduce_sum([tf.norm(weight, ord=2)
                                for weight in self.model.trainable_weights[::2]])
            return l2

        raise NotImplementedError(f'Regularization "{reg}" is not defined...')

    def build_model(self):
        """Set up feed-forward neural network for a given agent."""
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # construct the model
        self.model = tf.keras.Sequential(name=self.name)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.env.num_classes, activation=None))

        # compile and evaluate the model
        optimizer = getattr(tf.keras.optimizers, self.optimizer['name'])(**self.optimizer)
        if self.env.name == 'spotify':
            self.model.compile(optimizer=optimizer,
                               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                               metrics=tf.keras.metrics.TopKCategoricalAccuracy(k=5))
        else:
            self.model.compile(optimizer=optimizer,
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        self.evaluate_model()

    def evaluate_model(self, stochastic=False):
        """Evaluate the model and record metrics."""
        # compute model loss and accuracy
        loss, acc = self.model.evaluate(self.env.x_ts, self.env.y_ts, verbose=0)
        self.logs['loss'][self.name].append(loss)
        self.logs['accuracy'][self.name].append(acc)

        # compute policy entropy
        logits = self.model(self.env.x_ts) / self.temperature
        probs = tf.nn.softmax(logits, axis=1)
        entropy = tf.reduce_mean(tf.reduce_sum(-probs * tf.math.log(probs + 1e-8), axis=1))
        self.logs['entropy'][self.name].append(entropy)

        # compute action histogram
        actions = tf.random.categorical(logits, num_samples=1)
        hist = np.histogram(actions, bins=np.arange(self.env.num_classes+1), density=True)[0]
        self.logs['histogram'][self.name].append(hist)

        # compute model reward
        if stochastic:
            reward = self.env.evaluate_predictions(actions.numpy())
            self.logs['reward'][self.name].append(reward)
        else:
            actions = logits.numpy().argmax(axis=1)
            reward = self.env.evaluate_predictions(actions)
            self.logs['reward'][self.name].append(reward)

    def get_kernel_coefs(self):
        """No idea what this is, treating as a blackbox for now."""
        gamma = 1.414

        def indicator(x,y):
            if x == y:
                return 0.
            else:
                return 2.

        def kernel(x,y):
            return np.exp((-1.) * np.power(gamma,-2) * indicator(x,y))

        self.kernel_coefs = np.array([[[kernel(a,a_prime) - kernel(a,a_star)\
                                        for a_prime in range(self.env.num_classes)]\
                                        for a_star in range(self.env.num_classes)]\
                                        for a in range(self.env.num_classes)])

