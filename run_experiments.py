################################################
# Load config
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', default='config')
args = parser.parse_args()
config = yaml.safe_load(open(f'{args.config}.yml'))

seed = config['seed']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
test_epochs = config['test_epochs']
gamma = config['gamma']
use_l1 = False

environment_name = config['environment_name']
# options:
# - cifar10
# - mnist
# - spotify # doesn't currently work

################################################
# Imports
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

from tensorflow.keras.datasets import cifar10

from gym import Env
from gym.spaces import Discrete,Box

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.models import clone_model
import tensorflow as tf

from collections import deque

from tqdm import tqdm

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from environment import MnistEnv,Cifar10Env,SpotifyEnv

################################################
# Functions
def indicator(x,y):
    if x == y:
        return 0.
    else:
        return 2.

def kernel(x,y):
    return np.exp((-1.) * np.power(gamma,-2) * indicator(x,y))

def set_environment(env_name = 'mnist'):
    if env_name == 'mnist':
        return MnistEnv()
    elif env_name == 'cifar10':
        return Cifar10Env()
    elif env_name == 'spotify':
        return SpotifyEnv()
    else:
        print('Nope! That does not exist!')
        print('You will get MNIST!')
        return MnistEnv()

def loss_fcn(model,previous_model,states,mask,rewards,loss_type = 'q', use_entropy = False, use_l1 = False, use_mmd = False):
    
    if loss_type == 'q':
    
        preds = model(np.array(states))
        q_values = tf.reduce_sum(preds * mask, axis=1, keepdims=True)
        loss = mean_squared_error(q_values, rewards)
        alg_loss = loss.numpy()
        
        # Calculate entropy
        policy = tf.nn.softmax(preds)
        entropy = tf.reduce_sum((-1) * tf.math.log(policy + 1e-8) * policy)
        if use_entropy:
            loss -= 0.01 * entropy
        
        # Calculate L1 Norm
        if use_l1:
            loss += 0.001 * tf.norm(model.trainable_weights, ord=1)
        
        # Calculate MMD
        policy = tf.nn.softmax(preds)

        prev_preds = previous_model(np.array(states))
        prev_policy = tf.nn.softmax(preds)
        # Edited
        # Forcing uniform distribution
        prev_policy = tf.convert_to_tensor((1/env.action_space.n) * np.ones(shape = prev_policy.numpy().shape))

        prob_products = np.array([np.asmatrix(prev_policy.numpy()[s]).transpose() * np.asmatrix(policy.numpy()[s]) for s in range(len(states))])
        mmd_coefs = np.array([[(kernel_coefs[a] * prob_products[s]).sum() for a in range(10)] for s in range(len(prob_products))])
        mmd_loss = tf.reduce_sum(mmd_coefs * policy)
        if use_mmd:
            loss += 0.01 * mmd_loss
            
        
    elif loss_type == 'po':
        
        rewards = tf.reduce_sum(rewards,axis = 1,keepdims = True)
        preds = model(np.array(states))
        policy = tf.nn.softmax(preds)
        probs = tf.reduce_sum(policy * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean((-1) * rewards * probs)
        alg_loss = loss.numpy()
        
        # Calculate entropy
        entropy = tf.reduce_sum((-1) * tf.math.log(policy + 1e-6) * policy)
        if use_entropy:
            loss -= 0.01 * entropy
        
        # Calculate L1 Norm
        if use_l1:
            loss += 0.001 * tf.norm(model.trainable_weights, ord=1)
        
        # Calculate MMD
        prev_preds = previous_model(np.array(states))
        prev_policy = tf.nn.softmax(preds)
        # Edited
        # Forcing uniform distribution
        prev_policy = tf.convert_to_tensor((1/env.action_space.n) * np.ones(shape = prev_policy.numpy().shape))

        prob_products = np.array([np.asmatrix(prev_policy.numpy()[s]).transpose() * np.asmatrix(policy.numpy()[s]) for s in range(len(states))])
        mmd_coefs = np.array([[(np.multiply(kernel_coefs[a],prob_products[s])).sum() for a in range(10)] for s in range(len(prob_products))])
        mmd_loss = tf.reduce_sum(mmd_coefs * policy)
        if use_mmd:
            loss += 0.01 * mmd_loss

    elif loss_type == 'pg':
        
        rewards = tf.reduce_sum(rewards,axis = 1,keepdims = True)
        preds = model(np.array(states))
        policy = tf.nn.softmax(preds)
        probs = tf.reduce_sum(policy * mask, axis=1, keepdims=True)
        log_probs = tf.math.log(probs + 1e-6)
        loss = tf.reduce_mean((-1) * rewards * log_probs)
        alg_loss = loss.numpy()
        
        # Calculate entropy
        entropy = tf.reduce_sum((-1) * tf.math.log(policy + 1e-6) * policy)
        if use_entropy:
            loss -= 0.01 * entropy
        
        # Calculate L1 Norm
        if use_l1:
            loss += 0.001 * tf.norm(model.trainable_weights, ord=1)
        
        # Calculate MMD
        prev_preds = previous_model(np.array(states))
        prev_policy = tf.nn.softmax(preds)
        # Edited
        # Forcing uniform distribution
        prev_policy = tf.convert_to_tensor((1/env.action_space.n) * np.ones(shape = prev_policy.numpy().shape))

        prob_products = np.array([np.asmatrix(prev_policy.numpy()[s]).transpose() * np.asmatrix(policy.numpy()[s]) for s in range(len(states))])
        mmd_coefs = np.array([[(np.multiply(kernel_coefs[a],prob_products[s])).sum() for a in range(10)] for s in range(len(prob_products))])
        mmd_loss = tf.reduce_sum(mmd_coefs * policy)
        if use_mmd:
            loss += 0.01 * mmd_loss
    
    elif loss_type == 'po_escort':
        
        rewards = tf.reduce_sum(rewards,axis = 1,keepdims = True)
        preds = model(np.array(states))
        numerator = tf.math.pow(tf.math.abs(preds), 2)
        denominator = tf.reduce_sum(numerator, axis = 1, keepdims = True)
        policy = tf.math.divide(numerator, denominator)
        probs = tf.reduce_sum(policy * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean((-1) * rewards * probs)
    
    return loss,alg_loss,entropy,mmd_loss

def build_model():
    
    model = Sequential()

    model.add(Flatten(input_shape = env.reset().shape)) #(1,) + env.reset().shape))

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))

    model.add(Dense(10, activation = 'linear', use_bias = False))

    model.compile()
    
    return model

def train_model(
    model_name = 'linear',
    num_epochs = 10000,
    batch_size = 32,
    loss_type = 'po',
    use_entropy = False,
    use_l1 = False,
    use_mmd = False,
    test_epochs = 20, # number of times to save test data (after initialization)
    temperature = 1.0
):
    
    if use_entropy:
        entropy_used = '_entropy'
    else:
        entropy_used = ''
    
    if use_l1:
        l1_used = '_l1'
    else:
        l1_used = ''
    
    if use_mmd:
        mmd_used = '_mmd'
    else:
        mmd_used = ''
    
    model = build_model()
    previous_model = clone_model(model)

    if loss_type == 'q':
        replay_buffer = deque(maxlen = 10000)
        temperature = 10.
    else:
        replay_buffer = deque(maxlen = batch_size)
        temperature = 10.
        
    optimizer = Adam(learning_rate = 0.000001)

    # index = random.sample(list(range(len(x_test))), 1000)
    index = list(range(len(x_test)))
    x_test_small = [x_test[i] for i in index]
    y_test_small = [y_test[i] for i in index]
    train_losses = []
    train_alg_losses = []
    train_entropy = []
    train_mmd = []
    test_losses = []
    test_actions = []
    test_actions_random = []

    # Initial predictions
    preds = model(np.array(x_test_small))
    actions = [np.argmax(p) for p in preds]
    test_losses.append(accuracy_score(actions,y_test_small))
    test_actions.append(np.histogram(actions, bins = list(range(env.action_space.n + 1)), density = True)[0])
    actions = tf.random.categorical(tf.math.log(tf.nn.softmax(temperature * preds)), 1)
    test_actions_random.append(np.histogram(actions, bins = list(range(env.action_space.n + 1)), density = True)[0])

    for epoch in tqdm(range(num_epochs + batch_size), desc = f'{loss_type}{entropy_used}{l1_used}{mmd_used}', leave = False):

        obs = env.reset()
        obs = obs.reshape((1,) + obs.shape)

        pred = model.predict(obs)
        action = tf.random.categorical(tf.math.log(tf.nn.softmax(temperature * pred)), 1)
        _,reward,_,_ = env.step(action)
        replay_buffer.append((obs,action,reward))

        # Compute loss and train if memory buff is big enough
        if len(replay_buffer) < batch_size:
                pass
        else:
            sample = random.sample(replay_buffer,batch_size)
            states,actions,rewards = list(zip(*sample))
            mask = tf.one_hot([a.numpy().item() for a in actions],10)
            rewards = np.expand_dims(rewards, axis=1) * mask
            with tf.GradientTape() as tape:
                loss,alg_loss,entropy,mmd = loss_fcn(model,previous_model,states,mask,rewards,loss_type, use_entropy, use_l1, use_mmd)
            train_losses.append(loss.numpy())
            train_alg_losses.append(alg_loss)
            train_entropy.append(entropy.numpy())
            train_mmd.append(mmd.numpy())
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            previous_model = clone_model(model)

            if ((epoch - (batch_size - 1)) % (num_epochs // test_epochs) == 0) and (epoch > batch_size - 1):
                
                preds = model(np.array(x_test_small))
                
                # Greedy action selection
                actions = [np.argmax(p) for p in preds]
                test_losses.append(accuracy_score(actions,y_test_small))
                test_actions.append(np.histogram(actions, bins = list(range(env.action_space.n + 1)), density = True)[0])
                
                # Stochastic action selection
                actions = tf.random.categorical(tf.math.log(tf.nn.softmax(temperature * preds)), 1)
                test_actions_random.append(np.histogram(actions, bins = list(range(env.action_space.n + 1)), density = True)[0])

    model.save(f'models/{environment_name}_{model_name}_{loss_type}{entropy_used}{l1_used}{mmd_used}_{seed}.h5')
    
    temp = pd.DataFrame(
        data = test_losses,
        index = [(num_epochs // test_epochs) * i for i in range(1 + test_epochs)]
    ).to_csv(f'data/{environment_name}_{model_name}_testlosses_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')
    temp = pd.DataFrame(
        data = train_losses
    ).to_csv(f'data/{environment_name}_{model_name}_trainlosses_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')
    temp = pd.DataFrame(
        data = train_alg_losses
    ).to_csv(f'data/{environment_name}_{model_name}_trainalglosses_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')
    temp = pd.DataFrame(
        data = train_entropy
    ).to_csv(f'data/{environment_name}_{model_name}_trainentropy_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')
    temp = pd.DataFrame(
        data = train_mmd
    ).to_csv(f'data/{environment_name}_{model_name}_trainmmd_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')
    
    return train_losses,test_losses,test_actions,test_actions_random

def create_plots(model_name,test_actions,test_actions_random,num_epochs,test_epochs,loss_type,use_entropy = False,use_l1 = False,use_mmd = False):
    
    title = f'Action Selections for {environment_name} Test Set\nLoss: {loss_type.upper()}'
    
    if use_entropy:
        entropy_used = '_entropy'
        if use_l1:
            title += ' with entropy and l1 regularization'
        else:
            title += ' with entropy regularization'
    else:
        entropy_used = ''
        if use_l1:
            title += ' with l1 regularization'
    
    if use_l1:
        l1_used = '_l1'
    else:
        l1_used = ''
        
    if use_mmd:
        mmd_used = '_mmd'
        title += ' with MMD regularization'
    else:
        mmd_used = ''

    # Greedy
    df = pd.DataFrame(
        data = test_actions,
        index = [(num_epochs // test_epochs) * i for i in range(1 + test_epochs)]
    )

    df.to_csv(f'data/{environment_name}_{model_name}_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')

    fig = plt.figure(figsize = (10,5))
    ax = plt.subplot(111)
    df.plot.bar(stacked = True, ax = ax, width = 1)#, width = 0.9)
    ax.legend(bbox_to_anchor=(1, 0.5), loc = 'center left', title = 'Action')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Selection Percentages')
    ax.set_title(title)

    plt.savefig(f'images/{environment_name}_{model_name}_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.png', dpi = 250)
    
    # Stochastic
    df = pd.DataFrame(
        data = test_actions_random,
        index = [(num_epochs // test_epochs) * i for i in range(1 + test_epochs)]
    )

    df.to_csv(f'data/{environment_name}_{model_name}_random_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.csv')

    fig = plt.figure(figsize = (10,5))
    ax = plt.subplot(111)
    df.plot.bar(stacked = True, ax = ax, width = 1)#, width = 0.9)
    ax.legend(bbox_to_anchor=(1, 0.5), loc = 'center left', title = 'Action')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Stochastic Selection Percentages')
    ax.set_title(title)

    plt.savefig(f'images/{environment_name}_{model_name}_random_{loss_type}{entropy_used}{l1_used}{mmd_used}_{num_epochs}_{seed}.png', dpi = 250)


################################################
# Run experiments
for loss_type in tqdm(['po','q'], desc = 'Losses'): #['q','po','pg']

    for (use_entropy,use_mmd) in tqdm(
        [(False,False),(True,False),(False,True)],
        desc = 'Regularizers'
    ):

        env = set_environment(environment_name)

        # Set seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

        kernel_coefs = np.array([
            [
                [
                    kernel(a,a_prime) - kernel(a,a_star) for a_prime in range(10)
                ] for a_star in range(10)
            ] for a in range(10)
        ])

        x_test,y_test = env.test_set()

        train_losses,test_losses,test_actions,test_actions_random = train_model(
            '2layers',
            num_epochs,
            batch_size,
            loss_type,
            use_entropy,
            use_l1,
            use_mmd,
            test_epochs
        )

        create_plots(
            '2layers',
            test_actions,
            test_actions_random,
            num_epochs,
            test_epochs,
            loss_type,
            use_entropy,
            use_l1,
            use_mmd
        )
