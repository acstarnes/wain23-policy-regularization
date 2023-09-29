import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='Set2', font='monospace', font_scale=1.)


class Visualization:

    def __init__(self, logs):
        """Read the log data."""
        if isinstance(logs, str):
            try:
                with open(f'./logs/{logs}.pkl', 'rb') as logfile:
                    self.logs = pickle.load(logfile)
            except:
                raise OSError(f'Cannot open "./logs/{logs}.pkl" file...')
        else:
            self.logs = logs
        self.savedir = f'./images/{self.logs["exp_name"]}'
        os.makedirs(self.savedir, exist_ok=True)

    def plot(self, show=True, legend=True):
        """Plot saved metrics."""
        self.plot_accuracy(show=show, legend=legend)
        self.plot_entropy(show=show, legend=legend)
        self.plot_reward(show=show, legend=legend)
        if not legend:
            self.plot_legend(show=show)
        self.plot_histograms(show=show)

    def plot_loss(self, show=True, legend=True):
        """Plot the loss values."""
        fig, ax = plt.subplots(figsize=(6,3))
        for agent, loss in self.logs['loss'].items():
            ax.plot(self.logs['steps'], loss, linewidth=4, alpha=.8, label=agent)
        ##ax.set_title('Categorical crossentropy loss')
        ax.set_xlabel('agent-environment interactions')
        ax.set_ylabel('policy loss')
        if legend:
            ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        plt.savefig(f'{self.savedir}/loss.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_accuracy(self, show=True, legend=True):
        """Plot the accuracy values."""
        fig, ax = plt.subplots(figsize=(6,3))
        for agent, acc in self.logs['accuracy'].items():
            ax.plot(self.logs['steps'], acc, linewidth=4, alpha=.8, label=agent)
        ##ax.set_title('Accuracy')
        ax.set_xlabel('agent-environment interactions')
        ax.set_ylabel('policy accuracy')
        if legend:
            ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        plt.savefig(f'{self.savedir}/accuracy.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_entropy(self, show=True, legend=True):
        """Plot the entropy values."""
        fig, ax = plt.subplots(figsize=(6,3))
        for agent, ent in self.logs['entropy'].items():
            ax.plot(self.logs['steps'], ent, linewidth=4, alpha=.8, label=agent)
        ##ax.set_title('Policy entropy')
        ax.set_xlabel('agent-environment interactions')
        ax.set_ylabel('policy entropy')
        if legend:
            ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        plt.savefig(f'{self.savedir}/entropy.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_reward(self, show=True, legend=True):
        """Plot the reward values."""
        fig, ax = plt.subplots(figsize=(6,3))
        for agent, ent in self.logs['reward'].items():
            ax.plot(self.logs['steps'], ent, linewidth=4, alpha=.8, label=agent)
        ##ax.set_title('Policy reward')
        ax.set_xlabel('agent-environment interactions')
        ax.set_ylabel('agent reward')
        if legend:
            ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        plt.savefig(f'{self.savedir}/reward.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_legend(self, show=True):
        """Plot the legend containing all agents."""
        fig, ax = plt.subplots(figsize=(.01,.01))
        for agent in self.logs['loss'].keys():
            ax.plot([0], [0], linewidth=4, label=agent)
        ax.axis('off')
        ax.legend(ncol=len(self.logs['loss']), loc='center', bbox_to_anchor=(.5, .5))
        plt.savefig(f'{self.savedir}/legend.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_histograms(self, sort=True, show=True):
        """Plot the action selection histograms for each agent."""
        sns.set_palette('muted')
        for agent, hist in self.logs['histogram'].items():
            fig, ax = plt.subplots(figsize=(6,3))
            df = pd.DataFrame(hist, index=self.logs['steps'])
            if sort:
                df.values[:,::-1].sort(axis=1)
            df.plot.bar(stacked=True, width=1, ax=ax, linewidth=.1, legend=None)
            plt.xticks(np.linspace(0, len(df)-1, 11), rotation=30, ha='right')
            ax.set_ylim(0, 1)
            ##ax.set_title(f'{agent}-agent histogram')
            ax.set_xlabel('agent-environment interactions')
            ax.set_ylabel('action histogram')
            plt.tight_layout()
            plt.savefig(f'{self.savedir}/hist_{agent}.png', dpi=300)
            plt.show() if show else plt.close()


if __name__ == '__main__':

    # parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', default='mnist',
                        help='Name of the logs file in "./logs/"')
    args = parser.parse_args()

    # set up experiment and train the agents
    viz = Visualization(args.logs)
    viz.plot(show=False, legend=False)

