import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='tab20c', font='monospace', font_scale=1.)


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

    def plot(self, show=True):
        """Plot saved metrics."""
        sns.set_palette('muted')
        self.plot_loss(show=show)
        self.plot_accuracy(show=show)
        self.plot_entropy(show=show)
        self.plot_reward(show=show)
        self.plot_histograms(show=show)

    def plot_loss(self, show=True):
        """Plot the loss values."""
        fig, ax = plt.subplots(figsize=(8,5))
        for agent, loss in self.logs['loss'].items():
            ax.plot(self.logs['steps'], loss, linewidth=3, label=agent)
        ax.set_title('Categorical crossentropy loss')
        ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        ##plt.tight_layout()
        plt.savefig(f'{self.savedir}/loss.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_accuracy(self, show=True):
        """Plot the accuracy values."""
        fig, ax = plt.subplots(figsize=(8,5))
        for agent, acc in self.logs['accuracy'].items():
            ax.plot(self.logs['steps'], acc, linewidth=3, label=agent)
        ax.set_title('Accuracy')
        ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        ##plt.tight_layout()
        plt.savefig(f'{self.savedir}/accuracy.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_entropy(self, show=True):
        """Plot the entropy values."""
        fig, ax = plt.subplots(figsize=(8,5))
        for agent, ent in self.logs['entropy'].items():
            ax.plot(self.logs['steps'], ent, linewidth=3, label=agent)
        ax.set_title('Policy entropy')
        ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        ##plt.tight_layout()
        plt.savefig(f'{self.savedir}/entropy.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_reward(self, show=True):
        """Plot the reward values."""
        fig, ax = plt.subplots(figsize=(8,5))
        for agent, ent in self.logs['reward'].items():
            ax.plot(self.logs['steps'], ent, linewidth=3, label=agent)
        ax.set_title('Policy reward')
        ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(1., 1.))
        ##plt.tight_layout()
        plt.savefig(f'{self.savedir}/reward.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()

    def plot_histograms(self, sort=True, show=True):
        """Plot the action selection histograms for each agent."""
        sns.set_palette('muted')
        for agent, hist in self.logs['histogram'].items():
            fig, ax = plt.subplots(figsize=(8,5))
            df = pd.DataFrame(hist, index=self.logs['steps'])
            if sort:
                df.values[:,::-1].sort(axis=1)
            df.plot.bar(stacked=True, width=1, ax=ax, linewidth=.1, legend=None)
            plt.xticks(np.linspace(0, len(df)-1, 11), rotation=0)
            ax.set_ylim(0, 1)
            ax.set_title(f'{agent}-agent histogram')
            plt.tight_layout()
            plt.savefig(f'{self.savedir}/hist_{agent}.png', dpi=300)
            plt.show() if show else plt.close()


if __name__ == '__main__':

    logs = 'mnist'
    viz = Visualization(logs)
    viz.plot(show=False)

    ##for logs in os.listdir('./logs/'):
        ##print(logs)
        ##viz = Visualization(logs[:-4])
        ##viz.plot(show=False)

