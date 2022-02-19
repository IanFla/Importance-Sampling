import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim):
    file = open('../../data/simulation/kernel_cost_{}D'.format(dim), 'rb')
    data = np.array(pickle.load(file))
    dataf = [df.seconds + df.microseconds / 1000000 for df in data.flatten()]
    return np.array(dataf).reshape(data.shape)


def draw(dim, ax):
    data = read(dim).sum(axis=0)
    group = [[1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 6]]
    data = np.array([data[:, gp].sum(axis=1) for gp in group])
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    labels = ['NIS/DNIS/DNIS$^*$', 'REG', 'MLE']
    colors = ['b', 'r', 'm']
    for i, dat in enumerate(data):
        ax.plot(size_kns, dat, label=labels[i], color=colors[i])


def main():
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 3, figsize=[13.5, 3.5])
    ax = ax.flatten()
    for i, dim in enumerate([4, 6, 8]):
        draw(dim, ax[i])
        ax[i].plot([50, 500], [0, 0], color='k')
        ax[i].set_title('$d={}$'.format(dim))
        ax[i].legend(loc=2)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
