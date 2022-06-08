import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim):
    file = open('../../data/simulation/sample_cost_{}D'.format(dim), 'rb')
    data = np.array(pickle.load(file))
    dataf = [df.seconds + df.microseconds / 1000000 for df in data.flatten()]
    return np.array(dataf).reshape(data.shape)


def draw(dim, ax):
    data = read(dim).sum(axis=0)
    group = [[1, 3], [1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 6]]
    data = np.array([data[:, gp].sum(axis=1) for gp in group])
    data = data[1:] / data[0]
    size_ests = [1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 70000, 100000]
    labels = ['DIS', 'REG', 'MLE']
    colors = ['g', 'r', 'b']
    for i, dat in enumerate(data):
        ax.plot(size_ests, dat, label=labels[i], color=colors[i])


def main():
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 3, figsize=[13.5, 3.5])
    ax = ax.flatten()
    for i, dim in enumerate([4, 6, 8]):
        draw(dim, ax[i])
        # ax[i].plot([1000, 100000], [0, 0], color='k')
        ax[i].set_title('$d={}$'.format(dim))
        ax[i].legend(loc=2)
        if dim == 4:
            ax[i].set_ylabel('$\mathrm{Time}_\mathrm{NIS}$')

        ax[i].set_ylim([0.5, 4.5])
        ax[i].set_xlabel('$n$')

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
