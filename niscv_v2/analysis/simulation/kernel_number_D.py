import numpy as np
from matplotlib import pyplot as plt
import pickle
from niscv_v2.basics import utils


def read(dim):
    file = open('../../data/simulation/kernel_number_{}D'.format(dim), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [[1, 1, False, False], [1, 1, False, True], [1, 1, True, False],
                [2, 1, False, False], [2, 1, False, True], [2, 1, True, False],
                [3, 1, False, False], [3, 1, False, True], [3, 1, True, False],
                [4, 1, False, False], [4, 1, False, True], [4, 1, True, False],
                [-1, 1, False, False], [-1, 1, False, True], [-1, 1, True, False],
                [-1, 2, False, False], [-1, 2, False, True], [-1, 2, True, False]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, ax.size, 1, 1])
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    colors = ['k', 'b', 'y', 'g', 'r', 'm']
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    data = read(dim)
    nVar = 4000 * np.var(data, axis=0)
    nMSE = 4000 * np.mean((data - truth) ** 2, axis=0)
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].loglog(size_kns, nMSE[i, :, j], c=colors[j], label='{}'.format(estimator))
            ax[i].loglog(size_kns, nVar[i, :, j], '.', c=colors[j])

        ax[i].set_title(setting)

    groups = np.arange(ax.size).reshape([-1, 3])
    for group in groups:
        optimal = nMSE[group, :, :].min(axis=2).min(axis=0)
        for i in group:
            ax[i].loglog(size_kns, optimal, 'cx', label='OPT')


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(6, 3, figsize=[15, 20])
    ax = ax.flatten()
    draw(dim, ax)
    for a in ax:
        a.legend(loc=2)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(4)
    main(6)
    main(8)
