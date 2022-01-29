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
    settings = [[0, 0, False], [1, 1, False], [1, 0, False], [1, 0, True],
                [2, 0, False], [2, 0, True], [-1, 1, False], [-1, 1, True]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, 8, 1, 1])
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS$^*$', 'RIS',
                  'MLE$^*_{0}$', 'MLE$^*_{5}$', 'MLE$^*_{10}$', 'MLE$^*_{15}$', 'MLE$^*_{20}$',
                  'MLE$_{0}$', 'MLE$_{5}$', 'MLE$_{10}$', 'MLE$_{15}$', 'MLE$_{20}$']
    ests = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS$^*$', 'RIS', 'MLE$^*_{20}$', 'MLE$_{20}$']
    index = [estimators.index(est) for est in ests]
    colors = ['k', 'g', 'c', 'b', 'y', 'orange', 'r', 'm']
    # size_kns = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
    size_kns = [50, 100, 150, 200, 250, 300, 400, 500, 600]

    data = read(dim)
    nVar = 5000 * np.var(data, axis=0)
    nMSE = 5000 * np.mean((data - truth) ** 2, axis=0)
    for i, setting in enumerate(settings):
        for j, est in enumerate(ests):
            ax[i].loglog(size_kns, nMSE[i, :len(size_kns), index[j]], c=colors[j], label='{}'.format(est))
            ax[i].loglog(size_kns, nVar[i, :len(size_kns), index[j]], '.', c=colors[j])

        ax[i].set_title(setting)


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(4, 2, figsize=[10, 15])
    ax = ax.flatten()
    draw(dim, ax)
    for a in ax:
        a.legend(loc=2)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(3)
    main(6)
    main(9)
