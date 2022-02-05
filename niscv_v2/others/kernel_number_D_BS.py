import numpy as np
from matplotlib import pyplot as plt
import pickle
from niscv_v2.basics import utils


def read(dim, bootstrap):
    file = open('../../data/simulation/kernel_number_{}D_{}BS'.format(dim, bootstrap), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, bootstrap, ax):
    settings = [[0, 0, False], [1, 1, False], [1, 0, False], [1, 0, True],
                [2, 0, False], [2, 0, True], [-1, 1, False], [-1, 1, True]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, 8, 1, 1])
    data = read(dim, bootstrap)
    nVar = 2000 * np.var(data, axis=0)
    nMSE = 2000 * np.mean((data - truth) ** 2, axis=0)
    settings = ['m0c0F', 'm1c1F', 'm1c0F', 'm1c0T', 'm2c0F', 'm2c0T', 'm-1c1F', 'm-1c1T']
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS$^*$', 'RIS',
                  'MLE$^*_{0}$', 'MLE$^*_{5}$', 'MLE$^*_{10}$', 'MLE$^*_{15}$', 'MLE$^*_{20}$',
                  'MLE$_{0}$', 'MLE$_{5}$', 'MLE$_{10}$', 'MLE$_{15}$', 'MLE$_{20}$']
    ests = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS$^*$', 'RIS', 'MLE$^*_{20}$', 'MLE$_{20}$']
    index = [estimators.index(est) for est in ests]
    colors = ['w', 'y', 'g', 'c', 'b', 'r', 'm', 'k']
    line1 = '-' if bootstrap else '--'
    line2 = 'o' if bootstrap else 'x'
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400]
    for i, setting in enumerate(settings):
        for j, est in enumerate(ests):
            ax[i].loglog(size_kns, nMSE[i, :, index[j]], colors[j] + line1, label='{}'
                         .format(est) if bootstrap else None)
            ax[i].loglog(size_kns, nVar[i, :, index[j]], colors[j] + line2)

        ax[i].set_title(setting)


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(4, 2, figsize=[10, 15])
    ax = ax.flatten()
    draw(dim, True, ax)
    draw(dim, False, ax)
    for a in ax:
        a.legend(loc=2)
        a.set_ylim([1e-2, 1e2])

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(5)
