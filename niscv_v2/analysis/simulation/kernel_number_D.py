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
    settings = [[0, 0, False], [1, 0, False], [1, 1, False], [1, 1, True], [2, 0, False],
                [2, 0, True], [-1, 1, False], [-1, 1, True], [-1, 2, False], [-1, 2, True]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, 10, 1, 1])
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    colors = ['k', 'b', 'y', 'g', 'r', 'm']
    size_kns = [50, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600]

    data = read(dim)
    nVar = 5000 * np.var(data, axis=0)
    nMSE = 5000 * np.mean((data - truth) ** 2, axis=0)
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].loglog(size_kns, nMSE[i, :, j], c=colors[j], label='{}'.format(estimator))
            ax[i].loglog(size_kns, nVar[i, :, j], '.', c=colors[j])

        ax[i].set_title(setting)

    # groups = [[1, 2, 3], [4, 5], [6, 7]]
    # for group in groups:
    #     optimal = nMSE[group, :, :].min(axis=0).min(axis=1)
    #     for i in group:
    #         ax[i].loglog(size_kns, optimal, 'rx')


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(5, 2, figsize=[10, 15])
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
