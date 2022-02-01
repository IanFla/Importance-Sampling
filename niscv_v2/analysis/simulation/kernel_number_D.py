import numpy as np
from matplotlib import pyplot as plt
import pickle
from niscv_v2.basics import utils


def read(dim):
    file = open('../../data/simulation/kernel_number_adjust_{}D'.format(dim), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [[1, 1, False, False], [1, 1, False, True], [1, 1, True, False],
                [2, 1, False, False], [2, 1, False, True], [2, 1, True, False],
                [-1, 1, False, False], [-1, 1, False, True], [-1, 1, True, False]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, 9, 1, 1])
    # estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'NIS$^*$', 'MIS', 'RIS*', 'MLE*', 'RIS', 'MLE']
    # colors = ['k', 'b', 'y', 'g', 'r', 'm']
    colors = ['k', 'b', 'y', 'orange', 'g', 'lime', 'violet', 'r', 'm']
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
    fig, ax = plt.subplots(3, 3, figsize=[20, 15])
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
