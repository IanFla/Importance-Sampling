import numpy as np
from matplotlib import pyplot as plt
import pickle
from niscv_v2.basics import utils


def read(dim):
    file = open('../../data/simulation/resampling_ratio_{}D'.format(dim), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [[1, 1, False], [1, 1, True],
                [2, 1, False], [2, 1, True],
                [3, 1, False], [3, 1, True],
                [4, 1, False], [4, 1, True],
                [-1, 1, False], [-1, 1, True],
                [-1, 2, False], [-1, 2, True]]
    truth = np.array([utils.truth(setting[0], setting[1]) for setting in settings]).reshape([1, 12, 1, 1])
    # estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    # colors = ['k', 'b', 'y', 'g', 'r', 'm']
    estimators = ['IIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    colors = ['k', 'y', 'g', 'r', 'm']
    ratios = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028]

    data = read(dim)
    result = data[:, :, :, :6]
    result = result[:, :, :, [0, 2, 3, 4, 5]]
    nVar = 10000 * np.var(result, axis=0)
    nMSE = 10000 * np.mean((result - truth) ** 2, axis=0)
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].loglog(ratios, nMSE[i, :10, j], c=colors[j], label=estimator)
            ax[i].loglog(ratios, nVar[i, :10, j], '.', c=colors[j])
            ax[i].loglog(ratios, nMSE[i, 10:, j], '--', c=colors[j])
            ax[i].loglog(ratios, nVar[i, 10:, j], 'x', c=colors[j])

        ax[i].set_title(setting)


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 4, figsize=[20, 10])
    ax = ax.flatten()
    draw(dim, ax)
    for a in ax:
        a.legend(loc=2)

    fig.tight_layout()
    fig.show()

    data = read(dim)
    param = data[:, :, :, 6:]
    ratios = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028]
    rate1 = 2 * (500 * np.append(ratios, ratios)) / param[:, :, :, 0].mean(axis=0)
    rate2 = param[:, :, :, 1].mean(axis=0) / 500
    fig, ax = plt.subplots(3, 4, figsize=[20, 10])
    ax = ax.flatten()
    for i, a in enumerate(ax):
        a.plot(ratios, rate1[i, :10], label='mt')
        a.plot(ratios, rate1[i, 10:], label='st')
        a.legend(loc=2)

    fig.tight_layout()
    fig.show()
    # fig, ax = plt.subplots(3, 4, figsize=[20, 10])
    # ax = ax.flatten()
    # for i, a in enumerate(ax):
    #     a.plot(ratios, rate2[i, :10], label='mt')
    #     a.plot(ratios, rate2[i, 10:], label='st')
    #     a.legend(loc=2)
    #
    # fig.tight_layout()
    # fig.show()


if __name__ == '__main__':
    main(4)
    main(6)
    main(8)
