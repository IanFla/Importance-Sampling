import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim, bootstrap):
    file = open('../../data/simulation/resampling_ratio_{}D_{}'.format(dim, bootstrap), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, bootstrap, ax):
    settings = [[1, False], [1, True],
                [2, False], [2, True],
                [3, False], [3, True],
                [4, False], [4, True],
                [-1, False], [-1, True],
                [-2, False], [-2, True]]
    estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS']
    colors = ['k', 'b', 'y', 'g', 'r']
    ratios = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1028]

    data = read(dim, bootstrap)
    nMSE = 10000 * np.mean((data - 1) ** 2, axis=0)
    nVar = 10000 * np.var(data, axis=0)
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].loglog(ratios, nMSE[i, :, j], c=colors[j], label=estimator)
            ax[i].loglog(ratios, nVar[i, :, j], '.', c=colors[j])

        ax[i].set_ylim([0.8 * nMSE[i, :, -1].min(), 1.3 * nMSE[i, :, 0].max()])
        ax[i].set_title('$d$={}, $c$={}, sn={}'.format(dim, setting[0], setting[1]))


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 4, figsize=[18, 8])
    ax = ax.flatten()
    draw(dim, 'st', ax)
    for a in ax:
        a.legend(loc=1)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(4)
    main(6)
    # main(8)
