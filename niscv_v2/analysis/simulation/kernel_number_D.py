import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim):
    file = open('../../data/simulation/kernel_number_{}D'.format(dim), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [[1, False], [1, True],
                [2, False], [2, True],
                [3, False], [3, True],
                [4, False], [4, True],
                [-1, False], [-1, True],
                [-2, False], [-2, True]]
    estimators = ['NIS', 'DNIS', 'DIS', 'REG', 'MLE']
    colors = ['y', 'y', 'g', 'r', 'b']
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    data = read(dim)
    nMSE = 4000 * np.mean((data - 1) ** 2, axis=0)
    nVar = 4000 * np.var(data, axis=0)
    nVar = nVar[:, :, 1:] / nMSE[:, :, 0].reshape([12, 10, 1])
    nMSE = nMSE[:, :, 1:] / nMSE[:, :, 0].reshape([12, 10, 1])
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            if estimator != 'DNIS':
                ax[i].loglog(size_kns, nMSE[i, :, j], c=colors[j], label=estimator)
                ax[i].loglog(size_kns, nVar[i, :, j], '.', c=colors[j])

        # ax[i].set_xlabel('log(kernel number)')
        # ax[i].set_ylabel('nMSE/nVar')
        ax[i].set_title('$d$={}, $c$={}, sn={}'.format(dim, setting[0], setting[1]))
        ax[i].grid(which='both')
        if (setting[0] in {1, 3, -1}) and (not setting[1]):
            ax[i].set_ylabel('$\mathrm{MSE}_\mathrm{IIS}$ or $\mathrm{Var}_\mathrm{IIS}$')

        if setting[0] < 0:
            ax[i].set_xlabel('$m_0$')

    groups = np.arange(ax.size).reshape([-1, 2])
    for group in groups:
        optimal = nMSE[group, :, :].min(axis=2).min(axis=0)
        for i in group:
            ax[i].loglog(size_kns, optimal, 'c-.', label='Opt')
            ax[i].set_ylim([0.8 * optimal.min(), 1.3])


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 4, figsize=[18, 8])
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
