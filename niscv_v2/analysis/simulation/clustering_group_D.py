import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st


def read(dim):
    file = open('../../data/simulation/clustering_group_{}D'.format(dim), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [-1, -2, -3]
    # estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS']
    # colors = ['k', 'b', 'y', 'g', 'r', 'm']
    estimators = ['NIS', 'DNIS', 'DNIS$^*$', 'REG']
    colors = ['b', 'y', 'g', 'r']
    clusters = ['none', 'ind', 'km(2)', 'km(3)', 'km(4)', 'km(5)']
    refer1 = lambda x: ((1 - st.norm.cdf(x)) / st.norm.cdf(x)) * np.ones(6)
    refer2 = lambda x: 4 * ((1 - st.norm.cdf(x)) ** 2) * np.ones(6)

    data = read(dim)
    nMSE = 10000 * np.mean((data - 1) ** 2, axis=0)
    nMSE0 = nMSE[:, :, 0]
    nMSE = nMSE[:, :, 1:] / nMSE[:, :, 0].reshape([3, 6, 1])
    nVar = 10000 * np.var(data, axis=0)
    nVar = nVar[:, :, 1:] / nVar[:, :, 0].reshape([3, 6, 1])
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].semilogy(clusters, nMSE[i, :, j], '-', c=colors[j], label=estimator)
            ax[i].semilogy(clusters, nVar[i, :, j], '.', c=colors[j])

        ax[i].plot(clusters, refer1(setting) / nMSE0[i, :], 'c-.', label='Ref 1')
        ax[i].plot(clusters, refer2(setting) / nMSE0[i, :], 'm-.', label='Ref 2')
        ax[i].set_ylim([6e-3, 1.0])
        ax[i].set_title('$d$={}, $c$={}'.format(dim, setting))
        ax[i].grid(axis='x', which='major')
        ax[i].grid(axis='both', which='both')


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 3, figsize=[13.5, 3.5])
    ax = ax.flatten()
    draw(dim, ax)
    for a in ax:
        a.legend(loc=1)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(4)
    main(6)
