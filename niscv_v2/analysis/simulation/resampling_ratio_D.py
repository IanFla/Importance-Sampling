import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim, bootstrap):
    file = open('../../data/simulation/resampling_ratio_{}D_{}'.format(dim, bootstrap), 'rb')
    data = pickle.load(file)
    data = [dat[0] for dat in data]
    return np.array(data)


def draw(dim, ax):
    settings = [1, 2, 3, 4, -1, -2]
    # estimators = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS']
    # colors = ['k', 'b', 'y', 'g', 'r']
    estimators = ['NIS', 'DNIS', 'DNIS$^*$', 'REG']
    colors = ['b', 'y', 'g', 'r']
    ratios = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    reference = np.array(pickle.load(open('../../data/simulation/ess_ratio', 'rb')))
    reference = reference[0] if dim == 4 else reference[1]

    datast = read(dim, 'st')
    datasp = read(dim, 'sp')
    nMSEst = 10000 * np.mean((datast - 1) ** 2, axis=0)
    nMSEst = nMSEst[:, :, 1:] / nMSEst[:, :, 0].reshape([6, 11, 1])
    nVarst = 10000 * np.var(datast, axis=0)
    nVarst = nVarst[:, :, 1:] / nVarst[:, :, 0].reshape([6, 11, 1])
    nMSEsp = 10000 * np.mean((datasp - 1) ** 2, axis=0)
    nMSEsp = nMSEsp[:, :, 1:] / nMSEsp[:, :, 0].reshape([6, 11, 1])
    nVarsp = 10000 * np.var(datasp, axis=0)
    nVarsp = nVarsp[:, :, 1:] / nVarsp[:, :, 0].reshape([6, 11, 1])
    for i, setting in enumerate(settings):
        for j, estimator in enumerate(estimators):
            ax[i].loglog(ratios, nMSEst[i, :, j], '-', c=colors[j], label=estimator)
            ax[i].loglog(ratios, nVarst[i, :, j], '.', c=colors[j])
            ax[i].loglog(ratios, nMSEsp[i, :, j], '--', c=colors[j])
            ax[i].loglog(ratios, nVarsp[i, :, j], 'x', c=colors[j])

        ax[i].set_ylim([0.8 * min(nMSEst[i, :, -1].min(), nMSEsp[i, :, -1].min()), 1.3])
        ax[i].plot([reference[i], reference[i]],
                   [0.8 * min(nMSEst[i, :, -1].min(), nMSEsp[i, :, -1].min()), 1.3], 'c-.', label='Ref')
        ax[i].set_title('$d$={}, $c$={}'.format(dim, setting))
        ax[i].grid(which='both')


def main(dim):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 2, figsize=[9, 8])
    ax = ax.flatten()
    draw(dim, ax)
    for a in ax:
        a.legend(loc=1)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(4)
    main(6)
