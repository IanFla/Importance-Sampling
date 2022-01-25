import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../../data/test/data_exp', 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    nVar = 10000 * np.var(data, axis=0)
    nMSE = 10000 * np.mean((data - 1.0) ** 2, axis=0)
    settings = ['usn-bt', 'sn-bt', 'usn-sp', 'sn-sp']
    colors = ['g', 'c', 'r', 'm']
    estimators = ['IIS', 'NIS', 'MIS*', 'MIS', 'RIS*', 'RIS',
                  'MLE$_{0}$', 'MLE$_{5}$', 'MLE$_{10}$', 'MLE$_{15}$', 'MLE$_{20}$']
    fig, ax = plt.subplots(figsize=[10, 7])
    for i, setting in enumerate(settings):
        ax.semilogy(estimators, nMSE[i], colors[i]+'-', label='nMSE({})'.format(setting))
        ax.semilogy(estimators, nVar[i], colors[i] + '--', label='nVar({})'.format(setting))

    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
