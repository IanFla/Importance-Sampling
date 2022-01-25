import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../../data/test/data_exp_zeta', 'rb')
    data = pickle.load(file)
    data = [[da[0] for da in dat] for dat in data]
    return np.array(data)


def main():
    data = read()
    nVar = 40000 * np.var(data, axis=0)
    nMSE = 40000 * np.mean((data - 1.0) ** 2, axis=0)
    settings = ['IS-bt', 'SIS-bt', 'IS-sp', 'SIS-sp']
    colors = ['y', 'c', 'r', 'm']
    estimators = ['IIS', 'NIS', 'MIS*', 'MIS', 'RIS*', 'RIS',
                  'MLE*$_{0}$', 'MLE*$_{5}$', 'MLE*$_{10}$', 'MLE*$_{15}$', 'MLE*$_{20}$',
                  'MLE$_{0}$', 'MLE$_{5}$', 'MLE$_{10}$', 'MLE$_{15}$', 'MLE$_{20}$']
    fig, ax = plt.subplots(figsize=[8, 6])
    for i, setting in enumerate(settings):
        ax.semilogy(estimators, nMSE[i], colors[i]+'-', label='nMSE({})'.format(setting))
        ax.semilogy(estimators, nVar[i], colors[i] + '--', label='nVar({})'.format(setting))

    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
