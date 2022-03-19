import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(num):
    data = []
    for n in np.arange(1, num + 1):
        file = open('../../data/real/garch_estimate_{}'.format(n), 'rb')
        data.append(pickle.load(file))

    return np.vstack(data)


def main():
    plt.style.use('ggplot')
    file = open('../../data/real/truth', 'rb')
    truth = np.array(pickle.load(file)).reshape([1, 6, 1])

    file = open('../../data/real/time', 'rb')
    time = pickle.load(file)

    data = read(30)
    mean = np.mean(data, axis=0)
    # print(mean)
    # estimators = ['NIS', 'DNIS', 'DNIS$^*$', 'REG', 'MLE']
    estimators = ['NIS', 'DNIS---', 'DNIS', 'REG', 'MLE']
    colors = ['b', 'y', 'g', 'r', 'm']
    scenarios = ['(1, 0.05)', '(1, 0.01)', '(2, 0.05)', '(2, 0.01)', '(5, 0.05)', '(5, 0.01)']
    nMSE = 400000 * np.mean((data - truth) ** 2, axis=0)
    print(np.round(nMSE, 4))
    nMSE_time = nMSE * time.T
    print(np.round(nMSE_time), 4)
    nVar = 400000 * np.var(data, axis=0)
    nMSE_time = nMSE_time[:, 1:] / nMSE_time[:, 0].reshape([-1, 1])
    nVar = nVar[:, 1:] / nMSE[:, 0].reshape([-1, 1])
    nMSE = nMSE[:, 1:] / nMSE[:, 0].reshape([-1, 1])
    fig, ax = plt.subplots(1, 2, figsize=[10, 3])
    for i, est in enumerate(estimators):
        if i == 1:
            continue

        ax[0].semilogy(scenarios, nMSE[:, i], c=colors[i], label=est)
        # ax[0].semilogy(scenarios, nVar[:, i], '.', c=colors[i])
        ax[1].semilogy(scenarios, nMSE_time[:, i], c=colors[i], label=est)

    ax[0].set_xlabel(r'$(D,\alpha)$')
    ax[1].set_xlabel(r'$(D,\alpha)$')
    ax[0].set_ylabel('Error')
    ax[1].set_ylabel(r'$\mathrm{Error}\times\mathrm{Time}$')
    for a in ax:
        a.legend(loc=2)
        a.grid(axis='x', which='major')
        a.grid(axis='both', which='both')

    fig.tight_layout()
    fig.show()
    # print(nMSE[:, 1:] / nMSE[:, :-1])


if __name__ == '__main__':
    main()
