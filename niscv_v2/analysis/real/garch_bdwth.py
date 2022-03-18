import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../../data/real/garch_bdwth2', 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    Ds = [1, 2, 5]
    bdwths = [0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6]
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 3, figsize=[13.5, 3.5])
    ax = ax.flatten()
    for i, a in enumerate(ax):
        a.semilogy(bdwths, data[:, 2 * i], label=r'$\alpha=0.05$')
        a.semilogy(bdwths, data[:, 2 * i + 1], label=r'$\alpha=0.01$')
        a.set_title('$D={}$'.format(Ds[i]))
        a.legend(loc=1)
        if i == 0:
            a.set_ylabel(r'Estimate of $\mathrm{min}_{\mathbf{\beta}}\sigma_{D,\alpha}^2(\mathbf{\beta})$')

        a.set_xlabel('$h$')

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
