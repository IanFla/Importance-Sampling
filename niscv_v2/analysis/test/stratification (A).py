import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../../data/test/stratification0', 'rb')
    data = pickle.load(file)
    return np.array(data)


def main(i):
    data = read()
    var = np.var(data, axis=0)[i]
    mse = np.mean((data - 0.0) ** 2, axis=0)[i]
    methods = ['std', 'str', 'pst']
    colors = ['b', 'r', 'g']
    fig, ax = plt.subplots(figsize=[8, 6])
    for i, method in enumerate(methods):
        ax.loglog([1, 4, 16, 32, 64, 128, 256], mse[:, i], colors[i] + '-', label='MSE({})'.format(method))
        ax.loglog([1, 4, 16, 32, 64, 128, 256], var[:, i], colors[i] + '--', label='Var({})'.format(method))

    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(0)
    main(1)
    main(2)
    main(3)
