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
    data = read(6)
    mean = np.mean(data, axis=0)
    nVar = 400000 * np.var(data, axis=0)
    print(mean)
    print(nVar[:, 1:] / nVar[:, 0].reshape([-1, 1]))


if __name__ == '__main__':
    main()
