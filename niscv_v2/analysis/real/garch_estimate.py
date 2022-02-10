import numpy as np
import pickle


def read(num):
    data = []
    for n in np.arange(1, num + 1):
        file = open('../../data/real/garch_estimate_{}'.format(n), 'rb')
        data.append(pickle.load(file))

    return np.vstack(data)


def main():
    file = open('../../data/real/truth', 'rb')
    truth = np.array(pickle.load(file)).reshape([1, 6, 1])
    data = read(7)
    mean = np.mean(data, axis=0)
    nMSE = 400000 * np.mean((data - truth) ** 2, axis=0)
    nVar = 400000 * np.var(data, axis=0)
    print(mean)
    print(nMSE[:, 1:] / nMSE[:, 0].reshape([-1, 1]))
    print(nVar[:, 1:] / nVar[:, 0].reshape([-1, 1]))


if __name__ == '__main__':
    main()
