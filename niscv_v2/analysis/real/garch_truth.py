import numpy as np
import pickle


def read(num):
    data = []
    for n in np.arange(1, num + 1):
        file = open('../../data/real/garch_truth_{}'.format(n), 'rb')
        data.append(pickle.load(file))

    return np.vstack(data)


def main():
    data = read(1)
    print(data.shape)
    mean = np.mean(data, axis=0)
    print(mean)


if __name__ == '__main__':
    main()
