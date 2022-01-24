import numpy as np
import pickle


def read():
    file = open('../../data/test/test_exp', 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    nVar = 10000 * np.var(data, axis=1)
    nMSE = 10000 * np.mean((data - 1.0) ** 2, axis=1)
    print(nVar)
    print(nMSE)


if __name__ == '__main__':
    main()
