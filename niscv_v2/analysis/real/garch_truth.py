import numpy as np
import pickle


def read(num):
    data = []
    for n in np.arange(1, num + 1):
        file = open('../../data/real/garch_truth_{}'.format(n), 'rb')
        data.append(pickle.load(file))

    return np.vstack(data)


def main():
    data = read(5)
    mean = np.mean(data, axis=0)
    print(mean)
    with open('../../data/real/truth', 'wb') as file:
        pickle.dump(mean, file)


if __name__ == '__main__':
    main()
