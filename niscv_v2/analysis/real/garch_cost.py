import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../../data/real/garch_cost', 'rb')
    data = np.array(pickle.load(file))
    dataf = [df.seconds + df.microseconds / 1000000 for df in data.flatten()]
    return np.array(dataf).reshape(data.shape)


def main():
    data = read().sum(axis=0)
    print(data)
    group = [[0], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6, 7], [1, 2, 6, 8]]
    data = np.array([data[:, gp].sum(axis=1) for gp in group])
    labels = ['IIS', 'NIS', 'MIS$^*$', 'MIS', 'RIS', 'MLE']
    scenarios = ['(1, 0.05)', '(1, 0.01)', '(2, 0.05)', '(2, 0.01)', '(5, 0.05)', '(5, 0.01)']
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=[5, 3])
    for i, dat in enumerate(data):
        ax.plot(scenarios, dat, label=labels[i])

    ax.legend(loc=2)
    fig.tight_layout()
    fig.show()
    with open('../../data/real/time', 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    main()
