import numpy as np
from niscv_v2.basics import utils
from niscv_v2.experiments.simulation.kernel_cost import experiment
import pickle


def run(it, dim):
    size_ests = [1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 70000, 100000]
    results = []
    for size_est in size_ests:
        np.random.seed(19971107 + it)
        print(dim, it,  size_est)
        result1 = experiment(dim=dim, fun=utils.integrand(1), size_est=size_est, sn=False, size_kn=300, ratio=1000)
        result2 = experiment(dim=dim, fun=utils.integrand(1), size_est=size_est, sn=True, size_kn=300, ratio=1000)
        results.append(result1 + result2)

    return results


def main(dim):
    R = []
    for it in range(10):
        R.append(run(it, dim))

    with open('../../data/simulation/sample_cost_{}D'.format(dim), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(4)
    main(6)
    main(8)
