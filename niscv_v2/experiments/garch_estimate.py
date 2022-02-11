from niscv_v2.experiments.garch_truth import garch_model
from niscv_v2.basics.qtl import Qtl
import numpy as np
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(D, alpha, size_est, show, size_kn, ratio):
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, alpha, proposal, size_est=size_est, show=show)
    qtl.initial_estimation()
    qtl.resampling(size_kn, ratio)
    qtl.density_estimation(mode=2, local=True, gamma=0.3, bdwth=1.5, alpha0=0.1)
    qtl.nonparametric_estimation(mode=0)
    qtl.nonparametric_estimation(mode=1)
    qtl.nonparametric_estimation(mode=2)
    qtl.control_calculation()
    qtl.regression_estimation()
    qtl.likelihood_estimation()
    return qtl.result


def run(it, num):
    np.random.seed(1997 * num + 1107 + it)
    Ds = [1, 2, 5]
    alphas = [0.05, 0.01]
    ratios = [500, 1000, 2000]
    result = []
    for i, D in enumerate(Ds):
        for alpha in alphas:
            print(num, it, D, alpha)
            result.append(experiment(D, alpha, size_est=400000, show=False, size_kn=2000, ratio=ratios[i]))

    return result


def main(num):
    os.environ['OMP_NUM_THREADS'] = '3'
    with multiprocessing.Pool(processes=10) as pool:
        begin = dt.now()
        its = np.arange(1, 11)
        R = pool.map(partial(run, num=num), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/real/garch_estimate_{}'.format(num), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    for n in np.arange(25, 31):
        main(n)
