import numpy as np
from niscv_v2.basics.garch import GARCH
from niscv_v2.basics.qtl import Qtl
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


class IP:
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs


def garch_model(D):
    garch = GARCH()
    garch.laplace(inflate=2, df=1)
    target = lambda x: garch.target(x[:, :3], x[:, 3:])
    statistic = lambda x: x[:, 3:].sum(axis=1)
    proposal = IP(pdf=lambda x: garch.proposal(x[:, :3], x[:, 3:]),
                  rvs=lambda size: np.hstack(garch.predict(D, size)))
    return target, statistic, proposal


def experiment(D, alpha, size_est, show, size_kn, ratio):
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, alpha, proposal, size_est=size_est, show=show)
    qtl.resampling(size_kn, ratio)
    qtl.density_estimation(mode=2, local=True, gamma=0.3, bdwth=1.2, alpha0=0.1)
    qtl.nonparametric_estimation(mode=2)
    return qtl.result[0]


def run(it, num):
    np.random.seed(1997 * num + 1107 + it)
    Ds = [1, 2, 5]
    alphas = [0.05, 0.01]
    ratios = [500, 1000, 2000]
    result = []
    for i, D in enumerate(Ds):
        for alpha in alphas:
            print(num, it, D, alpha)
            result.append(experiment(D, alpha, size_est=1000000, show=False, size_kn=2000, ratio=ratios[i]))

    return result


def main(num):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=30) as pool:
        begin = dt.now()
        its = np.arange(1, 31)
        R = pool.map(partial(run, num=num), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/real/garch_truth_{}'.format(num), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    for n in np.arange(1, 21):
        main(n)
