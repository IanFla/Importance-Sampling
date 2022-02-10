import numpy as np
from niscv_v2.basics.garch import GARCH
from niscv_v2.basics.qtl import Qtl
import multiprocessing
import os
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


def run(it):
    np.random.seed(19971107 + it)
    Ds = [1, 2, 5]
    alphas = [0.05, 0.01]
    ratios = [500, 1000, 2000]
    result = []
    for i, D in enumerate(Ds):
        for alpha in alphas:
            print(it, D, alpha)
            result.append(experiment(D, alpha, size_est=4000000, show=False, size_kn=2000, ratio=ratios[i]))

    return result


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=30) as pool:
        begin = dt.now()
        its = np.arange(300)
        R = pool.map(run, its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/real/garch_truth', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
