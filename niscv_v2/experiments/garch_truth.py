import numpy as np
from niscv_v2.basics.garch import GARCH
from niscv_v2.basics.qtl import Qtl
import scipy.stats as st

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


def random_walk(target, x0, cov, factor, burn, size, thin):
    walk = lambda x: st.multivariate_normal(mean=x, cov=(factor ** 2) * cov).rvs()
    for b in range(burn):
        x1 = walk(x0)
        if (target(x1) / target(x0)) >= st.uniform.rvs():
            x0 = np.copy(x1)

    xs = []
    for s in range(size):
        print(s)
        for t in range(thin):
            x1 = walk(x0)
            if (target(x1) / target(x0)) >= st.uniform.rvs():
                x0 = np.copy(x1)

        xs.append(x0)

    return np.array(xs)


def experiment(it, D, size):
    print('it:', it, D)
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, None, proposal, size_est=None, show=False)
    samples = qtl.ini_rvs(100000)
    weights = target(samples) / (qtl.ini_pdf(samples) + 1.0 * (qtl.ini_pdf(samples) == 0))
    mean = np.sum(weights * samples.T, axis=1) / np.sum(weights)
    cov = np.cov(samples.T, aweights=weights)
    target2 = lambda x: target(x.reshape([1, -1]))[0]
    samples2 = random_walk(target=target2, x0=mean, cov=cov, factor=1.7 / np.sqrt(D + 3),
                           burn=1000, size=size, thin=10)
    statistics = statistic(samples2)
    return statistics


def run(D):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=30) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(partial(experiment, D=D, size=1000000), its)
        end = dt.now()
        print((end - begin).seconds)

    return R


def main():
    results = []
    for D in [1, 2, 5]:
        R = np.array(run(D)).flatten()
        result = [np.quantile(R, 0.05), np.quantile(R, 0.01)]
        print(result)
        results.append(result)

    with open('../data/real/truth', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
