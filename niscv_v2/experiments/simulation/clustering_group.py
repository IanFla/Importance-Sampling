import numpy as np
import scipy.stats as st
from niscv_v2.basics.exp import Exp
from niscv_v2.basics import utils
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(dim, fun, size_est, sn, show, size_kn, ratio, cluster):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, adjust=False, show=show)

    exp.initial_estimation()
    exp.resampling(size_kn, ratio, bootstrap='st')
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(mode=cluster, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    exp.nonparametric_estimation(mode=0)
    exp.nonparametric_estimation(mode=1)
    exp.nonparametric_estimation(mode=2)
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.control_calculation()
    exp.regression_estimation()
    if exp.show:
        exp.draw(grid_x, name='regression')

    exp.likelihood_estimation()
    return exp.result, exp.params


def run(it, dim):
    settings = [-1, -2, -3]
    clusters = [1, 0, 2, 3, 4, 5]
    Results = []
    Params = []
    for setting in settings:
        results = []
        params = []
        for cluster in clusters:
            np.random.seed(1997 + 1107 + it)
            print(dim, it, setting, cluster)
            res, par = experiment(dim=dim, fun=utils.integrand(setting), size_est=10000, sn=True,
                                  show=False, size_kn=300, ratio=1000, cluster=cluster)
            results.append(res)
            params.append(par)

        Results.append(results)
        Params.append(params)

    return [Results, Params]


def main(dim):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=60) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(partial(run, dim=dim), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../../data/simulation/clustering_group_{}D'.format(dim), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(4)
    main(6)
    main(8)
