import numpy as np
import scipy.stats as st
from niscv_v2.basics.exp import Exp
from niscv_v2.basics import utils
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(dim, fun, size_est, sn, show, size_kn, ratio, bootstrap):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, adjust=False, show=show)

    exp.initial_estimation()
    exp.resampling(size_kn, ratio, bootstrap=bootstrap)
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    exp.nonparametric_estimation(mode=0)
    exp.nonparametric_estimation(mode=1)
    exp.nonparametric_estimation(mode=2)
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.control_calculation()
    exp.regression_estimation()
    if exp.show:
        exp.draw(grid_x, name='regression')

    return exp.result, exp.params


def run(it, dim, bootstrap):
    settings = [1, 2, 3, 4, -1, -2]
    ratios = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1028]
    Results = []
    Params = []
    for setting in settings:
        results = []
        params = []
        for ratio in ratios:
            np.random.seed(19971107 + it)
            print(dim, bootstrap, it, setting, ratio)
            res, par = experiment(dim=dim, fun=utils.integrand(setting), size_est=10000, sn=False,
                                  show=False, size_kn=300, ratio=ratio, bootstrap=bootstrap)
            results.append(res)
            params.append(par)

        Results.append(results)
        Params.append(params)

    return [Results, Params]


def main(dim, bootstrap):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=60) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(partial(run, dim=dim, bootstrap=bootstrap), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../../data/simulation/resampling_ratio_{}D_{}'.format(dim, bootstrap), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(4, 'sp')
    main(6, 'sp')
    main(4, 'st')
    main(6, 'st')
