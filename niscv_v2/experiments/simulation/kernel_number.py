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
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, show=show)

    exp.initial_estimation()
    exp.resampling(size_kn, ratio, resample=True, bootstrap=bootstrap)
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    exp.nonparametric_estimation(mode=0)
    exp.nonparametric_estimation(mode=1)
    exp.nonparametric_estimation(mode=2)
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.control_calculation()
    exp.regression_estimation(mode=0)
    exp.regression_estimation(mode=1)
    if exp.show:
        exp.draw(grid_x, name='regression')

    exp.likelihood_setup(lim=20, sep=5)
    exp.likelihood_estimation(mode=1)
    exp.likelihood_estimation(mode=0)
    return exp.result, exp.params


def run(it, dim, bootstrap):
    settings = [[0, 0, False], [1, 1, False], [1, 0, False], [1, 0, True],
                [2, 0, False], [2, 0, True], [-1, 1, False], [-1, 1, True]]
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400]
    Results = []
    Params = []
    for setting in settings:
        results = []
        params = []
        for size_kn in size_kns:
            np.random.seed(1997 * it + 1107)
            print(it, setting, size_kn)
            res, par = experiment(dim=dim, fun=utils.integrand(setting[0], setting[1]), size_est=2000,
                                  sn=setting[2], show=False, size_kn=size_kn, ratio=50, bootstrap=bootstrap)
            results.append(res)
            params.append(par)

        Results.append(results)
        Params.append(params)

    return [Results, Params]


def main(dim, bootstrap):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=60) as pool:
        begin = dt.now()
        its = np.arange(200)
        R = pool.map(partial(run, dim=dim, bootstrap=bootstrap), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../../data/simulation/kernel_number_{}D_{}BS'.format(dim, bootstrap), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(2, True)
    main(2, False)
    main(4, True)
    main(4, False)
    main(6, True)
    main(6, False)
