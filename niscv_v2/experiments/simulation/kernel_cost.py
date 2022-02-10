import numpy as np
import scipy.stats as st
from niscv_v2.basics.exp import Exp
from niscv_v2.basics import utils
from datetime import datetime as dt
import pickle


def experiment(dim, fun, size_est, sn, size_kn, ratio):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, adjust=False, show=False)

    ts = [dt.now()]
    exp.initial_estimation()
    ts.append(dt.now())
    exp.resampling(size_kn, ratio, bootstrap='st')
    ts.append(dt.now())
    exp.density_estimation(mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    ts.append(dt.now())
    exp.nonparametric_estimation(mode=2)
    ts.append(dt.now())
    exp.control_calculation()
    ts.append(dt.now())
    exp.regression_estimation()
    ts.append(dt.now())
    exp.likelihood_estimation()
    ts.append(dt.now())
    ts = np.array(ts)
    return ts[1:] - ts[:-1]


def run(it, dim):
    size_kns = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results = []
    for size_kn in size_kns:
        np.random.seed(19971107 + it)
        print(dim, it,  size_kn)
        result1 = experiment(dim=dim, fun=utils.integrand(1), size_est=4000, sn=False, size_kn=size_kn, ratio=1000)
        result2 = experiment(dim=dim, fun=utils.integrand(1), size_est=4000, sn=True, size_kn=size_kn, ratio=1000)
        results.append(result1 + result2)

    return results


def main(dim):
    R = []
    for it in range(10):
        R.append(run(it, dim))

    with open('../../data/simulation/kernel_cost_{}D'.format(dim), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(4)
    main(6)
    main(8)
