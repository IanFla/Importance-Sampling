from niscv_v2.experiments.garch_truth import garch_model
from niscv_v2.basics.qtl import Qtl
import numpy as np
from datetime import datetime as dt
import pickle


def experiment(D, alpha, size_est, show, size_kn, ratio):
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, alpha, proposal, size_est=size_est, show=show)

    ts = [dt.now()]
    qtl.initial_estimation()
    ts.append(dt.now())
    qtl.resampling(size_kn, ratio)
    ts.append(dt.now())
    qtl.density_estimation(mode=2, local=True, gamma=0.3, bdwth=1.5, alpha0=0.1)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=0)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=1)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=2)
    ts.append(dt.now())
    qtl.control_calculation()
    ts.append(dt.now())
    qtl.regression_estimation()
    ts.append(dt.now())
    qtl.likelihood_estimation()
    ts.append(dt.now())
    ts = np.array(ts)
    return ts[1:] - ts[:-1]


def run(it):
    Ds = [1, 2, 5]
    alphas = [0.05, 0.01]
    ratios = [500, 1000, 2000]
    results = []
    for i, D in enumerate(Ds):
        for alpha in alphas:
            print(it, D, alpha)
            result = experiment(D, alpha, size_est=400000, show=False, size_kn=2000, ratio=ratios[i])
            results.append(result)

    return results


def main():
    os.environ['OMP_NUM_THREADS'] = '30'
    R = []
    for it in range(5):
        R.append(run(it))

    with open('../../data/simulation/garch_cost', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
