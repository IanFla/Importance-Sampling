from niscv_v2.experiments.garch_truth import garch_model
from niscv_v2.basics.qtl import Qtl
import multiprocessing
import os
from datetime import datetime as dt
import pickle


def experiment(D, alpha, size_est, show, size_kn, ratio, bdwth):
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, alpha, proposal, size_est=size_est, show=show)
    qtl.resampling(size_kn, ratio)
    qtl.density_estimation(mode=2, local=True, gamma=0.3, bdwth=bdwth, alpha0=0.1)
    qtl.nonparametric_estimation(mode=2)
    qtl.control_calculation()
    qtl.regression_estimation()
    qtl.asymptotic_variance()
    return qtl.params['aVar']


def run(bdwth):
    Ds = [1, 2, 5]
    alphas = [0.05, 0.01]
    ratios = [500, 1000, 2000]
    result = []
    for i, D in enumerate(Ds):
        for alpha in alphas:
            print(bdwth, D, alpha)
            result.append(experiment(D=D, alpha=alpha, size_est=100000, show=False,
                                     size_kn=2000, ratio=ratios[i], bdwth=bdwth))

    return result


def main():
    os.environ['OMP_NUM_THREADS'] = '2'
    with multiprocessing.Pool(processes=16) as pool:
        begin = dt.now()
        bdwths = [0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6]
        R = pool.map(run, bdwths)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/real/garch_bdwth', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
