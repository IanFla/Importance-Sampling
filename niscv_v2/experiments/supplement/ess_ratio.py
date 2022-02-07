import numpy as np
import scipy.stats as st
from niscv_v2.basics.exp import Exp
from niscv_v2.basics import utils
import pickle


def experiment(dim, fun, size_est, sn, show, size_kn, ratio):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, adjust=False, show=show)
    exp.initial_estimation()
    exp.resampling(size_kn, ratio, bootstrap='st')
    return 2 * (size_kn * ratio) / exp.params['ESS']


def run(dim):
    settings = [1, 2, 3, 4, -1, -2]
    ESS = []
    for setting in settings:
        ESS.append(experiment(dim=dim, fun=utils.integrand(setting), size_est=10000,
                              sn=True, show=False, size_kn=300, ratio=10000))

    return ESS


def main():
    R = []
    for dim in [4, 6]:
        R.append(run(dim))

    print(R)
    with open('../../data/simulation/ess_ratio', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
