import numpy as np
from niscv_v2.basics.kde import KDE
from scipy.linalg import sqrtm

from matplotlib import pyplot as plt
import scipy.stats as st


def leverage(target, proposal, size_kn, alpha0, size_est):
    centers = target(size_kn)
    kde = KDE(centers, np.ones(centers.shape[0]), local=False, gamma=1.0, bdwth=1.0)
    mix_pdf = lambda x: alpha0 * proposal.pdf(x) + (1 - alpha0) * kde.pdf(x)
    mix_rvs = lambda size: np.vstack([proposal.rvs(round(alpha0 * size)),
                                      kde.rvs(size - round(alpha0 * size), stratify=True)])
    samples = mix_rvs(size_est)
    controls = kde.kns(samples) / mix_pdf(samples) - 1
    controls_ = controls.T.dot(sqrtm(np.linalg.inv(np.cov(controls))))

    leverages = (controls_ ** 2).sum(axis=1)
    return leverages.max()


def main(dim, mode):
    mean = np.zeros(dim)
    modes = st.multivariate_normal(mean=mean).rvs(size=mode).reshape([mode, -1])

    def target(size):
        samples = []
        for m in modes:
            samples.append(st.multivariate_normal(mean=m).rvs(int(size / mode)))

        return np.vstack(samples)

    proposal = st.multivariate_normal(mean=mean, cov=9)
    size_kns = [10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
    leverages = []
    for size_kn in size_kns:
        print(dim, mode, size_kn)
        leverages.append(leverage(target, proposal, size_kn=size_kn, alpha0=0.1, size_est=100 * size_kn))

    plt.loglog(size_kns, leverages)
    plt.show()
    print(np.array(leverages) / np.array(size_kns))


if __name__ == '__main__':
    np.random.seed(19971107)
    main(dim=4, mode=1)
