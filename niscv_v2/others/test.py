import numpy as np
import scipy.stats as st
from niscv_v2.basics.kde import KDE
from datetime import datetime as dt


def main():
    target = lambda x: st.multivariate_normal(mean=[0, 0], cov=[10, 0.1]).pdf(x)
    proposal = st.multivariate_normal(mean=[0, 0], cov=[40, 0.4])
    samples = proposal.rvs(size=1000)
    weights = target(samples) / proposal.pdf(samples)
    kde = KDE(samples, weights, local=False, gamma=0.3, bdwth=1.0)
    samples2 = proposal.rvs(size=100000)

    start = dt.now()
    a = kde.pdf(samples2)
    end = dt.now()
    print(end - start)

    start = dt.now()
    b = kde.kns(samples2)
    end = dt.now()
    print(end - start)

    start = dt.now()
    c = kde.weights.dot(b)
    end = dt.now()
    print(end - start)

    print(np.sum(np.abs(a - c)))
    print(np.mean(b <= 1e-10))


if __name__ == '__main__':
    main()
