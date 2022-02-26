import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st


def main(dim=10, rep=100000, size=100):
    target = lambda x: st.multivariate_normal(mean=np.zeros(dim)).pdf(x)
    proposal1 = st.multivariate_normal(mean=np.zeros(dim), cov=0.9 ** 2)
    proposal2 = st.multivariate_normal(mean=np.zeros(dim), cov=1.1 ** 2)
    samples1 = proposal1.rvs(size=rep * size)
    samples2 = proposal2.rvs(size=rep * size)
    weights1 = target(samples1) / proposal1.pdf(samples1)
    weights2 = target(samples2) / proposal2.pdf(samples2)
    estimate1 = weights1.reshape([rep, size]).mean(axis=1)
    estimate2 = weights2.reshape([rep, size]).mean(axis=1)
    error1 = estimate1 - 1
    error2 = estimate2 - 1
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.boxplot([error1, error2], whis=5)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    np.random.seed(19971107)
    main()
