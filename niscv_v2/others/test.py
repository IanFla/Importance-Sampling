import numpy as np
import scipy.stats as st
from niscv_v2.basics.kde2 import KDE2
import matplotlib.pyplot as plt
from niscv_v2.basics.utils import support


def main():
    target = lambda x: st.multivariate_normal(mean=[0, 0], cov=[10, 0.1]).pdf(x)
    proposal = st.multivariate_normal(mean=[0, 0], cov=[40, 0.4])
    samples = proposal.rvs(size=1000000)
    weights = target(samples) / proposal.pdf(samples)
    centers = support(samples, weights, 300)
    kde = KDE2(centers, np.ones(centers.shape[0]), mode=1, local=False, gamma=0.3, bdwth=1.0)

    grid_x = np.linspace(-10, 10, 200)
    grid_y = np.linspace(-1, 1, 200)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_kde = kde.pdf(grids).reshape(grid_X.shape)

    fig, ax = plt.subplots(1, 2, figsize=[15, 7])
    ax[0].contour(grid_X, grid_Y, grid_Z_target)
    ax[0].scatter(centers[:, 0], centers[:, 1])
    ax[1].contour(grid_X, grid_Y, grid_Z_kde)
    for a in ax.flatten():
        a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()


if __name__ == '__main__':
    main()
