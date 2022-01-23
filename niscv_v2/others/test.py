import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from niscv_v2.basics.kde2 import KDE2
import rpy2.robjects as robj


def main():
    target = lambda x: st.multivariate_normal(mean=np.zeros(2)).pdf(x)
    proposal = st.multivariate_normal(mean=np.zeros(2), cov=4)
    samples = proposal.rvs(size=1000)
    weights = target(samples) / proposal.pdf(samples)

    kde = KDE2(samples, weights, mode=1, local=False, gamma=1.0, bdwth=1.0)
    grid_x = np.linspace(-3, 3, 200)
    grid_y = np.linspace(-3, 3, 200)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_kde = kde.pdf(grids).reshape(grid_X.shape)

    fig, ax = plt.subplots(1, 2, figsize=[15, 7])
    ax[0].contour(grid_X, grid_Y, grid_Z_target)
    ax[1].contour(grid_X, grid_Y, grid_Z_kde)
    for a in ax.flatten():
        a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()

    robj.r('''
    library('support')
    supp <- function(data, dim, weights, size_kn){
        samples <- matrix(data, byrow=T, ncol=dim)
        sp(n=size_kn, p=dim, dist.samp=samples, scale.flg=F, wts=weights)$sp
    }
    ''')
    data_r = robj.FloatVector(samples.flatten())
    weights_r = robj.FloatVector(weights)
    centers = np.array(robj.r['supp'](data=data_r, dim=2, weights=weights_r, size_kn=100))
    print(centers)
    # kde = KDE2(centers, np.ones(centers.shape[0]), mode=1, local=False, gamma=1.0, bdwth=1.0)
    # grid_x = np.linspace(-3, 3, 200)
    # grid_y = np.linspace(-3, 3, 200)
    # grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    # grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    # grid_Z_target = target(grids).reshape(grid_X.shape)
    # grid_Z_kde = kde.pdf(grids).reshape(grid_X.shape)
    #
    # fig, ax = plt.subplots(1, 2, figsize=[15, 7])
    # ax[0].contour(grid_X, grid_Y, grid_Z_target)
    # ax[1].contour(grid_X, grid_Y, grid_Z_kde)
    # for a in ax.flatten():
    #     a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
    #     a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))
    #
    # fig.show()


if __name__ == '__main__':
    main()
