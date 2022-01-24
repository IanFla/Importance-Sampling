import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from particles import resampling as rs
from niscv_v2.basics.kde import KDE

from matplotlib import pyplot as plt
import scipy.stats as st


class KDE2:
    def __init__(self, centers, weights, mode=1, labels=None, local=False, gamma=0.3, bdwth=1.0):
        if mode == 1:
            labels = np.zeros(centers.shape[0]).astype(np.int32)
        elif mode > 1:
            scaler = StandardScaler().fit(centers, sample_weight=weights)
            kmeans = KMeans(n_clusters=mode).fit(scaler.transform(centers), sample_weight=weights)
            labels = kmeans.labels_

        self.labels = labels
        num = labels.max(initial=0).astype(np.int32) + 1
        freq = np.array([weights[labels == i].sum() for i in range(num)])
        self.prop = freq / freq.sum()
        self.kdes = [KDE(centers[labels == i], weights[labels == i], local, gamma, bdwth) for i in range(num)]

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for l, kde in enumerate(self.kdes):
            density += self.prop[l] * kde.pdf(samples)

        return density

    def rvs(self, size, stratify=False):
        if stratify:
            index, sizes = np.unique(rs.stratified(self.prop, M=size), return_counts=True)
        else:
            index, sizes = np.unique(rs.multinomial(self.prop, M=size), return_counts=True)

        return np.vstack([self.kdes[index[j]].rvs(sizes[j], stratify) for j in range(index.size)])

    def kns(self, samples):
        return np.vstack([kde.kns(samples) for kde in self.kdes])


def main(mode, local, gamma, bdwth, seed=19971107):
    np.random.seed(seed)
    target = lambda x: 0.7 * st.multivariate_normal(mean=[-1, 0], cov=0.4).pdf(x) + \
                       0.3 * st.multivariate_normal(mean=[2, 0], cov=0.2).pdf(x)
    proposal = st.multivariate_normal(mean=[0, 0], cov=4).pdf
    centers = st.multivariate_normal(mean=[0, 0], cov=4).rvs(size=1000)
    weights = target(centers) / proposal(centers)

    kde1 = KDE(centers, weights, local=local, gamma=gamma, bdwth=bdwth)
    samples1 = kde1.rvs(size=2000)
    kde2 = KDE2(centers, weights, mode=mode, local=local, gamma=gamma, bdwth=bdwth)
    samples2 = kde2.rvs(size=2000)

    grid_x = np.linspace(-4, 4, 200)
    grid_y = np.linspace(-2, 2, 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_proposal = proposal(grids).reshape(grid_X.shape)
    grid_Z_kde1 = kde1.pdf(grids).reshape(grid_X.shape)
    grid_Z_kde2 = kde2.pdf(grids).reshape(grid_X.shape)

    fig, ax = plt.subplots(2, 2, figsize=[15, 10])
    ax[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    ax[0, 1].contour(grid_X, grid_Y, grid_Z_kde1)
    ax[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    ax[1, 0].scatter(centers[:, 0], centers[:, 1])
    ax[1, 1].contour(grid_X, grid_Y, grid_Z_kde1)
    ax[1, 1].scatter(samples1[:, 0], samples1[:, 1])
    for a in ax.flatten():
        a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()

    fig, ax = plt.subplots(2, 2, figsize=[15, 10])
    ax[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    ax[0, 1].contour(grid_X, grid_Y, grid_Z_kde2)
    ax[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    ax[1, 0].scatter(centers[:, 0], centers[:, 1])
    ax[1, 1].contour(grid_X, grid_Y, grid_Z_kde2)
    ax[1, 1].scatter(samples2[:, 0], samples2[:, 1])
    for a in ax.flatten():
        a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()

    print(kde1.kns(samples1).shape)
    print(kde2.kns(samples1).shape)


if __name__ == '__main__':
    main(mode=2, local=False, gamma=0.3, bdwth=1.0)
