import numpy as np
import scipy.stats as st
from scipy.linalg import sqrtm
from scipy.spatial import distance_matrix
from particles import resampling as rs

from matplotlib import pyplot as plt
from datetime import datetime as dt


class KDE:
    def __init__(self, centers, weights, local=False, gamma=0.3, bdwth=1.0):
        self.centers = centers
        self.weights = weights / weights.sum()
        self.m, self.d = centers.shape
        self.ESS = 1 / np.sum(self.weights ** 2)
        self.local = local
        if self.local:
            centers_ = centers.dot(sqrtm(np.linalg.inv(np.cov(self.centers.T, aweights=weights))))
            distances = distance_matrix(centers_, centers_)
            covs = []
            for j, center in enumerate(self.centers):
                index = np.argsort(distances[j])[:np.around(gamma * self.m).astype(np.int64)]
                covs.append(np.cov(self.centers[index].T, aweights=weights[index]))

        else:
            covs = np.cov(centers.T, aweights=weights)

        self.factor = bdwth * (self.ESS ** (-1 / (self.d + 4))) * (gamma ** (-1 / self.d) if self.local else 1.0)
        self.covs = (self.factor ** 2) * np.array(covs)

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.covs
            density += self.weights[j] * st.multivariate_normal.pdf(x=samples, mean=center, cov=cov)

        return density

    def rvs(self, size, stratify=True):
        if stratify:
            index, sizes = np.unique(rs.stratified(self.weights, M=size), return_counts=True)
        else:
            index, sizes = np.unique(rs.multinomial(self.weights, M=size), return_counts=True)

        cum_sizes = np.append(0, np.cumsum(sizes))
        samples = np.zeros([size, self.d])
        for j, center in enumerate(self.centers[index]):
            cov = self.covs[j] if self.local else self.covs
            samples[cum_sizes[j]:cum_sizes[j + 1]] = st.multivariate_normal.rvs(size=sizes[j], mean=center, cov=cov)

        return samples

    def kns(self, samples):
        kernels = np.zeros([self.m, samples.shape[0]])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.covs
            kernels[j] = st.multivariate_normal.pdf(x=samples, mean=center, cov=cov)

        return kernels


def main(local, gamma, bdwth, stratify, seed=19971107):
    np.random.seed(seed)
    target = lambda x: 0.7 * st.multivariate_normal(mean=[-1, 0], cov=[8, 0.2]).pdf(x) + \
                       0.3 * st.multivariate_normal(mean=[1, 0], cov=[0.25, 4]).pdf(x)
    proposal = st.multivariate_normal(mean=[-1, 0], cov=4).pdf
    centers = st.multivariate_normal(mean=[-1, 0], cov=4).rvs(size=1000)
    weights = target(centers) / proposal(centers)

    start1 = dt.now()
    kde = KDE(centers, weights, local=local, gamma=gamma, bdwth=bdwth)
    end1 = dt.now()
    start2 = dt.now()
    samples = kde.rvs(size=1000, stratify=stratify)
    end2 = dt.now()

    grid_x = np.linspace(-7, 5, 200)
    grid_y = np.linspace(-4, 4, 200)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_proposal = proposal(grids).reshape(grid_X.shape)
    start3 = dt.now()
    grid_Z_kde = kde.pdf(grids).reshape(grid_X.shape)
    end3 = dt.now()

    fig, ax = plt.subplots(2, 2, figsize=[15, 10])
    ax[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    ax[0, 1].contour(grid_X, grid_Y, grid_Z_kde)
    ax[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    ax[1, 0].scatter(centers[:, 0], centers[:, 1])
    ax[1, 1].contour(grid_X, grid_Y, grid_Z_kde)
    ax[1, 1].scatter(samples[:, 0], samples[:, 1])
    for a in ax.flatten():
        a.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        a.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    plt.tight_layout()
    plt.show()
    return end1 - start1, end2 - start2, end3 - start3


if __name__ == '__main__':
    start = dt.now()
    res = main(local=False, gamma=0.3, bdwth=1.0, stratify=True)
    end = dt.now()
    print(end - start)
    print(res)
