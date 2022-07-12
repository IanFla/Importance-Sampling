import numpy as np
import scipy.stats as st
from niscv_v2.basics import utils

import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


class KDE:
    def __init__(self, centers, bdwth=1.0):
        self.centers = centers
        self.m, self.d = centers.shape
        self.factor = bdwth * (self.m ** (-1 / (self.d + 4)))
        self.cov = (self.factor ** 2) * np.array(np.cov(centers.T))
        self.sizes = None

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for center in self.centers:
            density += st.multivariate_normal.pdf(x=samples, mean=center, cov=self.cov)

        return density / self.m

    def rvs(self, size, stratify=False):
        index, self.sizes = utils.resampler(np.ones(self.m) / self.m, size, stratify)
        cum_sizes = np.append(0, np.cumsum(self.sizes))
        samples = np.zeros([size, self.d])
        for j, center in enumerate(self.centers[index]):
            samples[cum_sizes[j]:cum_sizes[j + 1]] = \
                st.multivariate_normal.rvs(size=self.sizes[j], mean=center, cov=self.cov)

        return samples

    def kns(self, samples):
        kernels = np.zeros([self.m, samples.shape[0]])
        for j, center in enumerate(self.centers):
            kernels[j] = st.multivariate_normal.pdf(x=samples, mean=center, cov=self.cov)

        return kernels


class Estimation:
    def __init__(self, target, m, r):
        self.target = target.pdf
        centers = target.rvs(m)
        self.kde = KDE(centers)
        self.n = int(r * m)
        self.res = []

    def standard(self):
        samples = self.kde.rvs(self.n)
        weights = self.target(samples) / self.kde.pdf(samples)
        self.res.append(np.mean(weights))

    def stratified(self):
        samples = self.kde.rvs(self.n, stratify=True)
        weights = self.target(samples) / self.kde.pdf(samples)
        self.res.append(np.mean(weights))

    def post_stratified_old(self):
        samples = self.kde.rvs(self.n)
        weights = self.target(samples) / self.kde.pdf(samples)
        cum_sizes = np.append(0, np.cumsum(self.kde.sizes))
        means = np.zeros(self.kde.m)
        for j, size in enumerate(self.kde.sizes):
            if size == 0:
                continue

            means[j] = np.mean(weights[cum_sizes[j]:cum_sizes[j + 1]])

        self.res.append(np.mean(means))

    def post_stratified_new(self):
        samples = self.kde.rvs(self.n)
        densities = self.kde.kns(samples)
        proposal = np.zeros(self.n)
        for j, size in enumerate(self.kde.sizes):
            proposal += size * densities[j]

        weights = self.target(samples) / (proposal / self.n)
        self.res.append(np.mean(weights))

    def run(self):
        self.standard()
        self.stratified()
        self.post_stratified_old()
        self.post_stratified_new()


def sim(it, dim):
    print(it)
    np.random.seed(1997 + 1107 * it)
    target = st.multivariate_normal(mean=np.zeros(dim))
    M = [20, 100, 500]
    R = [1, 2, 4, 8, 16, 32]
    result = np.zeros([len(M), len(R), 4])
    for i, m in enumerate(M):
        for j, r in enumerate(R):
            estimator = Estimation(target, m, r)
            estimator.run()
            result[i, j] = estimator.res

    return result


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=3) as pool:
        begin = dt.now()
        its = np.arange(300)
        R = pool.map(partial(sim, dim=5), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../../data/test/stratification', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
