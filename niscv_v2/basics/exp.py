import numpy as np
from matplotlib import pyplot as plt
from niscv_v2.basics.kde2 import KDE2
from niscv_v2.basics import utils
import sklearn.linear_model as lm

import scipy.stats as st
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle
import warnings
warnings.filterwarnings("ignore")


class Exp:
    def __init__(self, dim, target, fun, proposal, size_est, sn=False, show=True):
        self.params = {'dim': dim, 'size est': size_est, 'sn': sn}
        self.dim = dim
        self.sn = sn
        self.show = show
        self.cache = []
        self.result = []

        self.target = target
        self.fun = fun
        self.ini_pdf = proposal.pdf
        self.ini_rvs = proposal.rvs
        self.size_est = size_est

        self.opt_pdf = None
        self.centers = None
        self.weights_kn = None

        self.kde = None
        self.kde_pdf = None
        self.kde_rvs = None
        self.mix_pdf = None
        self.mix_rvs = None
        self.mix0_rvs = None
        self.controls = None

        self.samples_ = None
        self.target_ = None
        self.funs_ = None
        self.proposal_ = None
        self.weights_ = None
        self.controls_ = None

        self.reg_y = None
        self.reg_w = None
        self.mu = None

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights=None, funs=None, name=None, mode=None, reg=None):
        if mode != 'regress':
            mu = np.sum(weights * funs) / np.sum(weights) if self.sn else np.mean(weights * funs)
            if mode != 'resample':
                self.result.append(mu)
                self.disp('{} est: {:.4f}'.format(name, mu))
            else:
                return mu
        else:
            X = reg[0]
            w = reg[1]
            y = reg[2]
            self.mu = np.sum(y - X.dot(reg[3])) / np.sum(w - X.dot(reg[4])) if self.sn else np.mean(y - X.dot(reg[3]))

    def initial_estimation(self):
        samples = self.ini_rvs(self.size_est)
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'IIS')

    def resampling(self, size_kn, ratio, resample=True, bootstrap=True):
        self.params.update({'size kn': size_kn, 'ratio': ratio, 'resample': resample, 'bootstrap': bootstrap})
        size_est = np.round(ratio * size_kn).astype(np.int64)
        samples = self.ini_rvs(size_est)
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        funs = self.fun(samples)
        mu = self.__estimate(weights, funs, mode='resample')
        self.opt_pdf = (lambda x: self.target(x) * np.abs(self.fun(x) - mu)) \
            if self.sn else (lambda x: self.target(x) * np.abs(self.fun(x)))

        if resample:
            weights_kn = self.__divi(self.opt_pdf(samples), self.ini_pdf(samples))
            ESS = utils.ess(weights_kn)
            self.disp('Resampling ratio reference: {:.0f} ({:.0f})'.format(size_est / ESS, ratio))
            self.params['ESS'] = ESS
            if bootstrap:
                index, sizes = utils.resampler(weights_kn, size_kn, True)
                self.centers = samples[index]
                self.weights_kn = sizes
                self.disp('Resampling rate: {}/{}'.format(self.weights_kn.size, size_kn))
                self.params['size* kn'] = self.weights_kn.size
            else:
                self.centers = utils.support(samples, weights_kn / weights_kn.sum(), size_kn)
                self.weights_kn = np.ones(size_kn)

            return ESS

        else:
            self.centers = self.ini_rvs(size_kn)
            self.weights_kn = self.__divi(self.opt_pdf(self.centers), self.ini_pdf(self.centers))

    def density_estimation(self, mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1):
        self.params.update({'cluster': mode, 'local': local, 'gamma': gamma, 'alpha0': alpha0})
        labels = 1.0 * (self.fun(self.centers) > 0)
        self.kde = KDE2(self.centers, self.weights_kn, mode=mode, labels=labels, local=local, gamma=gamma, bdwth=bdwth)

        self.kde_pdf = self.kde.pdf
        self.kde_rvs = self.kde.rvs
        self.mix_pdf = lambda x: alpha0 * self.ini_pdf(x) + (1 - alpha0) * self.kde_pdf(x)
        self.mix_rvs = lambda size: np.vstack([self.ini_rvs(round(alpha0 * size)),
                                               self.kde_rvs(size - round(alpha0 * size), stratify=True)])

        def mix0_rvs(size):
            size0 = np.random.binomial(n=size, p=alpha0)
            return np.vstack([self.ini_rvs(size0), self.kde_rvs(size - size0)])

        self.mix0_rvs = mix0_rvs
        self.controls = lambda x: self.kde.kns(x) - self.mix_pdf(x)

    def nonparametric_estimation(self, mode):
        if mode == 0:
            samples = self.kde_rvs(self.size_est)
            weights = self.__divi(self.target(samples), self.kde_pdf(samples))
            funs = self.fun(samples)
            self.__estimate(weights, funs, 'NIS')
        elif mode == 1:
            samples = self.mix0_rvs(self.size_est)
            weights = self.__divi(self.target(samples), self.mix_pdf(samples))
            funs = self.fun(samples)
            self.__estimate(weights, funs, 'MIS*')
        else:
            self.samples_ = self.mix_rvs(self.size_est)
            self.target_ = self.target(self.samples_)
            self.funs_ = self.fun(self.samples_)
            self.proposal_ = self.mix_pdf(self.samples_)
            self.weights_ = self.__divi(self.target_, self.proposal_)
            self.__estimate(self.weights_, self.funs_, 'MIS')

    def control_calculation(self):
        self.controls_ = self.controls(self.samples_)

    def regression_estimation(self, mode):
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * self.funs_
        if mode == 1:
            self.reg_y = lm.LinearRegression().fit(X, y)
            if self.sn:
                self.reg_w = lm.LinearRegression().fit(X, w)
                self.disp('Regression R2: {:.4f} / {:.4f}'.format(self.reg_y.score(X, y), self.reg_w.score(X, w)))
                self.params['R2'] = [self.reg_y.score(X, y), self.reg_w.score(X, w)]
                self.__estimate(mode='regress', reg=[X, w, y, self.reg_y.coef_, self.reg_w.coef_])
            else:
                self.disp('Regression R2: {:.4f}'.format(self.reg_y.score(X, y)))
                self.params['R2'] = self.reg_y.score(X, y)
                self.__estimate(mode='regress', reg=[X, w, y, self.reg_y.coef_])

            self.result.append(self.mu)
            self.disp('RIS est: {:.4f}'.format(self.mu))
        else:
            mid = self.size_est // 2
            flags = (np.random.permutation(np.append(np.ones(mid), np.zeros(self.size_est - mid))) == 1)
            X1, X2 = X[flags], X[~flags]
            w1, w2 = w[flags], w[~flags]
            y1, y2 = y[flags], y[~flags]
            reg1_y = lm.LinearRegression().fit(X1, y1)
            reg2_y = lm.LinearRegression().fit(X2, y2)
            if self.sn:
                reg1_w = lm.LinearRegression().fit(X1, w1)
                reg2_w = lm.LinearRegression().fit(X2, w2)
                self.disp('Regression 1 R2: {:.4f} / {:.4f}'.format(reg1_y.score(X2, y2), reg1_w.score(X2, w2)))
                self.disp('Regression 2 R2: {:.4f} / {:.4f}'.format(reg2_y.score(X1, y1), reg2_w.score(X1, w1)))
                self.__estimate(mode='regress', reg=[X2, w2, y2, reg1_y.coef_, reg1_w.coef_])
                mu1 = self.mu.copy()
                self.__estimate(mode='regress', reg=[X1, w1, y1, reg2_y.coef_, reg2_w.coef_])
                mu2 = self.mu.copy()
            else:
                self.disp('Regression 1 R2: {:.4f}'.format(reg1_y.score(X2, y2)))
                self.disp('Regression 2 R2: {:.4f}'.format(reg2_y.score(X1, y1)))
                self.__estimate(mode='regress', reg=[X2, w2, y2, reg1_y.coef_])
                mu1 = self.mu.copy()
                self.__estimate(mode='regress', reg=[X1, w1, y1, reg2_y.coef_])
                mu2 = self.mu.copy()

            self.result.append((mu1 + mu2) / 2)
            self.disp('RIS* est: {:.4f}'.format((mu1 + mu2) / 2))

    def likelihood_estimation(self, lim=20, sep=5):
        self.params.update({'lim': lim, 'sep': sep})
        gradient = lambda zeta: np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: -self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_)) ** 2)\
            .dot(self.controls_.T) / self.controls_.shape[1]
        X = (self.__divi(self.controls_, self.proposal_)).T
        zeta0 = np.linalg.solve(np.cov(X.T, bias=True), X.mean(axis=0))

        zetas = utils.newton(gradient, hessian, zeta0, lim=lim, sep=sep)
        self.disp('Dist/Norm (zeta(Opt),zeta(The)): {:.4f}/({:.4f},{:.4f})'
                  .format(np.sqrt(np.sum((zetas[-1] - zeta0) ** 2)),
                          np.sqrt(np.sum(zetas[-1] ** 2)), np.sqrt(np.sum(zeta0 ** 2))))
        for i, zetai in enumerate(zetas):
            weights = self.__divi(self.target_, self.proposal_ + zetai.dot(self.controls_))
            self.__estimate(weights, self.funs_, 'MLE({})'.format(sep * i))

    def draw(self, grid_x, name, d=0):
        grid_X = np.zeros([grid_x.size, self.dim])
        grid_X[:, d] = grid_x
        opt_pdf = self.opt_pdf(grid_X)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid_x, opt_pdf)
        if name == 'initial':
            ini_pdf = self.ini_pdf(grid_X)
            ax.plot(grid_x, opt_pdf.max() * ini_pdf / ini_pdf.max())
            ax.legend(['optimal proposal', 'initial proposal'])
        elif name == 'nonparametric':
            kde_pdf = self.kde_pdf(grid_X)
            mix_pdf = self.mix_pdf(grid_X)
            ax.plot(grid_x, opt_pdf.max() * kde_pdf / kde_pdf.max())
            ax.plot(grid_x, opt_pdf.max() * mix_pdf / mix_pdf.max())
            ax.legend(['optimal proposal', 'nonparametric proposal', 'mixture proposal'])
        elif name == 'regression':
            reg_pdf = (self.reg_y.coef_ - self.mu * self.reg_w.coef_).dot(self.controls(grid_X)) \
                if self.sn else self.reg_y.coef_.dot(self.controls(grid_X)) + self.mu * self.mix_pdf(grid_X)
            ax.plot(grid_x, np.abs(reg_pdf))
            ax.legend(['optimal proposal', 'regression proposal'])
        else:
            print('name err! ')

        ax.set_title('{}-D {} estimation ({}d slicing)'.format(self.dim, name, d + 1))
        plt.show()


def experiment(dim, size_est, sn, show, size_kn, ratio, bootstrap):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** 2
    proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, show=show)

    exp.initial_estimation()
    exp.resampling(size_kn, ratio, resample=True, bootstrap=bootstrap)
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    exp.nonparametric_estimation(mode=0)
    exp.nonparametric_estimation(mode=1)
    exp.nonparametric_estimation(mode=2)
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.control_calculation()
    exp.regression_estimation(mode=0)
    exp.regression_estimation(mode=1)
    if exp.show:
        exp.draw(grid_x, name='regression')

    exp.likelihood_estimation(lim=12, sep=4)
    return exp.result, exp.params


def run(it, dim):
    print(it, end=' ')
    settings = [[False, True], [True, True], [False, False], [True, False]]
    results = []
    for setting in settings:
        np.random.seed(1997 * it + 1107)
        results.append(experiment(dim=dim, size_est=40000, sn=setting[0], show=False,
                                  size_kn=300, ratio=20, bootstrap=setting[1]))

    return results


def main(dim):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=32) as pool:
        begin = dt.now()
        its = np.arange(100)
        R = pool.map(partial(run, dim=dim), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/test/data_exp', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(dim=5)
