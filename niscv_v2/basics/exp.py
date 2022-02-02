import numpy as np
from matplotlib import pyplot as plt
from niscv_v2.basics.kde2 import KDE2
from niscv_v2.basics import utils
import sklearn.linear_model as lm
import scipy.optimize as opt

import scipy.stats as st
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle
import warnings
warnings.filterwarnings("ignore")


class Exp:
    def __init__(self, dim, target, fun, proposal, size_est, sn=False, adjust=True, show=True):
        self.params = {'dim': dim, 'size est': size_est, 'sn': sn, 'adjust': adjust}
        self.show = show
        self.cache = []
        self.result = []

        self.target = target
        self.fun = fun
        self.ini_pdf = proposal.pdf
        self.ini_rvs = proposal.rvs

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

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights=None, funs=None, name=None, reg=None):
        if name != 'RIS':
            mu = np.sum(weights * funs) / np.sum(weights) if self.params['sn'] else np.mean(weights * funs)
            if name == 'SIR':
                return mu
        else:
            X = reg[0]
            w = reg[1]
            y = reg[2]
            mu = np.sum(y - X.dot(self.reg_y.coef_)) / np.sum(w - X.dot(self.reg_w.coef_)) \
                if self.params['sn'] else np.mean(y - X.dot(self.reg_y.coef_))

        if 'mu' in self.params.keys():
            mu += self.params['mu']

        self.result.append(mu)
        self.disp('{} est: {:.4f}'.format(name, mu))

    def initial_estimation(self):
        samples = self.ini_rvs(self.params['size est'])
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'IIS')

    def resampling(self, size_kn, ratio, bootstrap='st'):
        self.params.update({'size kn': size_kn, 'ratio': ratio, 'bootstrap': bootstrap})
        size_est = np.round(ratio * size_kn).astype(np.int64)
        samples = self.ini_rvs(size_est)
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        funs = self.fun(samples)
        mu = self.__estimate(weights, funs, 'SIR')
        if self.params['adjust']:
            self.params['mu'] = mu
            fun = self.fun
            self.fun = lambda x: fun(x) - mu
            self.opt_pdf = lambda x: self.target(x) * np.abs(self.fun(x))
        else:
            self.opt_pdf = (lambda x: self.target(x) * np.abs(self.fun(x) - mu)) \
                if self.params['sn'] else (lambda x: self.target(x) * np.abs(self.fun(x)))

        if ratio > 0:
            weights_kn = self.__divi(self.opt_pdf(samples), self.ini_pdf(samples))
            ESS = utils.ess(weights_kn)
            self.disp('Resampling ratio reference: {:.0f} ({:.0f})'.format(size_est / ESS, ratio))
            self.params['ESS'] = ESS
            if bootstrap == 'mt' or bootstrap == 'st':
                index, sizes = utils.resampler(weights_kn, size_kn, True if bootstrap == 'st' else False)
                self.centers = samples[index]
                self.weights_kn = sizes
                self.disp('Resampling rate: {}/{}'.format(self.weights_kn.size, size_kn))
                self.params['size kn*'] = self.weights_kn.size
            else:
                self.centers = utils.support(samples, weights_kn / weights_kn.sum(), size_kn)
                self.weights_kn = np.ones(size_kn)
        else:
            self.centers = self.ini_rvs(size_kn)
            self.weights_kn = self.__divi(self.opt_pdf(self.centers), self.ini_pdf(self.centers))

    def density_estimation(self, mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1):
        self.params.update({'cluster': mode, 'local': local, 'gamma': gamma, 'bdwth': bdwth, 'alpha0': alpha0})
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

    def nonparametric_estimation(self, mode):
        if mode == 0:
            samples = self.kde_rvs(self.params['size est'])
            weights = self.__divi(self.target(samples), self.kde_pdf(samples))
            funs = self.fun(samples)
            self.__estimate(weights, funs, 'NIS')
        elif mode == 1:
            samples = self.mix0_rvs(self.params['size est'])
            weights = self.__divi(self.target(samples), self.mix_pdf(samples))
            funs = self.fun(samples)
            self.__estimate(weights, funs, 'MIS*')
        else:
            self.samples_ = self.mix_rvs(self.params['size est'])
            self.target_ = self.target(self.samples_)
            self.funs_ = self.fun(self.samples_)
            self.proposal_ = self.mix_pdf(self.samples_)
            self.weights_ = self.__divi(self.target_, self.proposal_)
            self.__estimate(self.weights_, self.funs_, 'MIS')

    def control_calculation(self):
        self.controls = lambda x: self.kde.kns(x) - self.mix_pdf(x)
        self.controls_ = self.controls(self.samples_)

    def regression_estimation(self):
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * self.funs_
        self.reg_y = lm.LinearRegression().fit(X, y)
        if self.params['sn']:
            self.reg_w = lm.LinearRegression().fit(X, w)
            self.disp('Regression R2: {:.4f} / {:.4f}'.format(self.reg_y.score(X, y), self.reg_w.score(X, w)))
            self.params['R2'] = [self.reg_y.score(X, y), self.reg_w.score(X, w)]
        else:
            self.disp('Regression R2: {:.4f}'.format(self.reg_y.score(X, y)))
            self.params['R2'] = self.reg_y.score(X, y)

        self.__estimate(name='RIS', reg=[X, w, y])

    def likelihood_estimation(self):
        gradient = lambda zeta: np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: -self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_)) ** 2)\
            .dot(self.controls_.T) / self.controls_.shape[1]
        zeta0 = np.zeros(self.controls_.shape[0])
        res = opt.root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta0, method='lm', jac=True)
        zeta1 = res['x']
        weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
        self.__estimate(weights, self.funs_, 'MLE')

    def draw(self, grid_x, name, d=0):
        grid_X = np.zeros([grid_x.size, self.params['dim']])
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
            mu = self.result[-1] if 'mu' not in self.params.keys() else self.result[-1] - self.params['mu']
            reg_pdf = (self.reg_y.coef_ - mu * self.reg_w.coef_).dot(self.controls(grid_X)) if self.params['sn'] else \
                self.reg_y.coef_.dot(self.controls(grid_X)) + mu * self.mix_pdf(grid_X)
            ax.plot(grid_x, np.abs(reg_pdf))
            ax.legend(['optimal proposal', 'regression proposal'])
        else:
            print('name err! ')

        ax.set_title('{}-D {} estimation ({}d slicing)'.format(self.params['dim'], name, d + 1))
        plt.show()


def experiment(dim, size_est, sn, adjust, show, size_kn, ratio, bootstrap):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** 2
    proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Exp(dim, target, fun, proposal, size_est, sn=sn, adjust=adjust, show=show)

    exp.initial_estimation()
    exp.resampling(size_kn, ratio, bootstrap=bootstrap)
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    exp.nonparametric_estimation(mode=0)
    exp.nonparametric_estimation(mode=1)
    exp.nonparametric_estimation(mode=2)
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.control_calculation()
    exp.regression_estimation()
    if exp.show:
        exp.draw(grid_x, name='regression')

    exp.likelihood_estimation()
    return exp.result, exp.params


def run(it, dim):
    print(it, end=' ')
    settings = [[False, 'st'], [True, 'sp'], [False, 'st'], [True, 'sp']]
    results = []
    for setting in settings:
        np.random.seed(1997 * it + 1107)
        results.append(experiment(dim=dim, size_est=10000, sn=setting[0], adjust=False, show=False,
                                  size_kn=500, ratio=20, bootstrap=setting[1]))

    return results


def main(dim):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=32) as pool:
        begin = dt.now()
        its = np.arange(100)
        R = pool.map(partial(run, dim=dim), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/test/exp_data', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    # main(dim=5)
    np.random.seed(1234)
    experiment(dim=4, size_est=5000, sn=True, adjust=False, show=True,
               size_kn=300, ratio=20, bootstrap='mt')
    np.random.seed(1234)
    experiment(dim=4, size_est=5000, sn=True, adjust=False, show=True,
               size_kn=300, ratio=20, bootstrap='st')
    np.random.seed(1234)
    experiment(dim=4, size_est=5000, sn=True, adjust=True, show=True,
               size_kn=300, ratio=20, bootstrap='sp')
