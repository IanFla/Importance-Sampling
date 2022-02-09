import numpy as np
from niscv_v2.basics.kde2 import KDE2
from niscv_v2.basics import utils
from wquantiles import quantile
import sklearn.linear_model as lm
import scipy.optimize as opt

import scipy.stats as st
import warnings
from datetime import datetime as dt
warnings.filterwarnings("ignore")


class KDE3(KDE2):
    def __init__(self, centers, weights, labels, mode=1, local=False, gamma=0.3, bdwth=1.0):
        super().__init__(centers, weights)
        self.labels = mode * labels
        self.kdes = []
        num = labels.max(initial=0).astype(np.int32) + 1
        for i in range(num):
            index = (labels == i)
            kde = KDE2(centers[index], weights[index], mode=mode, local=local, gamma=gamma, bdwth=bdwth)
            self.labels[index] += kde.labels
            self.kdes.extend(kde.kdes)

        num = self.labels.max(initial=0).astype(np.int32) + 1
        freqs = np.array([weights[self.labels == i].sum() for i in range(num)])
        self.props = freqs / freqs.sum()


class Qtl:
    def __init__(self, dim, target, statistic, alpha, proposal, size_est, show=True):
        self.params = {'dim': dim, 'size est': size_est}
        self.show = show
        self.cache = []
        self.result = []

        self.target = target
        self.statistic = statistic
        self.alpha = alpha
        self.ini_pdf = proposal.pdf
        self.ini_rvs = proposal.rvs

        self.indicator = None
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
        self.statistics_ = None
        self.proposal_ = None
        self.weights_ = None
        self.controls_ = None

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights, statistics, name):
        VaR = quantile(statistics, weights, self.alpha)
        if name == 'SIR':
            return VaR

        self.result.append(VaR)
        self.disp('{} est: {:.4f}'.format(name, VaR))

    def initial_estimation(self):
        samples = self.ini_rvs(self.params['size est'])
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        statistics = self.statistic(samples)
        self.__estimate(weights, statistics, 'IIS')

    def resampling(self, size_kn, ratio):
        self.params.update({'size kn': size_kn, 'ratio': ratio})
        size_est = np.round(ratio * size_kn).astype(np.int64)
        samples = self.ini_rvs(size_est)
        weights = self.__divi(self.target(samples), self.ini_pdf(samples))
        statistics = self.statistic(samples)
        VaR = self.__estimate(weights, statistics, 'SIR')

        self.indicator = lambda x: 1 * (self.statistic(x) <= VaR)
        self.opt_pdf = lambda x: self.target(x) * np.abs(self.indicator(x) - self.alpha)
        weights_kn = self.__divi(self.opt_pdf(samples), self.ini_pdf(samples))
        ESS = utils.ess(weights_kn)
        self.disp('Resampling ratio reference: {:.0f} ({:.0f})'.format(size_est / ESS, ratio))
        self.params['ESS'] = ESS

        index, sizes = utils.resampler(weights_kn, size_kn, True)
        self.centers = samples[index]
        self.weights_kn = sizes
        self.disp('Resampling rate: {}/{}'.format(self.weights_kn.size, size_kn))
        self.params['size kn*'] = self.weights_kn.size

    def density_estimation(self, mode=1, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1):
        self.params.update({'cluster': mode, 'local': local, 'gamma': gamma, 'bdwth': bdwth, 'alpha0': alpha0})
        self.kde = KDE3(self.centers, self.weights_kn, self.indicator(self.centers),
                        mode=mode, local=local, gamma=gamma, bdwth=bdwth)
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
            statistics = self.statistic(samples)
            self.__estimate(weights, statistics, 'NIS')
        elif mode == 1:
            samples = self.mix0_rvs(self.params['size est'])
            weights = self.__divi(self.target(samples), self.mix_pdf(samples))
            statistics = self.statistic(samples)
            self.__estimate(weights, statistics, 'MIS*')
        else:
            self.samples_ = self.mix_rvs(self.params['size est'])
            self.target_ = self.target(self.samples_)
            self.statistics_ = self.statistic(self.samples_)
            self.proposal_ = self.mix_pdf(self.samples_)
            self.weights_ = self.__divi(self.target_, self.proposal_)
            self.__estimate(self.weights_, self.statistics_, 'MIS')

    def control_calculation(self):
        self.controls = lambda x: self.kde.kns(x) - self.mix_pdf(x)
        self.controls_ = self.controls(self.samples_)

    def regression_estimation(self):
        X = (self.__divi(self.controls_, self.proposal_)).T
        zeta = np.linalg.solve(np.cov(X.T, bias=True), X.mean(axis=0))
        weights = self.weights_ * (1 - (X - X.mean(axis=0)).dot(zeta))
        self.__estimate(weights, self.statistics_, 'RIS')

    def asymptotic_variance(self):
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * (self.statistics_ <= self.result[-1])
        yw = y - self.alpha * w
        regw = lm.LinearRegression().fit(X, w)
        regyw = lm.LinearRegression().fit(X, yw)
        aVar = np.mean(((yw - X.dot(regyw.coef_)) / np.mean(w - X.dot(regw.coef_))) ** 2)
        self.params['aVar'] = aVar

    def likelihood_estimation(self):
        gradient = lambda zeta: np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: -self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_)) ** 2)\
            .dot(self.controls_.T) / self.controls_.shape[1]
        zeta0 = np.zeros(self.controls_.shape[0])
        res = opt.root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta0, method='lm', jac=True)
        zeta1 = res['x']
        weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
        self.__estimate(weights, self.statistics_, 'MLE')


def experiment(dim, alpha, size_est, show, size_kn, ratio, mode):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    statistic = lambda x: x[:, 0]
    proposal = st.multivariate_normal(mean=mean, cov=4)
    qtl = Qtl(dim, target, statistic, alpha, proposal, size_est, show=show)

    ts = [dt.now()]
    qtl.initial_estimation()
    ts.append(dt.now())
    qtl.resampling(size_kn, ratio)
    ts.append(dt.now())
    qtl.density_estimation(mode=mode, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=0)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=1)
    ts.append(dt.now())
    qtl.nonparametric_estimation(mode=2)
    ts.append(dt.now())
    qtl.control_calculation()
    ts.append(dt.now())
    qtl.regression_estimation()
    ts.append(dt.now())
    qtl.asymptotic_variance()
    ts.append(dt.now())
    qtl.likelihood_estimation()
    ts.append(dt.now())
    ts = np.array(ts)
    return qtl.result, qtl.params['aVar'], ts[1:] - ts[:-1]


def main():
    np.random.seed(3033079628)
    results = []
    aVars = []
    Ts = []
    for i in range(200):
        print(i + 1)
        result, aVar, ts = experiment(dim=4, alpha=0.05, size_est=5000, show=False, size_kn=200, ratio=100, mode=1)
        results.append(result)
        aVars.append(aVar)
        Ts.append(ts)

    return np.array(results), np.array(aVars), np.array(Ts)


if __name__ == '__main__':
    truth = st.norm.ppf(0.05)
    pdf = st.norm.pdf(truth)
    R, V, T = main()
    nMSE = 5000 * np.mean((R - truth) ** 2, axis=0)
    nVar = 5000 * np.var(R, axis=0)
    mV = np.mean(V) / (pdf ** 2)
    print(nMSE)
    print(nVar)
    print(mV)
    print(np.sum(T, axis=0))
