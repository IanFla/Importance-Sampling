import numpy as np
from matplotlib import pyplot as plt


def newton(gradient, hessian, x0, lim=10):
    xs = [x0]
    for i in range(lim):
        xs.append(xs[-1] - np.linalg.solve(hessian(xs[-1]), gradient(xs[-1])))

    return np.array(xs)


    def regression_estimation(self, mode):
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * self.funs_
        if mode == 1:
            self.reg_y = lm.LinearRegression().fit(X, y)
            if self.params['sn']:
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
            mid = self.params['size est'] // 2
            flags = (np.random.permutation(np.append(np.ones(mid), np.zeros(self.params['size est'] - mid))) == 1)
            X1, X2 = X[flags], X[~flags]
            w1, w2 = w[flags], w[~flags]
            y1, y2 = y[flags], y[~flags]
            reg1_y = lm.LinearRegression().fit(X1, y1)
            reg2_y = lm.LinearRegression().fit(X2, y2)
            if self.params['sn']:
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

    def likelihood_estimation(self):
        # loglikelihood = lambda zeta: -np.mean(np.log(self.proposal_ + zeta.dot(self.controls_)))
        gradient = lambda zeta: -np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_)) ** 2)\
                                   .dot(self.controls_.T) / self.controls_.shape[1]
        # X = (self.__divi(self.controls_, self.proposal_)).T
        # zeta0 = np.linalg.solve(np.cov(X.T, bias=True), X.mean(axis=0))
        # zeta0 = np.zeros(self.controls_.shape[0]) if np.isnan(loglikelihood(zeta0)) else zeta0
        zeta0 = np.zeros(self.controls_.shape[0])
        res = opt.root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta0, method='lm', jac=True)
        zeta1 = res['x']
        self.disp('Dist/Norm (zeta(Opt),zeta(Ini)): {:.4f}/({:.4f},{:.4f})'
                  .format(np.sqrt(np.sum((zeta1 - zeta0) ** 2)),
                          np.sqrt(np.sum(zeta1 ** 2)), np.sqrt(np.sum(zeta0 ** 2))))
        weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
        self.__estimate(weights, self.funs_, 'MLE')


def support(samples, weights, size_kn):
    robj.r('''
        library('support')
        supp <- function(data, dim, weights, size_kn){
            samples <- matrix(data, byrow=T, ncol=dim)
            sp(n=size_kn, p=dim, dist.samp=samples, scale.flg=T, 
            wts=weights, iter.max=10000, iter.min=100, tol=1e-6)$sp
        }
        ''')
    data_r = robj.FloatVector(samples.flatten())
    weights_r = robj.FloatVector(weights / weights.sum())
    centers = robj.r['supp'](data=data_r, dim=samples.shape[1], weights=weights_r, size_kn=size_kn)
    return np.array(centers)


def main():
    target = lambda x: st.multivariate_normal(mean=[0, 0], cov=[10, 0.1]).pdf(x)
    proposal = st.multivariate_normal(mean=[0, 0], cov=[40, 0.4])
    samples = proposal.rvs(size=3000)
    weights = target(samples) / proposal.pdf(samples)
    centers = support(samples, weights, 100)
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



def random_walk(target, x0, cov, factor, burn, size, thin):
    walk = lambda x: st.multivariate_normal(mean=x, cov=(factor ** 2) * cov).rvs()
    for b in range(burn):
        x1 = walk(x0)
        if (target(x1) / target(x0)) >= st.uniform.rvs():
            x0 = np.copy(x1)

    xs = []
    for s in range(size):
        for t in range(thin):
            x1 = walk(x0)
            if (target(x1) / target(x0)) >= st.uniform.rvs():
                x0 = np.copy(x1)

        xs.append(x0)

    return np.array(xs)


def experiment(it, D, size):
    print('it:', it, D)
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, None, proposal, size_est=None, show=False)
    samples = qtl.ini_rvs(100000)
    weights = target(samples) / (qtl.ini_pdf(samples) + 1.0 * (qtl.ini_pdf(samples) == 0))
    mean = np.sum(weights * samples.T, axis=1) / np.sum(weights)
    cov = np.cov(samples.T, aweights=weights)
    target2 = lambda x: target(x.reshape([1, -1]))[0]
    samples2 = random_walk(target=target2, x0=mean, cov=cov, factor=1.7 / np.sqrt(D + 3),
                           burn=100, size=size, thin=10)
    statistics = statistic(samples2)
    return statistics