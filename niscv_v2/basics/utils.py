import numpy as np
from particles import resampling as rs
import rpy2.robjects as robj
import scipy.stats as st


def ess(weights):
    weights = 1.0 * np.array(weights)
    weights /= weights.sum()
    return 1 / np.sum(weights ** 2)


def resampler(weights, size, stratify):
    if stratify:
        index, sizes = np.unique(rs.stratified(weights / weights.sum(), M=size), return_counts=True)
    else:
        index, sizes = np.unique(rs.multinomial(weights / weights.sum(), M=size), return_counts=True)

    return index, sizes


def support(samples, weights, size_kn):
    data_r = robj.FloatVector(samples.flatten())
    weights_r = robj.FloatVector(weights / weights.sum())
    robj.r('''
        library('support')
        supp <- function(data, dim, weights, size_kn){
            samples <- matrix(data, byrow=T, ncol=dim)
            sp(n=size_kn, p=dim, dist.samp=samples, scale.flg=T, wts=weights, 
            iter.max=1000, iter.min=100, tol=1e-5, par.flg=F)$sp
        }
        ''')
    centers = robj.r['supp'](data=data_r, dim=samples.shape[1], weights=weights_r, size_kn=size_kn)
    return np.array(centers)


def newton(gradient, hessian, x0, lim=20, sep=5):
    xs = [x0]
    for i in range(lim):
        xs.append(xs[-1] - np.linalg.solve(hessian(xs[-1]), gradient(xs[-1])))

    return np.array(xs)[::sep]


def integrand(m, c):
    if m >= 0:
        return lambda x: x[:, 0] ** m + c
    else:
        return lambda x: 1 * (x[:, 0] >= c)


def truth(m, c):
    if m >= 0:
        return st.norm.moment(m) + c
    else:
        return st.norm.cdf(-c)


def main():
    print(ess([0, 0, 0, 1, 1.2, 10]))


if __name__ == '__main__':
    main()
