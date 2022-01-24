import numpy as np
import rpy2.robjects as robj


def ess(weights):
    weights = 1.0 * np.array(weights)
    weights /= weights.sum()
    return 1 / np.sum(weights ** 2)


def support(samples, weights, size_kn):
    data_r = robj.FloatVector(samples.flatten())
    weights_r = robj.FloatVector(weights / weights.sum())
    robj.r('''
        library('support')
        supp <- function(data, dim, weights, size_kn){
            samples <- matrix(data, byrow=T, ncol=dim)
            sp(n=size_kn, p=dim, dist.samp=samples, scale.flg=T, 
            wts=weights, iter.max=1000, iter.min=100, tol=1e-5)$sp
        }
        ''')
    centers = robj.r['supp'](data=data_r, dim=samples.shape[1], weights=weights_r, size_kn=size_kn)
    return np.array(centers)


def newton(gradient, hessian, x0, lim=20, sp=5):
    xs = [x0]
    for i in range(lim):
        xs.append(xs[-1] - np.linalg.solve(hessian(xs[-1]), gradient(xs[-1])))

    return np.array(xs)[::sp]


def main():
    print(ess([0, 0, 0, 1, 1.2, 10]))


if __name__ == '__main__':
    main()
