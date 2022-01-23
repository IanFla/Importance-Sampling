import numpy as np
import scipy.stats as st
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from scipy.linalg import sqrtm


def main():
    x = st.multivariate_normal.rvs(size=100, mean=np.zeros(2), cov=np.array([[1, 0], [0, 1]]))

    icov = np.linalg.inv(np.cov(x.T))
    distances1 = np.array([[mahalanobis(x1, x2, icov) for x1 in x] for x2 in x])

    scaler = StandardScaler().fit(x)
    x_ = scaler.transform(x)
    distances2 = distance_matrix(x_, x_)

    sicov = sqrtm(icov)
    x__ = x.dot(sicov)
    distances3 = distance_matrix(x__, x__)

    index1 = np.array([np.argsort(dis) for dis in distances1])
    index2 = np.array([np.argsort(dis) for dis in distances2])
    index3 = np.array([np.argsort(dis) for dis in distances3])
    print(np.all(index1 == index3))
    print(np.mean(index1 != index2))


if __name__ == '__main__':
    main()
