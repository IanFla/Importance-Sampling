import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st


def main(c=-1.5):
    target = lambda x: st.multivariate_normal.pdf(x=x, mean=[0, 0])
    prob = st.norm.cdf(c)
    fun = lambda x: 1.0 * (x[:, 0] <= c)
    optimal = lambda x: target(x) * np.abs(fun(x) - prob)

    grid_x = np.linspace(-3.5, 2.5, 1000)
    grid_y = np.linspace(-2.5, 2.5, 1000)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z = optimal(grids).reshape(grid_X.shape)

    fig, ax = plt.subplots(figsize=[4, 5 * 4 / 6])
    ax.contourf(grid_X, grid_Y, grid_Z, levels=30)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
