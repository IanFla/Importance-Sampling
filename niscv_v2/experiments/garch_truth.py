import numpy as np
from matplotlib import pyplot as plt
from niscv_v2.basics.garch import GARCH


class IP:
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs


def garch_model(d):
    garch = GARCH()
    garch.laplace(inflate=2, df=1)
    target = lambda x: garch.target(x[:, :3], x[:, 3:])
    statistic = lambda x: x[:, 3:].sum(axis=1)
    proposal = IP(pdf=lambda x: garch.proposal(x[:, :3], x[:, 3:]),
                  rvs=lambda size: np.hstack(garch.predict(d, size)))
    return target, statistic, proposal


def main():
    garch = GARCH()
    garch.laplace(inflate=2, df=1)


if __name__ == '__main__':
    main()
