import numpy as np
from matplotlib import pyplot as plt
from niscv_v2.basics.example.garch import GARCH


def main():
    garch = GARCH()
    estimator = garch.laplace(inflate=2, df=1)
    print(estimator(100000))


if __name__ == '__main__':
    main()
