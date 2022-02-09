import numpy as np
from matplotlib import pyplot as plt
from niscv_v2.experiments.garch_truth import garch_model
from niscv_v2.basics.qtl import Qtl
import pandas as pd
import seaborn as sb


def draw(qtl):
    df = pd.DataFrame(qtl.centers, columns=[r'$\log(\phi_0)$', r'$\phi_1$', r'$\beta$', r'$y_{T+1}$', r'$y_{T+2}$'])
    df['cluster'] = qtl.kde.labels + 1
    plt.style.use('ggplot')
    sb.pairplot(df, hue='cluster')
    plt.show()


def main(D=2, alpha=0.05):
    np.random.seed(19971107)
    target, statistic, proposal = garch_model(D)
    qtl = Qtl(D + 3, target, statistic, alpha, proposal, size_est=10000, show=False)
    qtl.resampling(size_kn=2000, ratio=400)
    qtl.density_estimation(mode=2, local=False, gamma=0.3, bdwth=1.0, alpha0=0.1)
    draw(qtl)


if __name__ == '__main__':
    main()
