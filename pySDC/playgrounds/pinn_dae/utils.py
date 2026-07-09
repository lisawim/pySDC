import matplotlib.pyplot as plt


def my_setup_mpl(fontsize: int = 16) -> None:
    """
    Setting up my personal settings for plotting.

    Parameters
    ----------
    fontsize : int, optional
        Fontsize used in plot.
    """

    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.linewidth'] = 0.5

    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.6
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.minor.size"] = 1

    plt.rcParams['lines.linewidth'] = 0.9
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = 2.9

    # sets fig.tight_layout()
    plt.rcParams["figure.autolayout"] = True