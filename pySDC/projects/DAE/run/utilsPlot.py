import matplotlib.pyplot as plt
from pathlib import Path


class Plotter:
    def __init__(self, nrows=1, ncols=1, orientation=None, figsize=(10, 5)):
        self.nrows = nrows
        self.ncols = ncols
        self.orientation = orientation
        self.figsize = figsize

        if self.orientation == 'horizontal':
            self.nrows = 1
            self.ncols = self.nrows * self.ncols
        elif self.orientation == 'vertical':
            self.ncols = 1
            self.nrows = self.nrows * self.ncols
        elif self.orientation is not None:
            raise ValueError("Orientation must be either 'horizontal', 'vertical', or None")
        
        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
        
        if self.nrows * self.ncols == 1:
            self.axes = [self.axes]  # Ensure self.axes is always a list
        else:
            self.axes = self.axes.flatten()

        self.default_legend_settings = {'loc': 'best', 'frameon': False, 'fontsize': 14, 'ncols': 2}
        self.default_plot_settings = {'linewidth': 4.0, 'markersize': 8.0, 'solid_capstyle': 'round'}

        # Set default tick parameters for all axes
        for ax in self.axes:
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', length=0)

    def _check_subplot_index(self, subplot_index):
        if subplot_index >= len(self.axes):
            raise IndexError(f"subplot_index {subplot_index} out of range for {len(self.axes)} subplots.")

    def plot(self, x, y, subplot_index=0, plot_type='plot', **kwargs):
        self._check_subplot_index(subplot_index)
        plot_function = getattr(self.axes[subplot_index], plot_type, None)
        if plot_function is None:
            raise ValueError(f"Plot type {plot_type} is not supported.")

        plot_settings = {**self.default_plot_settings, **kwargs}
        plot_function(x, y, **plot_settings)

    def set_title(self, title, subplot_index=0):
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].set_title(title)

    def set_suptitle(self, suptitle, **kwargs):
        self.fig.suptitle(suptitle, **kwargs)

    def set_xlabel(self, xlabel, subplot_index=0):
        if subplot_index is None:
            for ax in self.axes:
                ax.set_xlabel(xlabel, fontsize=22)
        else:
            self._check_subplot_index(subplot_index)
            self.axes[subplot_index].set_xlabel(xlabel, fontsize=22)
    
    def set_ylabel(self, ylabel, subplot_index=0):
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].set_ylabel(ylabel, fontsize=22)

    def set_xlim(self, xlim, subplot_index=0):
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].set_xlim(xlim)
    
    def set_ylim(self, ylim, subplot_index=0):
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].set_ylim(ylim)

    def set_legend(self, subplot_index=0, **kwargs):
        self._check_subplot_index(subplot_index)
        legend_settings = {**self.default_legend_settings, **kwargs}
        self.axes[subplot_index].legend(**legend_settings)

    def save(self, filename):
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(filename, dpi=400, bbox_inches='tight')
        plt.close(self.fig)


def getColor(problemType, i):
    colors = {
        'SPP': [
            'mistyrose',
            'lightsalmon',
            'lightcoral',
            'indianred',
            'firebrick',
            'brown',
            'maroon',
            'lightgray',
            'darkgray',
            'gray',
            'dimgray',
        ],
        'embeddedDAE': ['royalblue'],
        'constrainedDAE': ['black'],
    }
    return colors[problemType][i]


def getLabel(problemType):
    if problemType == 'embeddedDAE':
        return r" - $\mathtt{SDC-E}$"
    elif problemType == 'constrainedDAE':
        return r" - $\mathtt{SDC-C}$"
    elif problemType == 'SPP':
        return ""


def getMarker(problemType):
    return 'o' if problemType == 'embeddedDAE' else None
