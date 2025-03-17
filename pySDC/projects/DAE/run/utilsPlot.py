import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from pySDC.projects.DAE.run.utils import SDC_METHODS, RK_METHODS, COLLOCATION_METHODS


class Plotter:
    def __init__(self, nrows=1, ncols=1, orientation=None, figsize=None, hspace=0.4, wspace=0.4, layout=None, width_ratios=None, height_ratios=None):
        self.nrows = nrows
        self.ncols = ncols
        self.orientation = orientation
        self.figsize = figsize
        self.hspace = hspace
        self.wspace = wspace
        self.layout = layout
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios

        if self.orientation == 'horizontal':
            self.nrows = 1
            self.ncols = self.nrows * self.ncols
        elif self.orientation == 'vertical':
            self.ncols = 1
            self.nrows = self.nrows * self.ncols
        elif self.orientation is not None:
            raise ValueError("Orientation must be either 'horizontal', 'vertical', or None")

        # Pre-defined figure size
        if self.figsize is None:
            length_subplot = 6
            if self.nrows * self.ncols == 1:
                self.figsize = (length_subplot, length_subplot)
            elif self.nrows == 1 and self.ncols > 1:
                self.figsize = (self.ncols * length_subplot, 8)
            else:
                self.figsize = (self.ncols * length_subplot, self.nrows * length_subplot)

        # Define label size for ticks based on layout
        self._set_labelsize()

        if layout:
            self.fig = plt.figure(figsize=self.figsize)

            # Use gridspec for custom layouts
            max_row = max(row[1] for row in layout)
            max_col = max(col[1] for col in layout)
            self.gs = gridspec.GridSpec(max_row + 1, max_col + 1, 
                                        width_ratios=width_ratios if width_ratios else [1] * (max_col + 1),
                                        height_ratios=height_ratios if height_ratios else [1] * (max_row + 1),
                                        wspace=wspace, hspace=hspace)
            self.axes = [self.fig.add_subplot(self.gs[row_start:row_end, col_start:col_end]) 
                         for row_start, row_end, col_start, col_end in layout]
        else:
            # Standard nrows x ncols layout
            # self.gs = None
            self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

            if self.nrows * self.ncols == 1:
                self.axes = [self.axes]
            else:
                self.axes = self.axes.flatten()

        # This will hold the secondary y-axes if any are created with twinx
        self.secondary_axes = [None] * len(self.axes)

        # Set default tick parameters for all axes
        for ax in self.axes:
            ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
            ax.tick_params(axis='both', which='minor', length=0)

        self.default_legend_settings = {'loc': 'best', 'frameon': True, 'framealpha': 0.5, 'fontsize': 20, 'ncols': 2}
        self.default_plot_settings = {'markeredgewidth': 1.2, 'markeredgecolor': 'black', 'linewidth': 3.0, 'solid_capstyle': 'round'}
        self.default_scatter_settings = {'s': 80, 'alpha': 0.75}

    def _set_labelsize(self):
        """Set default label sizes based on nrows and ncols."""
        if self.nrows * self.ncols == 1:
            self.labelsize = 14
        elif self.nrows == 2 and self.ncols == 2:
            self.labelsize = 16
        elif self.nrows == 1 and self.ncols == 2:
            self.labelsize = 16
        elif self.nrows == 3 and self.ncols == 1:
            self.labelsize = 16
        elif self.nrows == 3 and self.ncols == 2:
            self.labelsize = 18
        elif self.nrows == 4 and self.ncols == 1:
            self.labelsize = 22
        elif self.nrows == 4 and self.ncols == 2:
            self.labelsize = 24
        else:
            self.labelsize = 18

    def _check_subplot_index(self, subplot_index):
        if subplot_index >= len(self.axes):
            raise IndexError(f"subplot_index {subplot_index} out of range for {len(self.axes)} subplots.")

    def barh(self, y, width, subplot_index=0, secondary=False, **kwargs):
        """
        Creates a horizontal bar plot using plt.barh.

        Parameters:
        - y (array-like): Y-axis positions (categorical or numerical).
        - width (array-like): Bar lengths (widths).
        - subplot_index (int): Subplot index.
        - secondary (bool): If True, applies to secondary y-axis.
        - **kwargs: Additional `barh` arguments (e.g., `color`, `height`, `edgecolor`).
        """
        self._check_subplot_index(subplot_index)
        ax = self.secondary_axes[subplot_index] if secondary and self.secondary_axes[subplot_index] else self.axes[subplot_index]

        barh_settings = {'color': 'blue', 'height': 0.4, 'edgecolor': 'black'}  # Default settings
        barh_settings.update(kwargs)  # Override with user input

        ax.barh(y, width, **barh_settings)

    def contour(self, X, Y, Z, subplot_index=0, secondary=False, **kwargs):
        """
        Create a contour plot.

        Parameters:
        - X, Y: Meshgrid arrays for contouring.
        - Z: 2D array representing function values.
        - subplot_index (int): Subplot index.
        - secondary (bool): If True, use the secondary y-axis.
        - **kwargs: Additional matplotlib contour arguments (e.g., `levels`, `cmap`).
        """
        self._check_subplot_index(subplot_index)
        ax = self.secondary_axes[subplot_index] if secondary and self.secondary_axes[subplot_index] else self.axes[subplot_index]

        cs = ax.contour(X, Y, Z, **kwargs)
        return cs  # Return contour set for further customization

    def contourf(self, X, Y, Z, subplot_index=0, secondary=False, **kwargs):
        """
        Create a filled contour plot.

        Parameters:
        - X, Y: Meshgrid arrays for contouring.
        - Z: 2D array representing function values.
        - subplot_index (int): Subplot index.
        - secondary (bool): If True, use the secondary y-axis.
        - **kwargs: Additional matplotlib contourf arguments (e.g., `levels`, `cmap`).
        """
        self._check_subplot_index(subplot_index)
        ax = self.secondary_axes[subplot_index] if secondary and self.secondary_axes[subplot_index] else self.axes[subplot_index]

        cf = ax.contourf(X, Y, Z, **kwargs)
        return cf  # Return contour set for further customization

    def fill_between(self, x, y1, y2, subplot_index=0, secondary=False, **kwargs):
        """
        Creates a filled region between y1 and y2 along the x-axis.

        Parameters:
        - x (array-like): X values.
        - y1 (array-like): Lower boundary of the fill region.
        - y2 (array-like): Upper boundary of the fill region.
        - subplot_index (int): Subplot index.
        - secondary (bool): If True, applies to secondary y-axis.
        - **kwargs: Additional `fill_between` arguments (e.g., `color`, `alpha`, `hatch`).
        """
        self._check_subplot_index(subplot_index)
        ax = self.secondary_axes[subplot_index] if secondary and self.secondary_axes[subplot_index] else self.axes[subplot_index]

        ax.fill_between(x, y1, y2, **kwargs)

    def plot(self, x, y, subplot_index=0, plot_type='plot', secondary=False, **kwargs):
        self._check_subplot_index(subplot_index)

        if secondary and self.secondary_axes[subplot_index] is not None:
            ax = self.secondary_axes[subplot_index]
        else:
            ax = self.axes[subplot_index]

        plot_function = getattr(ax, plot_type, None)
        if plot_function is None:
            raise ValueError(f"Plot type {plot_type} is not supported.")

        plot_settings = {**self.default_plot_settings, **kwargs}
        plot_function(x, y, **plot_settings)

    def scatter(self, x, y, subplot_index=0, secondary=False, **kwargs):
        """Scatter plot for the specified subplot."""
        self._check_subplot_index(subplot_index)
        # Determine which axis to use
        if secondary:
            # Create secondary y-axis if it doesn't exist
            if self.secondary_axes[subplot_index] is None:
                self.secondary_axes[subplot_index] = self.axes[subplot_index].twinx()
            ax = self.secondary_axes[subplot_index]
        else:
            ax = self.axes[subplot_index]
        
        scatter_settings = {**self.default_scatter_settings, **kwargs}
        ax.scatter(x, y, **scatter_settings)


    def matshow(self, matrix, subplot_index=0, cmap="viridis", colorbar=True, shared_colorbar=False, 
            vmin=None, vmax=None, xticks_bottom=True, **kwargs):
        """
        Plot a matrix using matshow on the specified subplot with optional fixed colorbar range.

        Parameters:
        - matrix (2D array-like): The matrix to visualize.
        - subplot_index (int): The index of the subplot where the matrix is plotted.
        - cmap (str): The colormap to use (default: "viridis").
        - colorbar (bool): Whether to add a colorbar (ignored if shared_colorbar=True).
        - shared_colorbar (bool): Whether to plot a single colorbar for all subplots.
        - vmin (float): Minimum value for the colorbar.
        - vmax (float): Maximum value for the colorbar.
        - xticks_bottom (bool): Whether to move x-axis ticks to the bottom.
        - kwargs: Additional arguments to pass to matshow.
        """
        self._check_subplot_index(subplot_index)
        
        # Plot the matrix on the specified subplot with fixed color range
        ax = self.axes[subplot_index]
        cax = ax.matshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        
        # Adjust x-axis tick position
        if xticks_bottom:
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_label_position('bottom')
        
        # Add a colorbar for a single subplot
        if colorbar and not shared_colorbar:
            self.fig.colorbar(cax, ax=ax)
        
        # Store the colorbar information if shared_colorbar is True
        if shared_colorbar:
            if not hasattr(self, "_shared_colorbar"):
                self._shared_colorbar = cax
                self._colorbar_axes = ax


    def set_title(self, title, subplot_index=0, pad=20, **kwargs):
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].set_title(title, pad=pad, **kwargs)

    def set_suptitle(self, suptitle, **kwargs):
        self.fig.suptitle(suptitle, **kwargs)

    def set_xlabel(self, xlabel="", subplot_index=None, remove=False, fontsize=22):
        """
        Set the x-label for a specific subplot or all subplots. Optionally remove x-axis labels.

        Parameters:
        - xlabel (str): The label text to set. Defaults to an empty string.
        - subplot_index (int or None): The index of the subplot. If None, applies to all subplots.
        - remove (bool): If True, removes the x-axis labels by hiding them.
        - fontsize (int): Fontsize for label.
        """
        if subplot_index is None:
            for ax in self.axes:
                if remove:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(xlabel, fontsize=fontsize)
        else:
            self._check_subplot_index(subplot_index)
            if remove:
                self.axes[subplot_index].set_xticklabels([])
            else:
                self.axes[subplot_index].set_xlabel(xlabel, fontsize=fontsize)


    def set_ylabel(self, ylabel, subplot_index=None, secondary=False, fontsize=22):
        """Set the y-label for a specific subplot or all subplots if subplot_index is None."""
        if subplot_index is None:
            for ax, sec_ax in zip(self.axes, self.secondary_axes):
                if secondary and sec_ax is not None:
                    sec_ax.set_ylabel(ylabel, fontsize=fontsize)
                else:
                    ax.set_ylabel(ylabel, fontsize=fontsize)
        else:
            self._check_subplot_index(subplot_index)
            if secondary and self.secondary_axes[subplot_index] is not None:
                self.secondary_axes[subplot_index].set_ylabel(ylabel, fontsize=fontsize)
            else:
                self.axes[subplot_index].set_ylabel(ylabel, fontsize=fontsize)

    def set_xlim(self, xlim, subplot_index=None, scale='linear', base=10):
        if subplot_index is None:
            for ax in self.axes:
                ax.set_xlim(xlim)
                if scale == 'log':
                    ax.set_xscale(scale, base=base)
                else:
                    ax.set_xscale(scale)
        else:
            self._check_subplot_index(subplot_index)
            self.axes[subplot_index].set_xlim(xlim)
            if scale == 'log':
                self.axes[subplot_index].set_xscale(scale, base=base)
            else:
                self.axes[subplot_index].set_xscale(scale)


    def set_ylim(self, ylim, subplot_index=None, scale='linear', base=10, secondary=False):
        if subplot_index is None:
            for ax, sec_ax in zip(self.axes, self.secondary_axes):
                if secondary and sec_ax is not None:
                    sec_ax.set_ylim(ylim)
                    if scale == 'log':
                        sec_ax.set_yscale(scale, base=base)
                    else:
                        sec_ax.set_yscale(scale)
                else:
                    ax.set_ylim(ylim)
                    if scale == 'log':
                        ax.set_yscale(scale, base=base)
                    else:
                        ax.set_yscale(scale)
        else:
            self._check_subplot_index(subplot_index)
            if secondary and self.secondary_axes[subplot_index] is not None:
                self.secondary_axes[subplot_index].set_ylim(ylim)
                if scale == 'log':
                    self.secondary_axes[subplot_index].set_yscale(scale, base=base)
                else:
                    self.secondary_axes[subplot_index].set_yscale(scale)
            else:
                self.axes[subplot_index].set_ylim(ylim)
                if scale == 'log':
                    self.axes[subplot_index].set_yscale(scale, base=base)
                else:
                    self.axes[subplot_index].set_yscale(scale)

    def sync_xlim(self, min_x_set=1e-5):
        """
        Synchronize x-axis limits across all subplots by finding the global min/max.
        Handles both linear and log scales correctly.
        """
        min_x, max_x = None, None

        # Find global min/max x-limits across all axes
        for ax in self.axes:
            x_limits = ax.get_xlim()

            # Ignore non-positive values for log scale
            if ax.get_xscale() == "log":
                x_limits = [x for x in x_limits if x > 0]
                if not x_limits:
                    continue  # Skip if there are no valid positive values

            if min_x is None or x_limits[0] < min_x:
                min_x = x_limits[0]
            if max_x is None or x_limits[1] > max_x:
                max_x = x_limits[1]

        # Apply the same limits to all subplots
        for ax in self.axes:
            if ax.get_xscale() == "log":
                if min_x is not None and min_x <= 0:
                    min_x = min_x_set  # Set a small positive value to avoid log error
                ax.set_xlim(min_x, max_x)
            else:
                ax.set_xlim(min_x, max_x)

    def sync_ylim(self, min_y_set=1e-5):
        """
        Synchronize y-axis limits across all subplots by finding the global min/max.
        Handles both linear and log scales correctly.
        """
        min_y, max_y = None, None

        # Find global min/max y-limits across all axes
        for ax in self.axes:
            y_limits = ax.get_ylim()
            
            # Ignore non-positive values for log scale
            if ax.get_yscale() == "log":
                y_limits = [y for y in y_limits if y > 0]  
                if not y_limits:
                    continue  # Skip if there are no valid positive values
            
            if min_y is None or y_limits[0] < min_y:
                min_y = y_limits[0]
            if max_y is None or y_limits[1] > max_y:
                max_y = y_limits[1]

        # Apply the same limits to all subplots
        for ax in self.axes:
            if ax.get_yscale() == "log":
                if min_y is not None and min_y <= 0:
                    min_y = min_y_set # Set a small positive value to avoid log error
                ax.set_ylim(min_y, max_y)
            else:
                ax.set_ylim(min_y, max_y)

    def set_tick_params(self, subplot_index=None, axis='both', which='major', labelsize=None, **kwargs):
        """
        Update tick parameters for all subplots or a specific subplot.

        Parameters:
        - subplot_index (int or None): If None, applies to all subplots. Otherwise, applies to a specific subplot.
        - axis (str): 'x', 'y', or 'both'. Default is 'both'.
        - which (str): 'major' or 'minor' ticks. Default is 'major'.
        - labelsize (int or None): Size of tick labels.
        - **kwargs: Additional arguments for tick_params (e.g., length, width).
        """
        axes = self.axes if subplot_index is None else [self.axes[subplot_index]]
        
        for ax in axes:
            ax.tick_params(axis=axis, which=which, labelsize=labelsize, **kwargs)


    def set_xticks(self, xticks_list=None, labels=None, subplot_index=None, clear_labels=False, fontsize=20):
        """
        Set custom xticks and labels or optionally remove only the labels while keeping the ticks.

        Parameters:
        - xticks_list (list or None): The positions for the xticks.
        - labels (list or None): The labels for the xticks. If None, labels are derived from xticks_list.
        - subplot_index (int or None): The index of the subplot. If None, applies to all subplots.
        - clear_labels (bool): If True, removes the labels but keeps the xticks.
        """
        if subplot_index is None:
            for ax in self.axes:
                if clear_labels:
                    ax.set_xticklabels([], fontsize=fontsize)  # Remove only the labels
                elif xticks_list is not None:
                    ax.set_xticks(xticks_list)
                    if labels is not None:
                        ax.set_xticklabels(labels, fontsize=fontsize)
                    else:
                        ax.set_xticklabels([str(x) for x in xticks_list], fontsize=fontsize)
        else:
            self._check_subplot_index(subplot_index)
            ax = self.axes[subplot_index]
            if clear_labels:
                ax.set_xticklabels([], fontsize=fontsize)  # Remove only the labels
            elif xticks_list is not None:
                ax.set_xticks(xticks_list)
                if labels is not None:
                    ax.set_xticklabels(labels, fontsize=fontsize)
                else:
                    ax.set_xticklabels([str(x) for x in xticks_list], fontsize=fontsize)

    def set_yticks(self, yticks_list, labels=None, subplot_index=None, fontsize=20):
        """
        Set custom y-ticks and labels for a specific subplot or all subplots.

        Parameters:
        - yticks_list (list): The positions for the y-ticks.
        - labels (list or None): The labels for the y-ticks. If None, labels are derived from yticks_list.
        - subplot_index (int or None): The index of the subplot. If None, applies to all subplots.
        """
        if subplot_index is None:
            # Apply to all subplots
            for ax in self.axes:
                ax.set_yticks(yticks_list)
                if labels is not None:
                    ax.set_yticklabels(labels, fontsize=fontsize)
                else:
                    ax.set_yticklabels([str(y) for y in yticks_list], fontsize=fontsize)
        else:
            # Apply to a specific subplot
            self._check_subplot_index(subplot_index)
            ax = self.axes[subplot_index]
            ax.set_yticks(yticks_list)
            if labels is not None:
                ax.set_yticklabels(labels, fontsize=fontsize)
            else:
                ax.set_yticklabels([str(y) for y in yticks_list], fontsize=fontsize)

    def set_legend(self, subplot_index=None, secondary=False, framealpha=0.0, **kwargs):
        if subplot_index is None:
            for ax, sec_ax in zip(self.axes, self.secondary_axes):
                if secondary and sec_ax is not None:
                    sec_ax.legend(**{**self.default_legend_settings, **kwargs})
                else:
                    ax.legend(**{**self.default_legend_settings, **kwargs})
        else:
            self._check_subplot_index(subplot_index)
            if secondary and self.secondary_axes[subplot_index] is not None:
                self.secondary_axes[subplot_index].legend(**{**self.default_legend_settings, **kwargs})
            else:
                self.axes[subplot_index].legend(**{**self.default_legend_settings, **kwargs})

    def set_shared_legend(self, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=1, **kwargs):
        """Set a single shared legend for all subplots."""
        handles, labels = [], []
        for ax in self.axes:
            h, l = ax.get_legend_handles_labels()
            
            # If `ax` is a `brokenaxes`, h and l may be nested lists/tuples. Flatten them.
            if isinstance(h, (list, tuple)) and any(isinstance(i, (list, tuple)) for i in h):
                h = [item for sublist in h for item in sublist]  # Flatten nested lists/tuples
            if isinstance(l, (list, tuple)) and any(isinstance(i, (list, tuple)) for i in l):
                l = [item for sublist in l for item in sublist]  # Flatten nested lists/tuples

            handles.extend(h)
            labels.extend(l)

        # Remove duplicates (optional, in case labels repeat)
        by_label = dict(zip(labels, handles))
        
        self.fig.legend(
            by_label.values(),
            by_label.keys(),
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            **kwargs
        )

    def set_group_shared_legend(self, group_indices, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=1, **kwargs):
        """
        Create a shared legend for a specific group of subplots.
        
        Parameters:
        - group_indices (list): Indices of subplots to include in the shared legend.
        - loc (str): Location of the legend (default: 'lower center').
        - bbox_to_anchor (tuple): Anchor position of the legend.
        - ncol (int): Number of columns in the legend.
        - kwargs: Additional keyword arguments for customization.
        """
        handles, labels = [], []
        for idx in group_indices:
            self._check_subplot_index(idx)
            h, l = self.axes[idx].get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        
        # Remove duplicates (optional, in case labels repeat)
        by_label = dict(zip(labels, handles))
        
        self.fig.legend(
            by_label.values(),
            by_label.keys(),
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            **kwargs
        )

    def add_shared_colorbar(self, **kwargs):
        """
        Add a single shared colorbar for the entire figure.

        Parameters:
        - kwargs: Additional arguments to pass to fig.colorbar.
        """
        if hasattr(self, "_shared_colorbar"):
            self.fig.colorbar(self._shared_colorbar, ax=self.axes, location="right", **kwargs)


    def set_grid(self, grid_on=True, subplot_index=None, secondary=False):
        if subplot_index is None:
            for ax in self.axes:
                ax.grid(grid_on)
        else:
            self._check_subplot_index(subplot_index)
            if secondary and self.secondary_axes[subplot_index] is not None:
                self.secondary_axes[subplot_index].grid(grid_on)
            else:
                self.axes[subplot_index].grid(grid_on)

    def set_xscale(self, scale="linear", subplot_index=0, base=None):
        """
        Set the x-axis scale for a specific subplot.

        Parameters:
        - scale (str): The scale type ('linear', 'log', 'symlog', 'logit').
        - subplot_index (int): The index of the subplot.
        - base (float, optional): The base of the logarithm (only for 'log' scale).
        """
        self._check_subplot_index(subplot_index)
        ax = self.axes[subplot_index]

        # Apply scale (base is only valid for log scale)
        if scale == "log":
            if base:
                ax.set_xscale(scale, base=base)
            else:
                ax.set_xscale(scale)
        else:
            ax.set_xscale(scale)

    def set_yscale(self, scale="linear", subplot_index=0, base=None, secondary=False):
        """
        Set the y-axis scale for a specific subplot.

        Parameters:
        - scale (str): The scale type ('linear', 'log', 'symlog', 'logit').
        - subplot_index (int): The index of the subplot.
        - base (float, optional): The base of the logarithm (only for 'log' scale).
        - secondary (bool): Whether to apply the scale to the secondary y-axis.
        """
        self._check_subplot_index(subplot_index)

        # Choose the appropriate axis (primary or secondary)
        if secondary:
            if self.secondary_axes[subplot_index] is None:
                raise ValueError(f"No secondary axis exists for subplot_index {subplot_index}. Create it using scatter().")
            ax = self.secondary_axes[subplot_index]
        else:
            ax = self.axes[subplot_index]

        # Apply scale (base is only valid for log scale)
        if scale == "log":
            if base:
                ax.set_yscale(scale, base=base)
            else:
                ax.set_yscale(scale)
        else:
            ax.set_yscale(scale)

    def twinx(self, subplot_index=0):
        self._check_subplot_index(subplot_index)
        self.secondary_axes[subplot_index] = self.axes[subplot_index].twinx()
        return self.secondary_axes[subplot_index]

    def add_vline(self, x, subplot_index=0, **kwargs):
        """Add a vertical line at x on the specified subplot."""
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].axvline(x, **kwargs)

    def add_hline(self, y, subplot_index=0, **kwargs):
        """Add a horizontal line at y on the specified subplot."""
        self._check_subplot_index(subplot_index)
        self.axes[subplot_index].axhline(y, **kwargs)

    def set_aspect(self, aspect='equal', subplot_index=None):
        """Set aspect ratio for all or a specific subplot."""
        if subplot_index is None:
            for ax in self.axes:
                ax.set_aspect(aspect)
        else:
            self._check_subplot_index(subplot_index)
            self.axes[subplot_index].set_aspect(aspect)

    def adjust_layout(self, num_subplots):
        """
        Adjust the layout to fit only the required number of subplots.
        
        Parameters:
        - num_subplots (int): Number of required subplots.
        """
        nrows, ncols = self.nrows, self.ncols
        total_subplots = nrows * ncols

        if num_subplots < total_subplots:
            for i, ax in enumerate(self.axes):
                if i >= num_subplots:
                    ax.remove()  # Completely remove the extra subplots

    def save(self, filename, uniform_size=True, use_constrained_layout=False):
        # Apply layout adjustments
        if use_constrained_layout:
            self.fig.set_constrained_layout(True)
        else:
            # self.fig.set_constrained_layout(False)  # Explicitly disable if not used

            # Ensure manual spacing adjustments are always applied
            if hasattr(self, "gs"):  # If GridSpec exists, enforce spacing
                self.gs.update(hspace=self.hspace, wspace=self.wspace)  # REDUCE spacing directly
            else:
                self.fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)

            # Only apply tight_layout when gridspec is NOT used
            if uniform_size and not hasattr(self, "gs"):
                self.fig.tight_layout()

        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(filename, dpi=400, bbox_inches='tight')
        plt.close(self.fig)


def getColor(problemType, i, QI):
    if QI in SDC_METHODS:
        colors = {
            "SPP": [
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
            "SPP-IMEX": [
                'honeydew',
                'palegreen',
                'lightgreen',
                'mediumspringgreen',
                'springgreen',
                'limegreen',
                'mediumseagreen',
                'seagreen',
                'forestgreen',
                'green',
                'darkgreen',
            ],
            "SPP-yp": [
                'paleturquoise',
                'darkturquoise',
                'lightblue',
                'skyblue',
                'deepskyblue',
                'cornflowerblue',
                'dodgerblue',
                'royalblue',
                'navy',
                'midnightblue',
                'black',
            ],
            "embeddedDAE": ["royalblue"],
            "constrainedDAE": ["mediumseagreen"],
            "fullyImplicitDAE": ["mediumorchid"],
            "semiImplicitDAE": ["sandybrown"],
        }
        return colors[problemType][i]
    elif (QI in RK_METHODS or QI in COLLOCATION_METHODS):
        colors = {
            "BE": ["peru"],
            "DIRK43": ["slategrey"],
            "EDIRK4": ["black"],
            "DIRK": ["slateblue"],
            "DIRK5":  ["indianred"],
            "DIRK5_2": ["lightgreen"],
            "ESDIRK53": ["orange"],
            "SDIRK3": ["lightseagreen"],
            "RadauIIA5": ["mediumseagreen"],
            "RadauIIA7": ["royalblue"],
            "RadauIIA9": ["mediumorchid"],
        }
        return colors[QI][i]


def getLabel(problemType, eps, QI):
    if QI in SDC_METHODS:
        if problemType == "constrainedDAE":
            return r"$\mathtt{SDC-C}$"
        elif problemType == "embeddedDAE":
            return r"$\mathtt{SDC-E}$"
        elif problemType == "SPP":
            return rf'$\varepsilon=${eps}'
        elif problemType == "SPP-IMEX":
            return rf'(imex) $\varepsilon=${eps}'
        elif problemType == "SPP-yp":
            return rf'yp-$\varepsilon=${eps}'
        elif problemType == "fullyImplicitDAE":
            return r"$\mathtt{FI-SDC}$"
        elif problemType == "semiImplicitDAE":
            return r"$\mathtt{SI-SDC}$"
    elif (QI in RK_METHODS or QI in COLLOCATION_METHODS):
        return f"{QI}"
    
def get_linestyle(problem_type, QI):
    linestyles = ["solid", "dotted", "dashdot", "dashed"]
    if QI in SDC_METHODS:
        if problem_type in ["SPP", "constrainedDAE", "embeddedDAE"]:
            return linestyles[0]
        elif problem_type in ["SPP-yp", "fullyImplicitDAE", "semiImplicitDAE"]:
            return linestyles[1]
        elif problem_type == "SPP-IMEX":
            return linestyles[2]
        else:
            raise NotImplementedError(f"No linestyle implemented for {problem_type}!")
    elif (QI in RK_METHODS or QI in COLLOCATION_METHODS):
        return random.choice(linestyles)


def getMarker(problemType, i, QI):
    markersize = 13.0

    if QI in SDC_METHODS:
        marker = {
            "SPP": ["o", "8", "s", "p", "P", "X", "D", "d", "h", "H", "*"],
            "SPP-IMEX": ["o", "8", "s", "p", "P", "X", "D", "d", "h", "H", "*"],
            "SPP-yp": ["o", "8", "s", "p", "P", "X", "D", "d", "h", "H", "*"], 
            "embeddedDAE": ["^"],
            "constrainedDAE": ["v"],
            "fullyImplicitDAE": ["<"],
            "semiImplicitDAE": [">"],
        }
        return {"marker": marker[problemType][i], "markersize": markersize}
    elif (QI in RK_METHODS or QI in COLLOCATION_METHODS):
        marker = {
            "BE": ["1"],
            "DIRK43": ["2"],
            "EDIRK4": ["3"],
            "DIRK": ["4"],
            "DIRK5":  ["+"],
            "DIRK5_2": ["x"],
            "ESDIRK53": ["d"],
            "SDIRK3": ["*"],
            "RadauIIA5": ["s"],
            "RadauIIA7": ["D"],
            "RadauIIA9": ["h"],
        }
        return {"marker": marker[QI][i], "markersize": markersize}
