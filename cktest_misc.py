"""
Miscellaneous utility functions for the Chapman-Kolmogorov test implementation.

This module provides helper functions for plotting and environment detection.
"""

import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike, NDArray


class misc:
    """Utility class containing static methods for CK test implementation."""

    @staticmethod
    def is_notebook() -> bool:
        """
        Check if code is running in a Jupyter notebook environment.
        
        Returns
        -------
        bool
            True if running in notebook, False otherwise
        """
        notebooks = ("<class 'google.colab._shell.Shell'>",
                    "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>")
        try:
            shell_type = str(type(get_ipython()))
            if shell_type in notebooks:
                return True
            else:
                return False
        except:
            return False

    @staticmethod
    def set_torch_device() -> torch.device:
        """
        Set up the torch device (CPU/GPU) based on availability.
        
        Returns
        -------
        torch.device
            Selected computation device
        """
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return device

    @staticmethod
    def sample_plot(figsize: Tuple[int, int],
                   dpi: int,
                   ck: ArrayLike,
                   lm: Tuple[ArrayLike, ArrayLike],
                   cbaru: float,
                   cbard: float,
                   file: Optional[str] = None,
                   colormap: str = 'inferno',
                   s: int = 50,
                   label: Optional[str] = None,
                   label_no: Union[str, int] = '',
                   show: bool = True) -> None:
        """
        Create a heatmap plot for sample CK test results.
        
        Parameters
        ----------
        figsize : tuple of int
            Figure dimensions (width, height)
        dpi : int
            Dots per inch for the plot
        ck : array_like
            CK test results matrix
        lm : tuple of array_like
            Markov length coordinates (x, y)
        cbaru : float
            Upper colorbar limit
        cbard : float
            Lower colorbar limit
        file : str, optional
            Output file path
        colormap : str, default='inferno'
            Matplotlib colormap name
        s : int, default=50
            Scatter plot point size
        label : str, optional
            Plot label
        label_no : str or int, default=''
            Label number for multiple plots
        show : bool, default=True
            Whether to display the plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        im = ax.imshow(ck, cmap=colormap, vmin=cbard, vmax=cbaru)
        ax.set_ylabel(r'$\longleftarrow$ $t_j$')
        ax.set_xlabel(r'$t_i$ $\longrightarrow$')
        ax.xaxis.set_label_position('top')
        ax.set_xticks(np.arange(0, ck.shape[1], int(ck.shape[1] / 10)))
        ax.set_yticks(np.arange(0, ck.shape[1], int(ck.shape[1] / 10)))
        ax.xaxis.tick_top()
        ax.scatter(x=lm[0], y=lm[1], s=s, c='w')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if label != None:
            clabel = r"$S_{0}\left(t_i, t_j\right)$".format(label)
        else:
            clabel = r"$S\left(t_i, t_j\right)$"
        cbar.set_label(clabel)
        if file != None:
            fig.savefig("{0}_{1}.pdf".format(file, str(label_no)),
                        format='pdf',
                        bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def single_plot(ck: ArrayLike,
                   ck_error: ArrayLike,
                   lm: int,
                   figsize: Tuple[int, int] = (12, 6),
                   dpi: int = 80,
                   style: str = '-',
                   file: Optional[str] = None,
                   grid: bool = True,
                   linewidth: float = 0.3,
                   s: int = 50,
                   label: str = ' ',
                   label_no: Union[str, int] = '',
                   show: bool = True) -> None:
        """
        Create a line plot for single CK test results.
        
        Parameters
        ----------
        ck : array_like
            CK test results
        ck_error : array_like
            CK test errors
        lm : int
            Markov length
        figsize : tuple of int, default=(12, 6)
            Figure dimensions
        dpi : int, default=80
            Dots per inch
        style : str, default='-'
            Line style
        file : str, optional
            Output file path
        grid : bool, default=True
            Whether to show grid
        linewidth : float, default=0.3
            Width of plot lines
        s : int, default=50
            Scatter plot point size
        label : str, default=' '
            Plot label
        label_no : str or int, default=''
            Label number for multiple plots
        show : bool, default=True
            Whether to display the plot
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.grid(grid, linestyle=":")
        plt.plot(range(2,
                       len(ck) + 2),
                 ck,
                 style,
                 c='k',
                 alpha=0.7,
                 linewidth=linewidth,
                 label=r'$S_{0}(\tau)$'.format(label))
        plt.plot(range(2, len(ck) + 2), ck_error, c='r', label=r'$\sigma_S$', linewidth=linewidth)
        plt.scatter(lm + 1, ck[lm - 1], zorder=10, label=r'$S(\tau)=0 \Rightarrow l_m = {0}$'.format(lm), s=s, alpha=0.7)
        plt.ylabel(r'$S_{0}(\tau)$'.format(label))
        plt.xlabel(r'$\tau$')
        
        # Fix for tick calculation
        step = max(1, int(len(ck)/10))  # Ensure step is at least 1
        if len(ck) > 10:
            ticks = np.round(range(2, len(ck) + 2, step), -len(str(len(ck)))+2)
            ticks[0] = 2
        else:
            ticks = range(2, len(ck) + 2)  # Use all points for small arrays
        plt.xticks(ticks)
        
        plt.legend()
        if file is not None:
            plt.savefig("{0}_{1}.pdf".format(file, str(label_no)),
                        format='pdf',
                        bbox_inches='tight')
        if show:
            plt.show()
