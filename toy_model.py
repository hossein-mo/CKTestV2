"""
Toy model implementations for testing Chapman-Kolmogorov test functionality.

This module provides various time series generators including ARFIMA processes
and coupled time series models.
"""

import numpy as np
from scipy.special import gamma
from cktest_misc import misc
from typing import Tuple, Union, Optional
from numpy.typing import ArrayLike, NDArray

if misc.is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ToyModel:
    """Class providing static methods for generating test time series data."""

    @staticmethod
    def _an_apprx(d: float, n: int) -> float:
        """
        Approximate ARFIMA coefficients for large n.
        
        Parameters
        ----------
        d : float
            Differencing parameter
        n : int
            Number of coefficients
            
        Returns
        -------
        float
            Approximated coefficient value
        """
        return d * (n - d - 1)**(-d - 0.5) / (np.sqrt(n) * gamma(1 - d))

    @staticmethod
    def _an(d: float, n: int) -> NDArray:
        """
        Calculate ARFIMA coefficients.
        
        Parameters
        ----------
        d : float
            Differencing parameter
        n : int
            Number of coefficients
            
        Returns
        -------
        ndarray
            Array of coefficients
        """
        na = np.arange(1, n + 1, 1)
        if n < 171:
            return d * gamma(na - d) / (gamma(1 - d) * gamma(na + 1))
        else:
            na = np.arange(1, n + 1, 1)
            an = d * gamma(na[:170] - d) / (gamma(1 - d) * gamma(na[:170] + 1))
            an = np.concatenate((an, ToyModel._an_apprx(d, na[170:])))
            return np.flip(an).reshape(n, 1)

    @staticmethod
    def arfima(time: int,
              samples: int,
              n: int,
              d: float = 0.3,
              progress_bar: bool = False) -> NDArray:
        """
        Generate ARFIMA process time series.
        
        Parameters
        ----------
        time : int
            Length of time series
        samples : int
            Number of sample series to generate
        n : int
            Number of coefficients
        d : float, default=0.3
            Differencing parameter
        progress_bar : bool, default=False
            Whether to show progress bar
            
        Returns
        -------
        ndarray
            Generated time series data
        """
        an = ToyModel._an(d, n)
        x = ToyModel._single_generator(time, samples, an, n, progress_bar)
        if samples == 1:
            return x.reshape(-1)
        else:
            return x

    @staticmethod
    def single(time: int,
              samples: int,
              lm: int,
              an: ArrayLike,
              progress_bar: bool = False) -> NDArray:
        """
        Generate single time series with given coefficients.
        
        Parameters
        ----------
        time : int
            Length of time series
        samples : int
            Number of sample series
        lm : int
            Memory length
        an : array_like
            Coefficients array
        progress_bar : bool, default=False
            Whether to show progress bar
            
        Returns
        -------
        ndarray
            Generated time series data
        """
        x = ToyModel._single_generator(time, samples, an, lm, progress_bar)
        if samples == 1:
            return x.reshape(-1)
        else:
            return x

    @staticmethod
    def coupled_arfima(time: int,
                      samples: int,
                      n: int,
                      dx: float = 0.3,
                      dy: float = 0.4,
                      w: float = 0.5,
                      progress_bar: bool = False) -> Tuple[NDArray, NDArray]:
        """
        Generate coupled ARFIMA processes.
        
        Parameters
        ----------
        time : int
            Length of time series
        samples : int
            Number of sample series
        n : int
            Number of coefficients
        dx, dy : float, default=0.3, 0.4
            Differencing parameters
        w : float, default=0.5
            Coupling strength
        progress_bar : bool, default=False
            Whether to show progress bar
            
        Returns
        -------
        tuple of ndarray
            Pair of generated time series (x, y)
        """
        anx_ = ToyModel._an(dx, n)
        any_ = ToyModel._an(dy, n)
        return ToyModel._coupled_generator(
            time, samples,
            (w * anx_, (1 - w) * any_, (1 - w) * anx_, w * any_),
            (n, n, n, n), progress_bar)

    @staticmethod
    def coupled(time: int,
               samples: int,
               lm: Tuple[int, ...],
               coeff: Tuple[ArrayLike, ...],
               progress_bar: bool) -> Tuple[NDArray, NDArray]:
        """
        Generate coupled time series with given coefficients.
        
        Parameters
        ----------
        time : int
            Length of time series
        samples : int
            Number of sample series
        lm : tuple of int
            Memory lengths
        coeff : tuple of array_like
            Coefficients for coupling terms
        progress_bar : bool
            Whether to show progress bar
            
        Returns
        -------
        tuple of ndarray
            Pair of generated time series (x, y)
        """
        return ToyModel._coupled_generator(time, samples, coeff, lm, progress_bar)

    @staticmethod
    def _single_generator(time: int,
                        samples: int,
                        an: ArrayLike,
                        n: int,
                        progress_bar: bool) -> NDArray:
        """Generate single time series with given parameters."""
        x = np.zeros((time + n, samples))
        x[:n] = np.random.normal(size=(n, samples))
        
        # Reshape an to match the required dimensions
        an = np.asarray(an).reshape(-1, 1)  # Make an a column vector
        
        for i in tqdm(range(n, time + n), disable=not progress_bar):
            history = x[i - n:i].reshape(n, samples)  # Ensure correct shape
            x[i] = np.random.normal(size=samples) + np.sum(an * history, axis=0)
        return x[n:]

    @staticmethod
    def _coupled_generator(time: int,
                         samples: int,
                         coeff: Tuple[ArrayLike, ...],
                         n: Tuple[int, ...],
                         progress_bar: bool) -> Tuple[NDArray, NDArray]:
        """Generate coupled time series with given parameters."""
        l = np.max(n)
        an, bn, cn, dn = [np.asarray(c).reshape(-1, 1) for c in coeff]  # Reshape all coefficients
        
        x = np.zeros((time + l, samples))
        y = np.zeros((time + l, samples))
        x[:l] = np.random.normal(size=(l, samples))
        y[:l] = np.random.normal(size=(l, samples))
        
        for i in tqdm(range(l, time + l), disable=not progress_bar):
            # Reshape history windows
            x_hist0 = x[i - n[0]:i].reshape(n[0], samples)
            y_hist1 = y[i - n[1]:i].reshape(n[1], samples)
            x_hist2 = x[i - n[2]:i].reshape(n[2], samples)
            y_hist3 = y[i - n[3]:i].reshape(n[3], samples)
            
            # Calculate next values
            x[i] = (np.random.normal(size=samples) + 
                   np.sum(an * x_hist0, axis=0) + 
                   np.sum(bn * y_hist1, axis=0))
            
            y[i] = (np.random.normal(size=samples) + 
                   np.sum(cn * x_hist2, axis=0) + 
                   np.sum(dn * y_hist3, axis=0))
            
        return x[l:], y[l:]
