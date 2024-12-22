"""
A module for performing Chapman-Kolmogorov (CK) tests on time series data.

This module provides classes and methods to perform Chapman-Kolmogorov tests on both single 
and coupled time series. The CK test helps verify if a process is Markovian by checking 
if the Chapman-Kolmogorov equation is satisfied within statistical error.

Classes:
    CKTestSingle: Performs CK test on a single time series
    CKTestCoupled: Performs CK test on two coupled time series
    CKTest: Static methods for CK test calculations
    Statistics: Static methods for statistical calculations
"""

import os
import sys
from cktest_misc import misc
from typing import Tuple, Union, List, Optional
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

if misc.is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import matplotlib.pyplot as plt
from pylab import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Class for performing CK test on a single process.
class CKTestSingle():
    """
    Class for performing Chapman-Kolmogorov test on a single time series.
    
    This class handles both single sample time series and multiple sample time series.
    For single samples, the test is performed with respect to time lag τ.
    For multiple samples, the test is performed with respect to times t_i and t_j.
    """

    # The constructor function.
    def __init__(self, 
                 x: ArrayLike,
                 dx: float = 1, 
                 bins_mode: str = 'std',
                 bins: Union[int, ArrayLike] = 6,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float64) -> None:
        """
        x: Array of data points. If x is a one dimensional array CK test will perform
        with respect to time lag τ. If x is two dimensional array CK test will perform with
        respect to t_i and t_j.

        dx: Determine the bin width of probability distributions.

        bins_mode: 'std' or 'real'. If set to 'std' the width of bins will be
        standard deviation of x multiplied by dx. If set to 'real' width of bins will be
        equal to dx.

        bins: 0 , any positive integer or array like object: If equals to 0 bins include
        all data points. If bins > 0 bins range from -bins × dx to bins × dx with width
        equals to dx. If bins is an array like object it should be the list of left edges
        of the bins.

        device: 'auto' or name of torch device: If set to 'auto' and a cuda device is
        avaliable to torch it will use gpu for computations otherwise it will use cpu.

        dtype: torch data type. Data type of torch arrays, better to leave unchanged.
        """
        self.x = x.copy()

        # Probability distributions will be saved in this list.
        self.pdfs = []
        # This will be used for CK test values.
        self.ck = 0
        self.dtype = dtype
        # Markov length will be saved in this variable.
        self.lm = -1

        # If a cuda device is avaliable torch will use it for its computations.
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Checking if x includes more than one sample or not.
        if np.ndim(x) == 2 and self.x.shape[0] > 1 and self.x.shape[1] > 1:
            self.is_sample = True
        else:
            self.is_sample = False
            self.x = self.x.reshape(-1)

        # Generating left edges of the bins.
        if np.ndim(bins) == 0:
            self.xbins = Statistics.find_bins(self.x, dx, bins_mode, max_bin=bins)
        else:
            self.xbins = bins.copy()

        self.pdf_shape = (len(self.xbins), len(self.xbins))

        # Finding data points corresponding index in xbins.
        self.xdigit = Statistics.digitize(self.x, self.xbins, dtype=self.dtype, device=self.device)

    # After creating a CKTestSingle() object this function should be called for performing the test.
    def run_test(self, 
                 lag: int,
                 progress_bar: bool = True) -> Tuple[NDArray, NDArray]:
        '''
        lag: positive integer. Ck test will continue until time lag τ=0 in the case if x includes
        only one sample or t_i - t_j = 0 in the case x has more than one samples.

        progress_bar: bool, default is True. If true a tqdm progress bar shows the progress of the test.
        '''

        if self.is_sample:
            # if there is more than one sample the condition is true and this part will run.

            # number of data points
            n = self.xdigit.shape[1]
            # crating torch arrays for CK test and its error.
            self.ck = torch.zeros((lag, lag), device=self.device, dtype=self.dtype)
            self.ck_error = torch.zeros((lag, lag), device=self.device, dtype=self.dtype)
            for t3 in tqdm(range(lag)):
                for t1 in range(lag):
                    t2 = int((t3 - t1) / 2)
                    t2 += t1

                    # calculating p(t3,t1)
                    xx = torch.cat((self.xdigit[t3], self.xdigit[t1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p31 = Statistics.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # calculating p(t3,t2)
                    xx = torch.cat((self.xdigit[t3], self.xdigit[t2]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p32 = Statistics.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # calculating p(t2,t1)
                    xx = torch.cat((self.xdigit[t2], self.xdigit[t1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    p21 = Statistics.pdf(xx,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    # performing the test using single() function from CKTest class.
                    self.ck[t1, t3], self.ck_error[t1, t3] = CKTest.single(p31, p32, p21, n)
        else:
            # if there is only one sample the condition is false and this part will run.

            if lag > len(self.pdfs) + 1:
                if len(self.pdfs) == 0:
                    self.ck = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                    self.ck_error = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                    n = len(self.xdigit) - 1
                    xx = torch.cat((self.xdigit[1:], self.xdigit[:-1]), 0)
                    xx = torch.reshape(xx, (2, n))
                    self.pdfs.append(
                        Statistics.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                       device=self.device))
                else:
                    self.ck = torch.cat((self.ck,
                                         torch.zeros(lag - len(self.pdfs),
                                                     device=self.device,
                                                     dtype=self.dtype)))
                    self.ck_error = torch.cat((self.ck_error,
                                               torch.zeros(lag - len(self.pdfs),
                                                           device=self.device,
                                                           dtype=self.dtype)))

                for t3 in tqdm(range(len(self.pdfs) + 1, lag + 1), disable=not progress_bar):
                    n = len(self.xdigit) - t3
                    xx = torch.cat((self.xdigit[t3:], self.xdigit[:n]), 0)
                    xx = torch.reshape(xx, (2, n))
                    self.pdfs.append(
                        Statistics.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                       device=self.device))

                    t2 = int(t3 / 2)
                    t1 = t3 - t2
                    self.ck[t3 - 2], self.ck_error[t3 - 2] = CKTest.single(
                        self.pdfs[t3 - 1].to_dense(), self.pdfs[t2 - 1].to_dense(),
                        self.pdfs[t1 - 1].to_dense(), n)
        ck_array = np.array(self.ck.tolist())
        ck_error_array = np.array(self.ck_error.tolist())
        self.lm = Statistics.markov_length(ck_array, ck_error_array)
        return ck_array, ck_error_array

    def run_sit(self,
                lag: int, 
                progress_bar: bool = True) -> Tuple[NDArray, NDArray]:
        """
        Performs a simplified independence test variant of the CK test.
        
        Parameters
        ----------
        lag : int
            Maximum time lag to test up to
        progress_bar : bool, optional
            Whether to display a progress bar during computation (default True)
            
        Returns
        -------
        tuple
            (ck_array, ck_error_array) containing the test results and their errors
        """
        if lag > len(self.pdfs) + 1:
            if len(self.pdfs) == 0:
                self.ck = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                self.ck_error = torch.zeros(lag - 1, device=self.device, dtype=self.dtype)
                n = len(self.xdigit) - 1
                xx = torch.cat((self.xdigit[1:], self.xdigit[:-1]), 0)
                xx = torch.reshape(xx, (2, n))
                self.pdfs.append(
                    Statistics.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                    device=self.device))
            else:
                self.ck = torch.cat((self.ck,
                                        torch.zeros(lag - len(self.pdfs),
                                                    device=self.device,
                                                    dtype=self.dtype)))
                self.ck_error = torch.cat((self.ck_error,
                                            torch.zeros(lag - len(self.pdfs),
                                                        device=self.device,
                                                        dtype=self.dtype)))

            for t3 in tqdm(range(len(self.pdfs) + 1, lag + 1), disable=not progress_bar):
                n = len(self.xdigit) - t3
                xx = torch.cat((self.xdigit[t3:], self.xdigit[:n]), 0)
                xx = torch.reshape(xx, (2, n))
                self.pdfs.append(
                    Statistics.pdf(xx, n, self.pdf_shape, dtype=self.dtype,
                                    device=self.device))

                t2 = int(t3 / 2)
                t1 = t3 - t2
                self.ck[t3 - 2], self.ck_error[t3 - 2] = CKTest.single_sit(self.pdfs[t3 - 1].to_dense(), n)
        ck_array = np.array(self.ck.tolist())
        ck_error_array = np.array(self.ck_error.tolist())
        self.lm = Statistics.markov_length(ck_array, ck_error_array)
        return ck_array, ck_error_array

    def plot(self,
             figsize: Tuple[int, int] = (12, 6),
             dpi: int = 80,
             style: str = '-',
             file: Optional[str] = None,
             grid: bool = True,
             linewidth: float = 0.3,
             s: int = 50) -> None:
        ck_array = np.array(self.ck.tolist())
        if self.is_sample:
            cbaru = np.max(ck_array)
            cbard = np.min(ck_array)
            lm_mask = self.lm > 0
            lm_x = self.lm[lm_mask]
            lm_y = np.arange(0, len(self.lm))
            lm_y = lm_y[lm_mask]
            misc.sample_plot(figsize, dpi, ck_array, (lm_x, lm_y), cbaru, cbard, s=s, file=file)
        else:
            ck_error_array = np.array(self.ck_error.tolist())
            misc.single_plot(ck_array,
                             ck_error_array,
                             self.lm,
                             figsize=figsize,
                             dpi=dpi,
                             style=style,
                             file=file,
                             grid=grid,
                             linewidth=linewidth,
                             s=s)


class CKTestCoupled():
    """
    Class for performing Chapman-Kolmogorov test on two coupled time series.
    
    This class handles both single sample and multiple sample coupled time series,
    testing for Markovian properties in the joint evolution of both variables.
    """

    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 dx: float = 1,
                 dy: float = 1,
                 bins_mode: str = 'std',
                 xbins: Union[int, ArrayLike] = 6,
                 ybins: Union[int, ArrayLike] = 6,
                 device: str = 'auto',
                 dtype: torch.dtype = torch.float64) -> None:
        self.x = x.copy()
        self.y = y.copy()
        self.pdfs = []
        self.ck = 0
        self.dtype = dtype
        self.lm = -1

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if np.ndim(x) == 2 and self.x.shape[0] > 1 and self.x.shape[1] > 1:
            self.is_sample = True
        else:
            self.is_sample = False
            self.x = self.x.reshape(-1)
            self.y = self.y.reshape(-1)

        if np.ndim(xbins) == 0:
            self.xbins = Statistics.find_bins(self.x, dx, bins_mode, max_bin=xbins)
        else:
            self.xbins = xbins.copy()

        if np.ndim(ybins) == 0:
            self.ybins = Statistics.find_bins(self.y, dy, bins_mode, max_bin=ybins)
        else:
            self.ybins = ybins.copy()

        self.pdf_shape = (len(self.xbins), len(self.ybins), len(self.xbins), len(self.ybins))

        self.xdigit = Statistics.digitize(self.x, self.xbins, dtype=self.dtype, device=self.device)
        self.ydigit = Statistics.digitize(self.y, self.ybins, dtype=self.dtype, device=self.device)

    def run_test(self, lag: int, progress_bar: bool = True) -> Tuple[NDArray, NDArray]:
        if self.is_sample:
            n = self.xdigit.shape[1]
            self.ck = torch.zeros((4, lag, lag), device=self.device, dtype=self.dtype)
            self.ck_error = torch.zeros((4, lag, lag), device=self.device, dtype=self.dtype)
            for t3 in tqdm(range(lag)):
                for t1 in range(lag):
                    t2 = int((t3 - t1) / 2)
                    t2 += t1

                    xyxy = torch.cat(
                        (self.xdigit[t3], self.ydigit[t3], self.xdigit[t1], self.ydigit[t1]), 0)
                    xyxy = torch.reshape(xyxy, (4, n))
                    p31 = Statistics.pdf(xyxy,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    xyxy = torch.cat(
                        (self.xdigit[t3], self.ydigit[t3], self.xdigit[t2], self.ydigit[t2]), 0)
                    xyxy = torch.reshape(xyxy, (4, n))
                    p32 = Statistics.pdf(xyxy,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    xyxy = torch.cat(
                        (self.xdigit[t2], self.ydigit[t2], self.xdigit[t1], self.ydigit[t1]), 0)
                    xyxy = torch.reshape(xyxy, (4, n))
                    p21 = Statistics.pdf(xyxy,
                                         n,
                                         self.pdf_shape,
                                         dtype=self.dtype,
                                         device=self.device).to_dense()

                    self.ck[:, t1, t3], self.ck_error[:, t1, t3] = CKTest.coupled(p31, p32, p21, n)

        else:
            if lag > len(self.pdfs) + 1:
                if len(self.pdfs) == 0:
                    self.ck = torch.zeros((4, lag - 1), device=self.device, dtype=self.dtype)
                    self.ck_error = torch.zeros((4, lag - 1), device=self.device, dtype=self.dtype)
                    n = len(self.xdigit) - 1
                    xyxy = torch.cat(
                        (self.xdigit[1:], self.ydigit[1:], self.xdigit[:-1], self.ydigit[:-1]), 0)
                    xyxy = torch.reshape(xyxy, (4, n))
                    self.pdfs.append(
                        Statistics.pdf(xyxy,
                                       n,
                                       self.pdf_shape,
                                       dtype=self.dtype,
                                       device=self.device))
                else:
                    self.ck = torch.cat(
                        (self.ck,
                         torch.zeros(
                             (4, lag - len(self.pdfs)), device=self.device, dtype=self.dtype)),
                        axis=1)
                    self.ck_error = torch.cat(
                        (self.ck_error,
                         torch.zeros(
                             (4, lag - len(self.pdfs)), device=self.device, dtype=self.dtype)),
                        axis=1)

                for t3 in tqdm(range(len(self.pdfs) + 1, lag + 1), disable=not progress_bar):
                    n = len(self.xdigit) - t3
                    xyxy = torch.cat(
                        (self.xdigit[t3:], self.ydigit[t3:], self.xdigit[:n], self.ydigit[:n]), 0)
                    xyxy = torch.reshape(xyxy, (4, n))
                    self.pdfs.append(
                        Statistics.pdf(xyxy,
                                       n,
                                       self.pdf_shape,
                                       dtype=self.dtype,
                                       device=self.device))

                    t2 = int(t3 / 2)
                    t1 = t3 - t2
                    self.ck[:, t3 - 2], self.ck_error[:, t3 - 2] = CKTest.coupled(
                        self.pdfs[t3 - 1].to_dense(), self.pdfs[t2 - 1].to_dense(),
                        self.pdfs[t1 - 1].to_dense(), n)
        ck_array = np.array(self.ck.tolist())
        ck_error_array = np.array(self.ck_error.tolist())
        self.lm = Statistics.markov_length(ck_array, ck_error_array, is_coupled=True)
        return ck_array, ck_error_array

    def plot(self,
             figsize: Tuple[int, int] = (12, 6),
             dpi: int = 80,
             style: str = '-',
             file: Optional[str] = None,
             grid: bool = True,
             linewidth: float = 0.3,
             s: int = 50,
             labels: Tuple[str, str, str, str] = ('x,x', 'x,y', 'y,x', 'y,y')) -> None:
        ck_array = np.array(self.ck.tolist())
        labels = ["{%s}" % i for i in labels]
        if self.is_sample:
            cbaru = np.max(ck_array)
            cbard = np.min(ck_array)
            lm_mask = self.lm > 0
            lm_y = np.arange(0, self.lm.shape[1])
            for i in range(4):
                misc.sample_plot(figsize,
                                 dpi,
                                 ck_array[i], (self.lm[i, lm_mask[i]], lm_y[lm_mask[i]]),
                                 cbaru,
                                 cbard,
                                 s=s,
                                 label=labels[i],
                                 label_no=i,
                                 file=file)
        else:
            ck_error_array = np.array(self.ck_error.tolist())
            for i in range(4):
                misc.single_plot(ck_array[i],
                                 ck_error_array[i],
                                 self.lm[i],
                                 figsize=figsize,
                                 dpi=dpi,
                                 style=style,
                                 file=file,
                                 grid=grid,
                                 linewidth=linewidth,
                                 s=s,
                                 label=labels[i],
                                 label_no=i)


class CKTest():
    """
    Static methods for performing Chapman-Kolmogorov test calculations.
    
    This class provides the core computational methods used by CKTestSingle 
    and CKTestCoupled classes.
    """

    @staticmethod
    def single_sit(p31: torch.Tensor,
                  n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs simplified independence test calculation for single time series.
        
        Parameters
        ----------
        p31 : torch.Tensor
            Joint probability distribution P(x3,x1)
        n : int
            Number of data points
            
        Returns
        -------
        tuple
            (ck, ck_var) containing test result and its variance
        """
        p1 = p31.sum(0)
        p3 = p31.sum(1)
        p1 = p1.reshape((1, p31.shape[1]))
        p3 = p3.reshape((p31.shape[0], 1))
        ck = torch.abs(p31 - torch.matmul(p3, p1)).sum()

        p31_var = Statistics.pdf_var(p31, n)
        p3_var = Statistics.pdf_var(p3, n)
        p1_var = Statistics.pdf_var(p1, n)
        ck_var = torch.sqrt((p31_var + torch.matmul(p3_var,p1**2) + torch.matmul(p3**2, p1_var))).sum()

        return ck, ck_var

    @staticmethod
    def single(p31: torch.Tensor,
              p32: torch.Tensor, 
              p21: torch.Tensor,
              n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs CK test calculation for single time series.
        
        Parameters
        ----------
        p31 : torch.Tensor
            Joint probability distribution P(x3,x1)
        p32 : torch.Tensor
            Joint probability distribution P(x3,x2)
        p21 : torch.Tensor
            Joint probability distribution P(x2,x1)
        n : int
            Number of data points
            
        Returns
        -------
        tuple
            (ck, ck_var) containing test result and its variance
        """
        # calculating the variance (square of error) of PDFs.
        p31_var = Statistics.pdf_var(p31, n)
        p21_var = Statistics.pdf_var(p21, n)

        # calculating p(x2)
        p2 = p21.sum(1)
        p2[p2 == 0] = float('inf')
        #calculating p(x3|x2) from p(x3,x2)
        p3c2 = p32 / p2

        p3c2_var = Statistics.pdf_var(p3c2, n) / p2
        # performing the test with given PDFs.
        ck = p31 - torch.matmul(p3c2, p21)
        ck = torch.abs(ck)

        # calculating the error.
        ck_var = p31_var + torch.matmul(p3c2_var, p21**2) + torch.matmul(p3c2**2, p21_var)

        return ck.sum(), torch.sqrt(ck_var).sum()

    @staticmethod
    def coupled(p31: torch.Tensor,
               p32: torch.Tensor,
               p21: torch.Tensor, 
               n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs CK test calculation for coupled time series.
        
        Parameters
        ----------
        p31 : torch.Tensor
            Joint probability distribution P(x3,y3,x1,y1)
        p32 : torch.Tensor
            Joint probability distribution P(x3,y3,x2,y2)
        p21 : torch.Tensor
            Joint probability distribution P(x2,y2,x1,y1)
        n : int
            Number of data points
            
        Returns
        -------
        tuple
            (ck, ck_var) containing test results and their variances for all variable combinations
        """
        device = p31.device
        ck = torch.zeros(4, device=device)
        ck_error = torch.zeros(4, device=device)

        p31_var = Statistics.pdf_var(p31, n)
        p21_var = Statistics.pdf_var(p21, n)

        p2 = p21.sum((2, 3))
        p2[p2 == 0] = float('inf')
        p3c2 = p32 / p2

        p3c2_var = Statistics.pdf_var(p3c2, n) / p2

        ck_tensor = p31 - torch.tensordot(p3c2, p21)
        ck_tensor_error = p31_var + torch.tensordot(p3c2_var, p21**2) + torch.tensordot(
            p3c2**2, p21_var)

        ck[0] = torch.abs(ck_tensor.sum((1, 3))).sum()
        ck_error[0] = torch.sqrt(ck_tensor_error.sum((1, 3))).sum()

        ck[1] = torch.abs(ck_tensor.sum((1, 2))).sum()
        ck_error[1] = torch.sqrt(ck_tensor_error.sum((1, 2))).sum()

        ck[2] = torch.abs(ck_tensor.sum((0, 3))).sum()
        ck_error[2] = torch.sqrt(ck_tensor_error.sum((0, 3))).sum()

        ck[3] = torch.abs(ck_tensor.sum((0, 2))).sum()
        ck_error[3] = torch.sqrt(ck_tensor_error.sum((0, 2))).sum()

        return ck, ck_error


class Statistics():
    """
    Static methods for statistical calculations used in CK tests.
    
    This class provides utility functions for binning data, calculating
    probability distributions and their variances, and determining Markov length.
    """

    @staticmethod
    def find_bins(x: ArrayLike,
                 dx: float,
                 bins_mode: str,
                 max_bin: int = 0) -> NDArray:
        """
        Determines bin edges for probability distribution calculations.
        
        Parameters
        ----------
        x : array_like
            Input data
        dx : float
            Bin width
        bins_mode : {'std', 'real'}
            Method for determining bin width
        max_bin : int, optional
            Maximum number of bins (default 0 for automatic)
            
        Returns
        -------
        array
            Array of bin edges
        """
        if bins_mode == 'std':
            std = np.std(x)
            x_std = x / std
        elif bins_mode == 'real':
            std = 1
            x_std = x.copy()
        if max_bin == 0:

            xdlimit = int(np.amin(x_std) * (1 / dx) - 1)
            xulimit = int(np.amax(x_std) * (1 / dx) + 1)
            xdlimit *= dx
            xulimit *= dx
            xbins = np.arange(xdlimit, xulimit + 2 * dx, dx) - dx / 2
        else:
            xdlimit = -max_bin * dx
            xulimit = max_bin * dx
            xbins = np.arange(xdlimit, xulimit + 2 * dx, dx) - dx / 2
        xbins *= std
        return xbins

    @staticmethod
    def digitize(x: ArrayLike,
                bins: ArrayLike,
                dtype: Optional[torch.dtype] = None,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Digitize data into bins."""
        x[x > bins[-1]] = bins[-1]
        x[x < bins[0]] = bins[0]
        xdig = np.digitize(x, bins) - 1
        # Convert numpy array to tensor directly
        xdig = torch.tensor(xdig, dtype=dtype, device=device)
        return torch.reshape(xdig, x.shape)

    @staticmethod
    def pdf(x_digitize: torch.Tensor,
           n: int,
           shape: Tuple[int, ...],
           dtype: Optional[torch.dtype] = None,
           device: Optional[torch.device] = None) -> torch.Tensor:
        """Calculate probability density function."""
        v = torch.ones_like(x_digitize[0], dtype=dtype, device=device)
        # Use sparse_coo_tensor instead of deprecated constructors
        p = torch.sparse_coo_tensor(
            indices=x_digitize.long(),
            values=v,
            size=torch.Size(shape),
            dtype=dtype,
            device=device
        )
        return (p.to_dense() / n).to_sparse()

    @staticmethod
    def pdf_var(p: torch.Tensor,
                n: int) -> torch.Tensor:
        return p * (1 - p) / n

    @staticmethod
    def markov_length(ck: ArrayLike,
                     ck_error: ArrayLike,
                     is_coupled: bool = False) -> NDArray:
        """
        Calculates the Markov length from CK test results.
        
        Parameters
        ----------
        ck : array_like
            CK test results
        ck_error : array_like
            CK test errors
        is_coupled : bool, optional
            Whether the input is from a coupled test (default False)
            
        Returns
        -------
        array
            Markov length(s) for the tested time series
        """
        diff = ck - ck_error

        if is_coupled:
            if len(diff.shape) == 3:
                b = np.ones_like(diff)
                diff = np.triu(diff, k=1) + np.tril(b, k=1)
                lm = np.argmax(diff <= 0, axis=2)
            else:
                lm = np.argmax(diff <= 0, axis=1) + 1

        else:
            if np.ndim(ck) == 2 and ck.shape[0] > 1:
                b = np.ones_like(diff)
                diff = np.triu(diff, k=1) + np.tril(b, k=1)
                lm = np.argmax(diff <= 0, axis=1)
            else:
                lm = np.argmax(diff <= 0) + 1
        return lm
