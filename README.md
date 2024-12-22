# Chapman-Kolmogorov Test Implementation

A Python implementation for performing Chapman-Kolmogorov (CK) tests on time series data. This package helps verify if a process is Markovian by checking if the Chapman-Kolmogorov equation is satisfied within statistical error.

## Installation

### Requirements
- Python 3.7+
- NumPy
- PyTorch
- Matplotlib
- tqdm

Install the required packages:
```bash
pip install numpy torch matplotlib tqdm
```

For GPU acceleration, make sure you have CUDA installed and install the CUDA-enabled version of PyTorch:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

```python
import numpy as np
from cktest import CKTestSingle, CKTestCoupled

# Generate some sample data
time_series = np.random.normal(size=1000)

# Create a CK test instance
# GPU will be used automatically if available
ck = CKTestSingle(
    x=time_series,
    dx=1,           # bin width
    bins_mode='std' # binning mode ('std' or 'real')
)

# Or explicitly specify device
ck = CKTestSingle(
    x=time_series,
    device='cuda:0'  # use 'cpu' for CPU-only computation
)

# Run the test
lag = 10  # maximum time lag to test
ck_values, ck_errors = ck.run_test(lag=lag)

# Plot the results
ck.plot()
```

### Testing Coupled Time Series

```python
# For two coupled time series
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

ck_coupled = CKTestCoupled(
    x=x,
    y=y,
    dx=1,
    dy=1,
    bins_mode='std'
)

ck_values, ck_errors = ck_coupled.run_test(lag=10)
ck_coupled.plot()
```

### Using Toy Models

The package includes toy models for testing:

```python
from toy_model import ToyModel

# Generate ARFIMA process
time = 1000
samples = 1
n_coeffs = 100
d = 0.3  # differencing parameter

x = ToyModel.arfima(
    time=time,
    samples=samples,
    n=n_coeffs,
    d=d
)

# Generate coupled ARFIMA processes
x, y = ToyModel.coupled_arfima(
    time=time,
    samples=samples,
    n=n_coeffs,
    dx=0.3,
    dy=0.4,
    w=0.5  # coupling strength
)
```

## Key Components

### CKTestSingle
- Tests single time series for Markovian properties
- Supports both single and multiple sample analysis
- Provides visualization tools for results

### CKTestCoupled
- Tests two coupled time series
- Analyzes cross-dependencies between variables
- Supports both single and multiple sample analysis

### ToyModel
- Generates test data including ARFIMA processes
- Creates coupled time series with controllable parameters
- Useful for testing and validation

## Parameters

### CKTestSingle/CKTestCoupled
- `x`, `y`: Input time series data
- `dx`, `dy`: Bin widths for probability distributions
- `bins_mode`: Method for determining bin width ('std' or 'real')
- `bins`: Number of bins or custom bin edges
- `device`: Computation device ('auto', 'cpu', or 'cuda')

### Test Methods
- `run_test(lag)`: Performs CK test up to specified time lag
- `run_sit(lag)`: Performs simplified independence test
- `plot()`: Visualizes test results

## Output

The test methods return:
1. CK test values: Measure of deviation from Markov property
2. CK test errors: Statistical uncertainties
3. Markov length: Time scale at which Markov property breaks down

## Examples

### Testing a Single Time Series

```python
import numpy as np
from cktest import CKTestSingle

# Generate random walk
steps = np.random.normal(size=1000)
position = np.cumsum(steps)

# Perform CK test
ck_test = CKTestSingle(position, dx=0.5, bins_mode='std')
values, errors = ck_test.run_test(lag=20)

# Plot results
ck_test.plot(figsize=(10, 6), dpi=100)
```

### Testing Coupled Variables

```python
from cktest import CKTestCoupled

# Create coupled test instance
ck_coupled = CKTestCoupled(
    x=x_data,
    y=y_data,
    dx=0.5,
    dy=0.5,
    bins_mode='std'
)

# Run test and plot
values, errors = ck_coupled.run_test(lag=20)
ck_coupled.plot(
    labels=('x,x', 'x,y', 'y,x', 'y,y'),
    figsize=(12, 8)
)

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Motahari, H., Noudehi, M. G., Jamali, T., & Jafari, G. R. (2024). Generalization of Chapman-Kolmogorov equation to two coupled processes. Research Square. [https://doi.org/10.21203/rs.3.rs-3993646/v1](https://doi.org/10.21203/rs.3.rs-3993646/v1)

   This paper introduces the generalized version of the Chapman-Kolmogorov equation (CKE) for estimating memory size in coupled processes, which forms the theoretical foundation for the coupled time series analysis implemented in this package.
