import numpy as np


def vc_generalization_bound(emp_error, h, N, delta=0.05):
    """Compute a simple VC-dimension generalization bound.

    This preserves the original formula used in the app.
    """
    if N <= 0 or h <= 0:
        return None
    epsilon = np.sqrt((h * (np.log(2 * N / h + 1) + 1) + np.log(4 / delta)) / N)
    return emp_error + epsilon
