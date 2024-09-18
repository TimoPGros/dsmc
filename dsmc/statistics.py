import numpy as np
from dsmc.eval_results import eval_results
from scipy.stats import norm

def CH(kappa: float, eps: float):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))

def APMC(s2: float, kappa: float, eps: float):
    z = norm.ppf(1 - kappa / 2)
    return np.ceil(4 * z * s2 / np.power(eps, 2))

def construct_confidence_interval_length(results: eval_results, kappa: float, epsilon: float):
    interval = results.get_confidence_interval(kappa)
    confidence_interval_length = interval[1] - interval[0]
    return confidence_interval_length