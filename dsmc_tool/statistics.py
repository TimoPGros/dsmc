import numpy as np
from scipy.stats import norm
from dsmc_tool.eval_results import eval_results

# Calculates the Chernoff-Hoeffding bound: maximum necessary number of episodes, according to kappa and epsilon
def CH(eps: float = 0.1, kappa: float = 0.05):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))

# Calculates another maximum necessary number of episodes, according to kappa, epsilon, and the property's variance
def APMC(var: float = 0, eps: float = 0.1, kappa: float = 0.05):
    z = norm.ppf(1 - kappa / 2)
    return np.ceil(4 * z * var / np.power(eps, 2))

# Calculates the length of a confidence interval
def construct_confidence_interval_length(results: eval_results = np.array([]), kappa: float = 0.05):
    interval = results.get_confidence_interval(kappa)
    confidence_interval_length = interval[1] - interval[0]
    return confidence_interval_length