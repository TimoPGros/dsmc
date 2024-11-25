import numpy as np
from scipy.stats import norm
from dsmc_tool.eval_results import Eval_results

# Calculates the Chernoff-Hoeffding bound: maximum necessary number of episodes, according to kappa and epsilon
def CH(eps: float = 0.1, kappa: float = 0.05):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))

# Calculates the length of a confidence interval
def construct_confidence_interval_length(results: Eval_results = np.array([]), kappa: float = 0.05):
    interval = results.get_confidence_interval(kappa)
    confidence_interval_length = interval[1] - interval[0]
    return confidence_interval_length