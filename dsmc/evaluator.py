import numpy as np
from dsmc.eval_results import eval_results
from scipy.stats import norm
import gymnasium as gym
from gymnasium import Env as GymEnv

from typing import List, Tuple, Any, Dict

#DONE: Added type hints and default values
#DONE: Made functions private

def __CH(kappa: float, eps: float):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))

def __APMC(s2: float, kappa: float, eps: float):
    z = norm.ppf(1 - kappa / 2)
    return np.ceil(4 * z * s2 / np.power(eps, 2))

#DONE: implemented this function
def __construct_confidence_interval_length(results: eval_results, kappa: float, epsilon: float):
    interval = results.get_confidence_interval(kappa, epsilon)
    confidence_interval_length = interval[1] - interval[0]
    return confidence_interval_length

# base class for evaluation properties
class Property:
    def __init__(self, name: str):
        self.name = name
        pass

    # we assume a trajectory is a list of tuples (observation, action, reward)
    def __check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        pass

# metric that checks if the goal has been reached
class GoalReachingProbabilityProperty(Property):
    def __init__(self, name: str = "grp", goal_reward: float = 100):
        super().__init__(name)
        self.goal_reward = goal_reward
        self.binomial = True

    def __check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        if trajectory[-1][2] == self.goal_reward:
            return 1.0
        else:
            return 0.0

# metric that calculates the return of a trajectory
class ReturnProperty(Property):
    def __init__(self, name: str = "return", gamma: float = 0.99):
        super().__init__(name)
        self.gamma = gamma
        self.binomial = False

    def __check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret

class Evaluator:

    def __init__(self, env: GymEnv = gym.make("pgtg-v2"), gamma: float = 0.99, initial_episodes: int = 100, episodes_per_run: int = 50):
        self.env = env
        self.gamma = gamma
        self.initial_episodes = initial_episodes
        self.episodes_per_run = episodes_per_run

        self.properties = {}

    # register a new evaluation property
    def register_property(self, property: Property = ReturnProperty()):
        self.properties[property.name] = property

    def __run_policy(self, agent, num_episodes: int, results_per_property: Dict[str, eval_results], act_function):
        if act_function is None:
            act_function = agent.predict
            
        if not callable(act_function):
            raise ValueError("act_function should be a function.")
        #DONE: Checked this RL loop
        for _ in range(num_episodes):
            state = self.env.reset()
            trajectory = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act_function(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            # store new results in EvaluationResults object
            for property in self.properties.values():
                results_per_property[property.name].extend(property.__check(trajectory))

    def eval(self, agent, epsilon: float = 0.1, kappa: float = 0.05, act_function = None):
        # initialize EvaluationResults object for each class and whether the property converged
        results_per_property = {}
        converged_per_property = {}
        for property in self.properties.values():
            results_per_property[property.name] = eval_results(property=property.name)
            converged_per_property[property.name] = False

        # run initial episodes - one run, such that the first run of the while loop checks convergence
        
        #TODO: why -self.episodes_per_run?
        
        self.__run_policy(agent, self.initial_episodes - self.episodes_per_run, results_per_property, act_function)
        made_episodes = self.initial_episodes - self.episodes_per_run
        for property in self.properties.values():
                results_per_property[property.name].total_episodes = self.initial_episodes - self.episodes_per_run

        # compute the CH bound
        ch_bound = __CH(kappa, epsilon)
        # run the policy until all properties have converged
        while True:
            # run the policy for the specified number of episodes
            self.__run_policy(agent, self.episodes_per_run, results_per_property, act_function)
            made_episodes += self.episodes_per_run
            for property in self.properties.values():
                results_per_property[property.name].total_episodes += self.episodes_per_run

            # compute for each property the APMC bound and the confidence interval length
            for property in self.properties.values():
                property_results = results_per_property[property.name]

                apmc_bound = __APMC(property_results.get_variance(), kappa, epsilon)
                confidence_interval_length = __construct_confidence_interval_length(property_results, kappa, epsilon)

                # check if the property has converged, property can also become non-converged again!!!
                if made_episodes > ch_bound or made_episodes > apmc_bound or confidence_interval_length < 2 * epsilon:
                    converged_per_property[property.name] = True
                else:
                    converged_per_property[property.name] = False

            if all(converged_per_property.values()):
                break

        return results_per_property
