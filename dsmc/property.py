import numpy as np
from typing import List, Tuple, Any

# base class for evaluation properties
class Property:
    def __init__(self, name: str):
        self.name = name
        pass

    # we assume a trajectory is a list of tuples (observation, action, reward)
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        pass

# metric that checks if the goal has been reached
class GoalReachingProbabilityProperty(Property):
    def __init__(self, name: str = "grp", goal_reward: float = 100):
        super().__init__(name)
        self.goal_reward = goal_reward
        self.binomial = True

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
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

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret