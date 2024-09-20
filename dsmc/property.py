import numpy as np
from typing import List, Tuple, Any

#DONE: customize save file for property

# base class for evaluation properties
class Property:
    def __init__(self, name: str, json_filename: str):
        self.name = name
        self.json_filename = json_filename
        pass

    # we assume a trajectory is a list of tuples (observation, action, reward)
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        pass

# metric that checks if the goal has been reached
class GoalReachingProbabilityProperty(Property):
    def __init__(self, name: str = "grp", json_filename: str = "grp.json", goal_reward: float = 100):
        super().__init__(name, json_filename)
        self.goal_reward = goal_reward
        self.binomial = True

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        if trajectory[-1][2] == self.goal_reward:
            return 1.0
        else:
            return 0.0

# metric that calculates the return of a trajectory
class ReturnProperty(Property):
    def __init__(self, name: str = "return", json_filename: str = "return.json", gamma: float = 0.99):
        super().__init__(name, json_filename)
        self.gamma = gamma
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret