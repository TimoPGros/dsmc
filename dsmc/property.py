import numpy as np
from typing import List, Tuple, Any

# base class for evaluation properties
class Property:
    def __init__(self, name: str):
        self.name = name
        self.json_filename = name + ".json"
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

# metric that calculates the normalized return of a trajectory       
class NormalizedReturnProperty(Property):
    def __init__(self, name: str = "normalized_return", gamma: float = 0.99):
        super().__init__(name)
        self.gamma = gamma
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret / len(trajectory)
        

# metric that returns the episode length   
class EpisodeLengthProperty(Property):
    def __init__(self, name: str = "episode_length"):
        super().__init__(name)
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        return len(trajectory)

# metric that calculates the ratio of unique actions taken   
class ActionDiversityProperty(Property):
    def __init__(self, name: str = "action_diversity", num_actions: int = 10):
        super().__init__(name)
        self.binomial = False
        self.num_actions = num_actions
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        return len(set(actions)) / self.num_actions

# metric that calculates the ratio of unique states visited   
class StateCoverageProperty(Property):
    def __init__(self, name: str = "state_coverage", num_states: int = 100):
        super().__init__(name)
        self.binomial = False
        self.num_states = num_states
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        states = [state for state, _, _ in trajectory]
        return len(set(tuple(state) for state in states)) / self.num_states