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

# metric that calculates the ratio of unique actions taken   
class ActionDiversityProperty(Property):
    def __init__(self, name: str = "action_diversity", num_actions: int = 10):
        super().__init__(name)
        self.binomial = False
        self.num_actions = num_actions
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        return len(set(actions)) / self.num_actions
    
# metric that calculates the entropy of the actions taken  
class ActionEntropyProperty(Property):
    def __init__(self, name: str = "action_entropy", num_actions: int = 10):
        super().__init__(name)
        self.binomial = False
        self.num_actions = num_actions
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        action_counts = np.bincount(actions, minlength=self.num_actions)
        action_probs = action_counts / len(actions)
        action_probs = action_probs[action_probs > 0]
        return -np.sum(action_probs * np.log(action_probs))
    
# metric that tests if agent has taken a specific action
class ActionTakenProperty(Property):
    def __init__(self, name: str = "action_taken", action: Any = 0):
        super().__init__(name)
        self.action = action
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        return 1.0 if self.action in actions else 0.0

# metric that tests if agent takes a specific action more than a specified number of times
class ActionThresholdProperty(Property):
    def __init__(self, name: str = "action_threshold", action: Any = 0, threshold: int = 10):
        super().__init__(name)
        self.action_id = action
        self.threshold = threshold
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        return 1.0 if actions.count(self.action_id) >= self.threshold else 0.0

# metric that tests if a certain number of unique actions was taken
class ActionVarietyProperty(Property):
    def __init__(self, name: str = "action_variety", threshold: int = 3):
        super().__init__(name)
        self.threshold = threshold
        self.binomial = True

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        unique_actions = set([action for _, action, _ in trajectory])
        return 1.0 if len(unique_actions) >= self.threshold else 0.0

# metric that tests if agent has taken specific action for a specified number of consecutive steps
class ConsecutiveSameActionProperty(Property):
    def __init__(self, name: str = "consecutive_same_action", action: Any = 0, threshold: int = 10):
        super().__init__(name)
        self.action = action
        self.threshold = threshold
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        consecutive = 0
        for action in actions:
            if action == self.action:
                consecutive += 1
                if consecutive >= self.threshold:
                    return 1.0
            else:
                consecutive = 0
        return 0.0

# metric that tests if episode was terminated before a certain step maximum
class EarlyTerminationProperty(Property):
    def __init__(self, name: str = "early_termination", step_maximum: int = 100):
        super().__init__(name)
        self.step_maximum = step_maximum
        self.binomial = True

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        return 1.0 if len(trajectory) <= self.step_maximum else 0.0

# metric that returns the episode length   
class EpisodeLengthProperty(Property):
    def __init__(self, name: str = "episode_length"):
        super().__init__(name)
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        return len(trajectory)

# metric that tests if agent reaches goal state within a specified number of steps
class GoalBeforeStepLimitProperty(Property):
    def __init__(self, name: str = "goal_before_step_limit", goal_reward: float = 100, step_limit: int = 100):
        super().__init__(name)
        self.goal_reward = goal_reward
        self.step_limit = step_limit
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        if trajectory[-1][2] == self.goal_reward and len(trajectory) <= self.step_limit:
            return 1.0
        else:
            return 0.0

# metric that checks if the goal has been reached
class GoalReachingProbabilityProperty(Property):
    def __init__(self, name: str = "goal_reaching_probability", goal_reward: float = 100):
        super().__init__(name)
        self.goal_reward = goal_reward
        self.binomial = True

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        if trajectory[-1][2] == self.goal_reward:
            return 1.0
        else:
            return 0.0

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

# metric that calculates how efficient the path taken is compared to the optimal path    
class PathEfficiencyProperty(Property):
    def __init__(self, name: str = "path_efficiency", optimal_path: List[Any] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__(name)
        self.optimal_path = optimal_path
        self.binomial = False
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actions = [action for _, action, _ in trajectory]
        return np.mean([a == b for a, b in zip(actions, self.optimal_path)])

# metric that calculates how efficient the path taken is compared to the optimal path's length (if whole path not explicitely known)
class PathLengthEfficiencyProperty(Property):
    def __init__(self, name: str = "path_length_efficiency", optimal_path_length: int = 1):
        super().__init__(name)
        self.binomial = False
        self.optimal_path_length = optimal_path_length

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        actual_path_length = len(trajectory)
        return self.optimal_path_length / actual_path_length if actual_path_length > 0 else 0.0

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

# metric that tests if return is above a specified threshold
class ReturnThresholdProperty(Property):
    def __init__(self, name: str = "return_threshold", gamma: int = 0.99, threshold: float = 100):
        super().__init__(name)
        self.gamma = gamma
        self.threshold = threshold
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return 1.0 if ret >= self.threshold else 0.0

# metric that calculates the ratio of rewards to the length of the trajectory
class RewardToLengthRatioProperty(Property):
    def __init__(self, name: str = "reward_to_length_ratio"):
        super().__init__(name)
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        total_reward = sum([reward for _, _, reward in trajectory])
        return total_reward / len(trajectory) if len(trajectory) > 0 else 0.0

# metric that calculates the variance of the rewards    
class RewardVarianceProperty(Property):
    def __init__(self, name: str = "reward_variance"):
        super().__init__(name)
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        rewards = [reward for _, _, reward in trajectory]
        return np.var(rewards)

# metric that calculates the ratio of unique states visited   
class StateCoverageProperty(Property):
    def __init__(self, name: str = "state_coverage", num_states: int = 100):
        super().__init__(name)
        self.binomial = False
        self.num_states = num_states
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        states = [state for state, _, _ in trajectory]
        return len(set(tuple(state) for state in states)) / self.num_states    

# metric that calculates the smoothness of the state transitions
# IMPORTANT: this metric assumes that states are represented as vectors
class StateTransitionSmoothnessProperty(Property):
    def __init__(self, name: str = "state_transition_smoothness"):
        super().__init__(name)
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        states = [state for state, _, _ in trajectory]
        if len(states) < 2:
            return 0.0
        diffs = [np.linalg.norm(np.array(states[i]) - np.array(states[i-1])) for i in range(1, len(states))]
        return np.mean(diffs)

# metric that tests if agent has visited a specific, given state
class StateVisitProperty(Property):
    def __init__(self, name: str = "state_visit", target_state: Any = [0, 0]):
        super().__init__(name)
        if target_state is None:
            raise ValueError("target_state must be set")
        self.state = target_state
        self.binomial = True
        
    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        states = [state for state, _, _ in trajectory]
        return 1.0 if self.state in states else 0.0