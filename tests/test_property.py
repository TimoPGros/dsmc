#Test Suite for property.py

import pytest
import numpy as np
import dsmc.property as prop
import dsmc.eval_results as er

def test_property_rename():
    property = prop.GoalReachingProbabilityProperty(name="goal")
    assert property.name == "goal"
    
def test_goal_reaching_probability_property1():
    property = prop.GoalReachingProbabilityProperty(goal_reward=33)
    trajectory = [(None, None, 1), (None, None, 55), (None, None, 33)]
    assert property.check(trajectory) == 1.0

def test_goal_reaching_probability_property2():
    property = prop.GoalReachingProbabilityProperty(goal_reward=33)
    trajectory = [(None, None, 1), (None, None, 55), (None, None, 32)]
    assert property.check(trajectory) == 0.0
    
def test_return_property1():
    property = prop.ReturnProperty(gamma = 1.0)
    trajectory = [(None, None, 1), (None, None, 552), (None, None, 33)]
    assert property.check(trajectory) == 89.0
    
def test_return_property2():
    property = prop.ReturnProperty(gamma = 0.5)
    trajectory = [(None, None, 1), (None, None, 55), (None, None, 33)]
    assert property.check(trajectory) == 36.75
    
def test_normalized_return_property1():
    property = prop.NormalizedReturnProperty(gamma = 1.0)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 2.0
    
def test_normalized_return_property2():
    property = prop.NormalizedReturnProperty(gamma = 0.5)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert round(property.check(trajectory), 3) == 0.917
    
def test_episode_length_property1():
    property = prop.EpisodeLengthProperty()
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 3
    
def test_episode_length_property2():
    property = prop.EpisodeLengthProperty()
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)]
    assert property.check(trajectory) == 4
    
def test_action_diversity_property():
    property = prop.ActionDiversityProperty(num_actions=5)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 0.6
    
def test_state_coverage_property():
    property = prop.StateCoverageProperty(num_states=10)
    trajectory = [("s1", None, None), ("s2", None, None), ("s3", None, None)]
    assert property.check(trajectory) == 0.3
    
def test_action_entropy_property1():
    property = prop.ActionEntropyProperty(num_actions=5)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert round(property.check(trajectory), 3) == 1.099
    
def test_action_entropy_property2():
    property = prop.ActionEntropyProperty(num_actions=5)
    trajectory = [(None, 1, None), (None, 1, None), (None, 1, None)]
    assert property.check(trajectory) == 0.0
    
def test_path_efficiency_property1():
    property = prop.PathEfficiencyProperty(optimal_path=[1, 2, 3])
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 1.0
    
def test_path_efficiency_property2():
    property = prop.PathEfficiencyProperty(optimal_path=[1, 2, 3, 4])
    trajectory = [(None, 1, None), (None, 2, None), (None, 1, None), (None, 2, None)]
    assert property.check(trajectory) == 0.5
    
def test_path_length_efficiency_property1():
    property = prop.PathLengthEfficiencyProperty(optimal_path_length=3)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 1.0
    
def test_path_length_efficiency_property2():
    property = prop.PathLengthEfficiencyProperty(optimal_path_length=3)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None), (None, 4, None)]
    assert property.check(trajectory) == 0.75
    
def test_reward_variance_property1():
    property = prop.RewardVarianceProperty()
    trajectory = [(None, None, 1), (None, None, 1), (None, None, 1)]
    assert property.check(trajectory) == 0.0
    
def test_reward_variance_property2():
    property = prop.RewardVarianceProperty()
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert round(property.check(trajectory), 3) == 0.667
    
def test_state_transition_smoothness_property1():
    property = prop.StateTransitionSmoothnessProperty()
    trajectory = [("s1", None, None)]
    assert property.check(trajectory) == 0.0
    
def test_state_transition_smoothness_property2():
    property = prop.StateTransitionSmoothnessProperty()
    trajectory = [([0,0], None, None), ([3,4], None, None)] 
    assert property.check(trajectory) == 5.0
    
def test_state_transition_smoothness_property3():
    property = prop.StateTransitionSmoothnessProperty()
    trajectory = [([0,0], None, None), ([3,4], None, None), ([6,8], None, None)]
    assert property.check(trajectory) == 5.0
    
def test_reward_to_length_ratio_property1():
    property = prop.RewardToLengthRatioProperty()
    trajectory = [(None, None, 1), (None, None, 1), (None, None, 1)]
    assert property.check(trajectory) == 1.0
    
def test_reward_to_length_ratio_property2():
    property = prop.RewardToLengthRatioProperty()
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 2.0
    
def test_state_visit_property1():
    property = prop.StateVisitProperty(target_state="s1")
    trajectory = [("s1", None, None), ("s2", None, None), ("s3", None, None)]
    assert property.check(trajectory) == 1.0
    
def test_state_visit_property2():
    property = prop.StateVisitProperty(target_state="s4")
    trajectory = [("s1", None, None), ("s2", None, None), ("s3", None, None)]
    assert property.check(trajectory) == 0.0
    
def test_action_threshold_property1():
    property = prop.ActionThresholdProperty(action = 1, threshold=2)
    trajectory = [(None, 1, None), (None, 1, None), (None, 1, None)]
    assert property.check(trajectory) == 1.0
    
def test_action_threshold_property2():
    property = prop.ActionThresholdProperty(action = 1, threshold=2)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 0.0
    
def test_goal_before_step_limit_property1():
    property = prop.GoalBeforeStepLimitProperty(goal_reward=3, step_limit=5)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 1.0
    
def test_goal_before_step_limit_property2():
    property = prop.GoalBeforeStepLimitProperty(goal_reward=3, step_limit=2)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 0.0
    
def test_action_taken_property1():
    property = prop.ActionTakenProperty(action=2)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 1.0
    
def test_action_taken_property2():
    property = prop.ActionTakenProperty(action=4)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 0.0
    
def test_consecutive_same_action_property1():
    property = prop.ConsecutiveSameActionProperty(action=1, threshold=3)
    trajectory = [(None, 1, None), (None, 1, None), (None, 1, None)]
    assert property.check(trajectory) == 1.0
    
def test_consecutive_same_action_property2():
    property = prop.ConsecutiveSameActionProperty(action=1, threshold=3)
    trajectory = [(None, 1, None), (None, 2, None), (None, 1, None)]
    assert property.check(trajectory) == 0.0
    
def test_return_threshold_property1():
    property = prop.ReturnThresholdProperty(gamma=1.0, threshold=5)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 1.0
    
def test_return_threshold_property2():
    property = prop.ReturnThresholdProperty(gamma=1.0, threshold=7)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 0.0
    
def test_action_variety_property1():
    property = prop.ActionVarietyProperty(threshold=3)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 1.0
    
def test_action_variety_property2():
    property = prop.ActionVarietyProperty(threshold=4)
    trajectory = [(None, 1, None), (None, 2, None), (None, 3, None)]
    assert property.check(trajectory) == 0.0
    
def test_early_termination_property1():
    property = prop.EarlyTerminationProperty(step_maximum=5)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 1.0
    
def test_early_termination_property2():
    property = prop.EarlyTerminationProperty(step_maximum=2)
    trajectory = [(None, None, 1), (None, None, 2), (None, None, 3)]
    assert property.check(trajectory) == 0.0