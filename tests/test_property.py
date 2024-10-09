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