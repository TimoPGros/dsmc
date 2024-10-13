# test suite for evaluator.py

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import dsmc_tool.property as prop
import pgtg
from dsmc_tool.eval_results import eval_results
from stable_baselines3 import DQN
from unittest.mock import patch

from typing import Dict

from dsmc_tool.evaluator import Evaluator

@pytest.fixture
def mock_input():
    with patch('builtins.input', return_value="50"):
        yield

def test_register_property():
    env = gym.make("pgtg-v3")
    evaluator = Evaluator(env=env)
    early_property = prop.ReturnProperty()
    evaluator.register_property(early_property)
    assert early_property.name in evaluator.properties
    
def test_run_policy_episode_update():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.StateCoverageProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, False, False, True, 1)
    assert results_per_property[early_property.name].total_episodes == 7
    assert evaluator.made_episodes == 7
    
def test_run_policy_trajectory_update():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.PathEfficiencyProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, False, False, True, 1)
    assert results_per_property[early_property.name].get_all().size == 7
    
def test_run_policy_act_function():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.PathEfficiencyProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    act_function = 3
    with pytest.raises(ValueError):
        evaluator._Evaluator__run_policy(agent, 7, results_per_property, act_function, False, False, True, 1)
        
def test_run_policy_interim_results():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.StateTransitionSmoothnessProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, True, 1)
    assert results_per_property[early_property.name].total_episodes == 7
    assert evaluator.made_episodes == 7
    
def test_run_policy_interim_results_final():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ActionDiversityProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, True, 1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, False, 1)
    assert results_per_property[early_property.name].total_episodes == 14
    assert evaluator.made_episodes == 14 
    
def test_run_policy_no_interim_results():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.GoalBeforeStepLimitProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, False, False, True, 1)
    assert results_per_property[early_property.name].total_episodes == 7
    assert evaluator.made_episodes == 7
    
def test_run_policy_no_interim_results_final():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ActionThresholdProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, False, False, True, 1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, False, False, False, 1)
    assert results_per_property[early_property.name].total_episodes == 14
    assert evaluator.made_episodes == 14
    
def test_run_policy_interim_interval():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ActionTakenProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, True, 2)
    assert results_per_property[early_property.name].total_episodes == 7
    assert evaluator.made_episodes == 7
    
def test_run_policy_interim_interval_final():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ConsecutiveSameActionProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, True, 2)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, False, False, 2)
    assert results_per_property[early_property.name].total_episodes == 14
    assert evaluator.made_episodes == 14
    
def test_run_policy_interim_results_full_results_list():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ActionThresholdProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, True, True, 1)
    assert results_per_property[early_property.name].total_episodes == 7
    assert evaluator.made_episodes == 7
    
def test_run_policy_interim_results_final_full_results_list():
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env)
    early_property = prop.ActionThresholdProperty()
    evaluator.register_property(early_property)
    results_per_property = {}
    for property in evaluator.properties.values():
            results_per_property[property.name] = eval_results(property=property)
    agent = DQN("MlpPolicy", env, verbose=1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, True, True, 1)
    evaluator._Evaluator__run_policy(agent, 7, results_per_property, None, True, True, False, 1)
    assert results_per_property[early_property.name].total_episodes == 14
    assert evaluator.made_episodes == 14
    
def test_eval_correct_totals_interim(mock_input):
    env = gym.make("pgtg-v3")
    env = FlattenObservation(env)
    evaluator = Evaluator(env=env, initial_episodes=49, evaluation_episodes=49)
    early_property = prop.ActionThresholdProperty()
    evaluator.register_property(early_property)
    agent = DQN("MlpPolicy", env, verbose=1)
    results = evaluator.eval(agent, epsilon=0.1, kappa=0.05, save_interim_results=True)
    assert results[early_property.name].total_episodes % 49 == 0
    assert evaluator.made_episodes % 49 == 0
    

    