# Test Suite for eval_results.py

import pytest
import numpy as np
from dsmc.eval_results import eval_results
import dsmc.property as prop


def test_get_all_initial():
    property = prop.ReturnProperty()
    results = eval_results(property)
    assert results.get_all().size == 0
    
def test_get_all_extended():
    property = prop.ReturnProperty()
    results = eval_results(property)
    results.extend(1.0)
    results.extend(2.0)
    results.extend(3.0) 
    assert results.get_all().size == 3
    
def test_getter_initial():
    property = prop.ReturnProperty()
    results = eval_results(property)
    assert np.isnan(results.get_mean())
    assert np.isnan(results.get_variance())
    assert np.isnan(results.get_std())
    assert np.isnan(results.get_confidence_interval()[0])
    assert np.isnan(results.get_confidence_interval()[1])
    
def test_getter_extended1():
    property = prop.ReturnProperty()
    results = eval_results(property)
    results.extend(1.0)
    results.extend(2.0)
    results.extend(3.0)
    assert results.get_mean() == 2.0
    assert results.get_variance() == 1.0
    assert results.get_std() == 1.0
    assert round(results.get_confidence_interval()[0], 3) == -0.484
    assert round(results.get_confidence_interval()[1], 3) == 4.484
    
def test_getter_extended2():
    property = prop.ReturnProperty()
    results = eval_results(property)
    results.extend(2.826)
    results.extend(87.16)
    results.extend(87.59)
    assert round(results.get_mean(), 3) == 59.192
    assert round(results.get_variance(), 3) == 2382.891
    assert round(results.get_std(), 3) == 48.815
    assert round(results.get_confidence_interval()[0], 3) == -62.071
    assert round(results.get_confidence_interval()[1], 3) == 180.455
    
def test_getter_extended_binomial():
    property = prop.GoalReachingProbabilityProperty()
    results = eval_results(property)
    results.extend(1.0)
    results.extend(0.0)
    results.extend(1.0)
    results.total_episodes = 3
    assert round(results.get_mean(), 3) == 0.667
    assert round(results.get_variance(), 3) == 0.333
    assert round(results.get_std(), 3) == 0.577
    assert round(results.get_confidence_interval()[0], 3) == 0.013
    assert round(results.get_confidence_interval()[1], 3) == 1.32
    
def test_get_variance_extended_binomial():
    property = prop.GoalReachingProbabilityProperty()
    results = eval_results(property)
    results.extend(1.0)
    results.total_episodes = 1
    assert results.get_variance() == 0.0
    
    
    