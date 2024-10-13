#Test Suite for statistics.py

import pytest
import dsmc_tool.statistics as stats

def test_CH():
    assert stats.CH(0.05, 0.1) == 1198
    assert stats.CH(0.1, 0.1) == 299
    assert stats.CH(0.05, 0.05) == 1475
    
def test_APMC():
    assert stats.APMC(0.1, 0.05, 0.1) == 264
    assert stats.APMC(0.2, 0.1, 0.1) == 132
    assert stats.APMC(0.3, 0.05, 0.05) == 941   