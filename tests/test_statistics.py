#Test Suite for statistics.py

import pytest
import dsmc.statistics as stats

def test_CH():
    assert stats.CH(0.05, 0.1) == 185
    assert stats.CH(0.1, 0.1) == 150
    assert stats.CH(0.05, 0.05) == 738
    
def test_APMC():
    assert stats.APMC(0.1, 0.05, 0.1) == 39
    assert stats.APMC(0.2, 0.1, 0.1) == 55
    assert stats.APMC(0.3, 0.05, 0.05) == 461    