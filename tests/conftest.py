import numpy as np
import pytest


@pytest.fixture
def a():
    return np.random.rand(256)


@pytest.fixture
def b():
    return np.random.rand(256)

