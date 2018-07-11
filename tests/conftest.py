from typing import Tuple

import logging
import os
import pytest

# could use hypothesis numpy instead
@pytest.fixture
def a():
    return np.random.rand(256)

@pytest.fixture
def b():
    return np.random.rand(256)

