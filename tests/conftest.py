import pytest

from data_loader import load_all


@pytest.fixture(scope="session")
def package():
    return load_all()
