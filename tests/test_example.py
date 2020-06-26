"""
This is an example test file, it starts with 'test_' so pytest can
identify it. To manually run tests run 'pytest' in cmd line.
For each test pytest will return '.','F' if test passes, fails
"""
import sigmet.example_module as example_module


def test_add_zeros():
    """
    This is an example test, can start, end with "test_", "_test" respectively
    """
    x = 0
    y = 0
    assert 0 == example_module.add(x, y)


def test_add_positive_negative():
    """
    Another test, ideally each test is structured so similar cases are grouped but
    if a test fails can see which function in the error traceback
    """
    x = 5
    y = -2
    assert 3 == example_module.add(x, y)
