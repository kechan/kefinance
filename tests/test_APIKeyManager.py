import pytest
from kefinance import APIKeyManager  # replace "your_module" with the actual module name where APIKeyManager is defined

@pytest.fixture(scope='module')
def manager():
    return APIKeyManager(key_store='data/keys_df')

def test_valid_key_retrieval(manager):
    # Test the positive case for retrieving an API key with a known script name and part
    key = manager.get_key('abc.py', 0)

    # This assertion checks that the returned key is a non-empty string
    # If you know what the key looks like, you can be more specific in this test
    assert isinstance(key, str) and len(key) > 0

def test_invalid_script_name(manager):
    # Test the negative case for retrieving an API key with a non-existing script name
    with pytest.raises(ValueError):
        manager.get_key('junk.py')

def test_valid_key_last_used(manager):
    # Retrieve the key and then immediately record its usage
    key = manager.get_key('abc.py', 0)
    manager.record_key_usage(key)

    # Now, if we try to retrieve the same key, it should raise an error because it was just used
    with pytest.raises(ValueError):
        manager.get_key('abc.py', 0)


