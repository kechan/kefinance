import pytest
from datetime import datetime, timedelta
import pandas as pd
from kefinance import APIKeyManager  # replace "your_module" with the actual module name where APIKeyManager is defined

@pytest.fixture(scope='function')
def manager():
    m = APIKeyManager(key_store='data/keys_df')
    reset_key(m)
    yield m
    reset_key(m)

def reset_key(manager):

    manager._read_keystore()

    # put key back to prinstine state
    # key = manager.api_keys_df.loc[0]['value']  # this need to change is >1 key in df
    for k, row in manager.api_keys_df.iterrows():
        key = row['value']

        past_time = datetime.now() - timedelta(days=2)  # more than 24 hours ago
        manager.record_key_usage(key, last_used=past_time)
        manager.unlock(key)

    # Ensure the key can be retrieved again after the reset
    # assert key == manager.get_available_key()

def test_test_keys_df_datashape(manager):
    # if data shape failed, it can lead to hard to track test failures.
    assert manager.api_keys_df.shape == (2, 5)

def test_valid_key_retrieval(manager):
    key = manager.get_available_key()

    # This assertion checks that the returned key is a non-empty string
    # If you know what the key looks like, you can be more specific in this test
    assert isinstance(key, str) and len(key) > 0


def test_record_key_usage(manager):
    # Retrieve the key and then immediately record its usage
    key = manager.get_available_key()
    manager.record_key_usage(key)

    # key should now be unavailable
    assert not manager.is_key_available(key)

def test_record_key_usage_custom_timestamp(manager):
    key = manager.get_available_key()
    past_time = datetime.now() - timedelta(days=2, hours=1)  # more than 24 hours ago
    manager.record_key_usage(key, last_used=past_time)

    # Despite just recording its usage, since we used a timestamp of more than 24 hours ago, 
    # the key should still be available
    assert manager.is_key_available(key)
    


def test_lock_unlock_key(manager):
    # Lock a known key and then try to retrieve it. It should raise an error.
    key = manager.get_available_key()
    manager.lock(key)

    # key should become unavailable
    assert not manager.is_key_available(key)
    
    # Now, unlock the key and confirm its availability
    manager.unlock(key)
    assert manager.is_key_available(key)


def test_key_retrieval_without_script_name(manager):
    # Retrieve any available key
    key = manager.get_available_key()

    # This assertion checks that the returned key is a non-empty string
    assert isinstance(key, str) and len(key) > 0

def test_record_key_usage_basic(manager):
    key = manager.get_available_key()
    manager.record_key_usage(key)

    key_entry = manager.api_keys_df[manager.api_keys_df['value'] == key].iloc[0]
    
    # Only the timestamp 'last_used' should be updated and 'last_usage' should remain None.
    assert key_entry['last_usage'] is None

def test_record_key_usage_script_name(manager):
    key = manager.get_available_key()
    script_name = "test_script.py"
    manager.record_key_usage(key, script_name=script_name)
    
    key_entry = manager.api_keys_df[manager.api_keys_df['value'] == key].iloc[0]
    
    assert key_entry['last_usage'] == script_name


def test_record_key_usage_script_and_part(manager):
    key = manager.get_available_key()
    script_name = "test_script.py"
    part_number = 3
    manager.record_key_usage(key, script_name=script_name, part_number=part_number)
    
    key_entry = manager.api_keys_df[manager.api_keys_df['value'] == key].iloc[0]
    
    assert key_entry['last_usage'] == f"{script_name} part {part_number}"

def test_multiple_keys_retrieval(manager):
    # Assumption: Your df has at least two keys to begin with, 
    # and the reset_key function resets both keys.
    
    # Use the first key
    key1 = manager.get_available_key()
    manager.record_key_usage(key1)

    # Now, the first key has been used and its timestamp is less than 24 hours.
    # Therefore, the next call to get_available_key should fetch the next key.

    key2 = manager.get_available_key()
    
    assert key1 != key2  # Ensure the two keys are different

    # Additional checks can be added to ensure key2 hasn't been used in the last 24 hours, 
    # and so on based on your application's requirements.


