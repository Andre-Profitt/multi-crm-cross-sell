import asyncio
import os

import pandas as pd
import pytest

# Set test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def salesforce_config():
    """Sample Salesforce configuration."""
    return {
        "org_id": "test_org",
        "org_name": "Test Organization",
        "instance_url": "https://test.salesforce.com",
        "auth_type": "password",
        "client_id": "test_client_id",
        "client_secret": "test_secret",
        "username": "test@example.com",
        "password": "testpass",
        "security_token": "testtoken",
    }


@pytest.fixture
def sample_accounts_df():
    """Return a sample accounts DataFrame."""
    return pd.DataFrame(
        {
            "Id": ["001xx000003DHP0", "001xx000003DHP1"],
            "Name": ["Acme Corp", "Global Tech Inc"],
            "Industry": ["Technology", "Finance"],
            "AnnualRevenue": [1000000, 5000000],
            "NumberOfEmployees": [100, 500],
            "BillingCountry": ["USA", "USA"],
            "CreatedDate": ["2023-01-01", "2023-06-01"],
            "LastActivityDate": ["2024-01-01", "2024-06-01"],
        }
    )


@pytest.fixture
def ml_config():
    """Return a default ML ModelConfig."""
    from src.ml.pipeline import ModelConfig

    return ModelConfig()
