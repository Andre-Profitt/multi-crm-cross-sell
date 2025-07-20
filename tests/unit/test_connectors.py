"""Unit tests for CRM connectors."""
import pandas as pd
import pytest

from src.connectors.base import BaseCRMConnector, CRMConfig


class TestCRMConfig:
    def test_config_validation(self):
        """Ensure validation errors are raised for missing fields."""
        cfg = CRMConfig(org_id="1", org_name="Org", instance_url="https://ex.com")
        assert cfg.org_id == "1"

        with pytest.raises(ValueError):
            CRMConfig(org_id="", org_name="Org", instance_url="https://ex.com")

        with pytest.raises(ValueError):
            CRMConfig(org_id="1", org_name="Org", instance_url="")


class TestSalesforceConnector:
    def test_connector_init(self):
        """Verify BaseCRMConnector sets config and defaults."""

        class DummyConnector(BaseCRMConnector):
            async def authenticate(self) -> bool:
                return True

            async def test_connection(self) -> bool:
                return True

            async def extract_accounts(self, *_, **__) -> pd.DataFrame:
                return pd.DataFrame()

            async def extract_opportunities(self, *_, **__) -> pd.DataFrame:
                return pd.DataFrame()

            async def extract_products(self, *_, **__) -> pd.DataFrame:
                return pd.DataFrame()

            async def extract_contacts(self, *_, **__) -> pd.DataFrame:
                return pd.DataFrame()

        cfg = CRMConfig(org_id="1", org_name="Org", instance_url="https://ex.com")
        connector = DummyConnector(cfg)

        assert connector.config is cfg
        assert not connector._authenticated
