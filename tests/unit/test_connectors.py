"""Unit tests for CRM connectors"""
import pytest

from src.connectors.base import BaseCRMConnector, CRMConfig
from src.connectors.salesforce import SalesforceConfig, SalesforceConnector


class TestCRMConfig:
    def test_config_validation(self):
        """Test config validation"""
        # Missing org_id should raise ValueError
        with pytest.raises(ValueError):
            CRMConfig(org_id="", org_name="Org", instance_url="https://example.com")

        # Missing instance_url should raise ValueError
        with pytest.raises(ValueError):
            CRMConfig(org_id="1", org_name="Org", instance_url="")

        cfg = CRMConfig(org_id="1", org_name="Org", instance_url="https://example.com")
        assert cfg.org_id == "1"
        assert cfg.instance_url.startswith("https://")


class TestSalesforceConnector:
    def test_connector_init(self):
        """Test connector initialization"""
        cfg = SalesforceConfig(
            org_id="1",
            org_name="Test Org",
            instance_url="https://example.com",
            client_id="cid",
            username="user",
            private_key_path="/tmp/key.pem",
        )

        connector = SalesforceConnector(cfg)

        assert isinstance(connector, BaseCRMConnector)
        assert connector.config.org_name == "Test Org"
        assert not connector._authenticated
