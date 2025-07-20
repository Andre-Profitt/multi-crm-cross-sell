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

import asyncio
from unittest.mock import AsyncMock, patch

class TestSalesforceConnectorAuth:
    @pytest.mark.asyncio
    async def test_authenticate_success(self, tmp_path):
        cfg = SalesforceConfig(
            org_id="1",
            org_name="Test Org",
            instance_url="https://example.com",
            client_id="cid",
            username="user",
            password="pwd",
            security_token="tok",
            auth_type="password",
            private_key_path=str(tmp_path / "key.pem"),
        )
        connector = SalesforceConnector(cfg)

        async def mock_auth():
            connector._authenticated = True
            connector.config.access_token = "token"
            return True

        with patch.object(connector.token_manager, "get_token", AsyncMock(return_value=None)):
            with patch.object(connector, "_auth_password", side_effect=mock_auth):
                result = await connector.authenticate()

        assert result is True
        assert connector._authenticated
        assert connector.config.access_token == "token"

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, tmp_path):
        cfg = SalesforceConfig(
            org_id="1",
            org_name="Test Org",
            instance_url="https://example.com",
            client_id="cid",
            username="user",
            password="pwd",
            security_token="tok",
            auth_type="password",
            private_key_path=str(tmp_path / "key.pem"),
        )
        connector = SalesforceConnector(cfg)

        async def mock_auth():
            raise Exception("bad auth")

        with patch.object(connector.token_manager, "get_token", AsyncMock(return_value=None)):
            with patch.object(connector, "_auth_password", side_effect=mock_auth):
                result = await connector.authenticate()

        assert result is False
        assert connector._authenticated is False
