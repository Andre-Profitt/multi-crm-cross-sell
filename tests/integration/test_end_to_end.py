"""Integration tests exercising orchestrator and Salesforce connector."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.connectors.salesforce import SalesforceConfig, SalesforceConnector
from src.orchestrator import CrossSellOrchestrator


@pytest.mark.asyncio
async def test_salesforce_extract_all(tmp_path: Path) -> None:
    """End-to-end extraction using mocked API calls."""
    config = SalesforceConfig(
        org_id="1",
        org_name="Test Org",
        instance_url="https://example.com",
        client_id="cid",
        username="user",
        private_key_path=str(tmp_path / "key.pem"),
    )
    connector = SalesforceConnector(config)
    connector._authenticated = True
    connector.config.access_token = "token"

    sample = pd.DataFrame({"Id": ["1"], "Name": ["Acme"]})
    with patch.object(
        SalesforceConnector, "_query_rest", AsyncMock(return_value=sample)
    ), patch.object(SalesforceConnector, "_query_bulk", AsyncMock(return_value=sample)):
        data = await connector.extract_all()

    assert set(data.keys()) == {"accounts", "opportunities", "products", "contacts"}
    for df in data.values():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_with_mocked_salesforce(tmp_path: Path) -> None:
    """Run orchestrator pipeline with mocked Salesforce connector."""
    cfg_path = tmp_path / "orgs.json"
    cfg_path.write_text(
        json.dumps(
            {
                "organizations": [
                    {
                        "org_id": "o1",
                        "org_name": "Test Org",
                        "instance_url": "https://example.com",
                        "crm_type": "salesforce",
                    }
                ]
            }
        )
    )

    orchestrator = CrossSellOrchestrator(str(cfg_path))
    orchestrator.initialize = AsyncMock()
    orchestrator.scorer = MagicMock(is_trained=True)
    await orchestrator.initialize()

    mock_data = {
        "accounts": pd.DataFrame({"Id": ["1"], "Name": ["Acme"]}),
        "opportunities": pd.DataFrame(),
        "products": pd.DataFrame(),
        "contacts": pd.DataFrame(),
    }

    with patch(
        "src.connectors.salesforce.SalesforceConnector.extract_all",
        AsyncMock(return_value=mock_data),
    ), patch.object(orchestrator, "_process_data", AsyncMock(return_value={})), patch.object(
        orchestrator,
        "_generate_recommendations",
        AsyncMock(return_value=pd.DataFrame()),
    ), patch.object(
        orchestrator, "_save_recommendations", AsyncMock()
    ), patch.object(
        orchestrator, "_send_notifications", AsyncMock()
    ):
        await orchestrator.run_pipeline()

    assert orchestrator.last_run_status.get("status") == "success"
