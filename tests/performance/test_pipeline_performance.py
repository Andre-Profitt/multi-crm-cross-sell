"""Simple performance tests for the orchestrator pipeline."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.orchestrator import CrossSellOrchestrator


@pytest.mark.asyncio
@pytest.mark.performance
async def test_pipeline_run_time(tmp_path: Path) -> None:
    """Ensure pipeline completes quickly with mocked connectors."""
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
        "accounts": pd.DataFrame(),
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
        start = time.time()
        await orchestrator.run_pipeline()
        duration = time.time() - start

    assert duration < 2
    assert orchestrator.last_run_status.get("status") == "success"
