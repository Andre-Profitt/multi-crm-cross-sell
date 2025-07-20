import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from src.orchestrator import CrossSellOrchestrator


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_with_mocked_salesforce(self, tmp_path):
        config_path = tmp_path / "test_orgs.json"
        config_data = {
            "organizations": [
                {
                    "org_id": "test_org",
                    "org_name": "Test Org",
                    "instance_url": "https://test.salesforce.com",
                    "crm_type": "salesforce",
                }
            ]
        }
        config_path.write_text(json.dumps(config_data))

        with patch("src.connectors.salesforce.SalesforceConnector.extract_all") as mock_extract:
            mock_extract.return_value = {
                "accounts": pd.DataFrame(
                    {
                        "Id": ["001", "002"],
                        "Name": ["Company A", "Company B"],
                        "AnnualRevenue": [1000000, 2000000],
                    }
                ),
                "opportunities": pd.DataFrame(),
                "products": pd.DataFrame(),
                "contacts": pd.DataFrame(),
            }

            orchestrator = CrossSellOrchestrator(str(config_path))
            await orchestrator.initialize()
            await orchestrator.run_pipeline()

            assert orchestrator.last_run_status["status"] == "success"
