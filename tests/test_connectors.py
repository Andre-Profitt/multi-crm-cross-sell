import asyncio
import json
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.connectors.salesforce import SalesforceConfig, SalesforceConnector, RateLimiter


class TestSalesforceConnector:
    """Tests for Salesforce connector."""

    @pytest.mark.asyncio
    async def test_connector_initialization(self, salesforce_config):
        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)

        assert connector.config.org_id == "test_org"
        assert connector.config.auth_type == "password"
        assert not connector._authenticated

    @pytest.mark.asyncio
    async def test_authentication_success(self, salesforce_config):
        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "access_token": "test_token",
            "instance_url": "https://test.salesforce.com",
        })

        with patch.object(connector, "_session") as mock_session:
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            result = await connector.authenticate()

            assert result is True
            assert connector._authenticated is True
            assert connector.config.access_token == "test_token"

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        limiter = RateLimiter(calls_per_minute=60)

        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1
        assert limiter.tokens < 60

    @pytest.mark.asyncio
    async def test_data_extraction(self, salesforce_config, sample_accounts_df):
        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)
        connector._authenticated = True
        connector.config.access_token = "test_token"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"done": True, "records": sample_accounts_df.to_dict("records")})

        with patch.object(connector, "_get_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_ctx.return_value = mock_session

            result = await connector.extract_accounts()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "_org_id" in result.columns
            assert result["_org_id"].iloc[0] == "test_org"


class TestOrchestrator:
    """Tests for the orchestrator."""

    def test_config_loading(self, tmp_path):
        pytest.importorskip("torch")
        from src.orchestrator import CrossSellOrchestrator
        config_path = tmp_path / "test_orgs.json"
        config_data = {
            "organizations": [
                {
                    "org_id": "org1",
                    "org_name": "Test Org 1",
                    "instance_url": "https://test1.salesforce.com",
                }
            ]
        }
        config_path.write_text(json.dumps(config_data))

        orchestrator = CrossSellOrchestrator(str(config_path))
        assert len(orchestrator.orgs) == 1
        assert orchestrator.orgs[0]["org_id"] == "org1"

    @pytest.mark.asyncio
    async def test_pipeline_run(self, tmp_path):
        pytest.importorskip("torch")
        from src.orchestrator import CrossSellOrchestrator
        config_path = tmp_path / "test_orgs.json"
        config_path.write_text('{"organizations": []}')

        orchestrator = CrossSellOrchestrator(str(config_path))

        orchestrator._extract_all_org_data = AsyncMock(return_value={})
        orchestrator._process_data = AsyncMock(return_value={"org_data": {}})
        orchestrator._generate_recommendations = AsyncMock(return_value=pd.DataFrame())
        orchestrator._save_recommendations = AsyncMock()
        orchestrator._send_notifications = AsyncMock()
        orchestrator.initialize = AsyncMock()

        await orchestrator.run_pipeline()

        orchestrator._extract_all_org_data.assert_called_once()
        orchestrator._process_data.assert_called_once()
        orchestrator._generate_recommendations.assert_called_once()
        orchestrator._save_recommendations.assert_called_once()

    def test_scheduler_setup(self):
        pytest.importorskip("torch")
        from src.orchestrator import CrossSellOrchestrator

        orchestrator = CrossSellOrchestrator()
        orchestrator.schedule_runs("daily")

        assert orchestrator.scheduler.running
        jobs = orchestrator.scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "cross_sell_pipeline"

        orchestrator.scheduler.shutdown()
