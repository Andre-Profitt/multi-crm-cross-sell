"""
Comprehensive test suite for Cross-Sell Intelligence Platform
Tests for connectors, orchestrator, API, and ML pipeline
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jwt
import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

from tests.utils import torch_available


# Test fixtures
@pytest.fixture
def salesforce_config():
    """Sample Salesforce configuration"""
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
    """Sample accounts DataFrame"""
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
    """Sample ML configuration"""
    from src.ml.pipeline import ModelConfig

    return ModelConfig()


# ============= Connector Tests =============


class TestSalesforceConnector:
    """Tests for Salesforce connector"""

    @pytest.mark.asyncio
    async def test_connector_initialization(self, salesforce_config):
        """Test connector can be initialized with config"""
        from src.connectors.salesforce import SalesforceConfig, SalesforceConnector

        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)

        assert connector.config.org_id == "test_org"
        assert connector.config.auth_type == "password"
        assert not connector._authenticated

    @pytest.mark.asyncio
    async def test_authentication_success(self, salesforce_config):
        """Test successful authentication"""
        from src.connectors.salesforce import SalesforceConfig, SalesforceConnector

        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)

        # Mock the session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "access_token": "test_token",
                "instance_url": "https://test.salesforce.com",
            }
        )

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
        """Test rate limiter functionality"""
        from src.connectors.salesforce import RateLimiter

        limiter = RateLimiter(calls_per_minute=60)

        # Should complete quickly for first few calls
        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant
        assert limiter.tokens < 60  # Tokens should be consumed

    @pytest.mark.asyncio
    async def test_data_extraction(self, salesforce_config, sample_accounts_df):
        """Test data extraction with mocked response"""
        from src.connectors.salesforce import SalesforceConfig, SalesforceConnector

        config = SalesforceConfig(**salesforce_config)
        connector = SalesforceConnector(config)
        connector._authenticated = True
        connector.config.access_token = "test_token"

        # Mock query response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"done": True, "records": sample_accounts_df.to_dict("records")}
        )

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


# ============= Orchestrator Tests =============


class TestOrchestrator:
    """Tests for the orchestrator"""

    def test_config_loading(self, tmp_path):
        """Test configuration loading"""
        from src.orchestrator import CrossSellOrchestrator

        # Create test config
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
        """Test full pipeline run"""
        from src.orchestrator import CrossSellOrchestrator

        # Create minimal config
        config_path = tmp_path / "test_orgs.json"
        config_path.write_text('{"organizations": []}')

        orchestrator = CrossSellOrchestrator(str(config_path))

        # Mock all the methods
        orchestrator._extract_all_org_data = AsyncMock(return_value={})
        orchestrator._process_data = AsyncMock(return_value={"org_data": {}})
        orchestrator._generate_recommendations = AsyncMock(return_value=pd.DataFrame())
        orchestrator._save_recommendations = AsyncMock()
        orchestrator._send_notifications = AsyncMock()
        orchestrator.initialize = AsyncMock()

        await orchestrator.run_pipeline()

        # Verify all steps were called
        orchestrator._extract_all_org_data.assert_called_once()
        orchestrator._process_data.assert_called_once()
        orchestrator._generate_recommendations.assert_called_once()
        orchestrator._save_recommendations.assert_called_once()

    def test_scheduler_setup(self):
        """Test scheduler configuration"""
        from src.orchestrator import CrossSellOrchestrator

        orchestrator = CrossSellOrchestrator()
        orchestrator.schedule_runs("daily")

        assert orchestrator.scheduler.running
        jobs = orchestrator.scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "cross_sell_pipeline"

        orchestrator.scheduler.shutdown()


# ============= ML Pipeline Tests =============


class TestMLPipeline:
    """Tests for ML components"""

    def test_feature_engineering(self, sample_accounts_df):
        """Test feature engineering"""
        from src.ml.pipeline import FeatureEngineering

        fe = FeatureEngineering()
        features = fe.create_account_features(sample_accounts_df)

        assert isinstance(features, pd.DataFrame)
        assert "revenue_log" in features.columns
        assert "employees_log" in features.columns
        assert "is_enterprise" in features.columns
        assert len(features) == len(sample_accounts_df)

    def test_cross_org_features(self, sample_accounts_df):
        """Test cross-org feature creation"""
        from src.ml.pipeline import FeatureEngineering

        fe = FeatureEngineering()
        account1 = sample_accounts_df.iloc[0]
        account2 = sample_accounts_df.iloc[1]

        features = fe.create_cross_org_features(account1, account2)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(isinstance(f, (int, float)) for f in features)

    def test_ensemble_scorer_init(self, ml_config):
        """Test ensemble scorer initialization"""
        from src.ml.pipeline import EnsembleScorer

        scorer = EnsembleScorer(ml_config)

        assert scorer.config == ml_config
        assert not scorer.is_trained
        assert len(scorer.models) == 0

    @pytest.mark.skipif(not torch_available(), reason="PyTorch not available")
    def test_neural_network_forward(self):
        """Test neural network forward pass"""
        import torch

        from src.ml.pipeline import CrossSellNeuralNetwork

        model = CrossSellNeuralNetwork(input_dim=10, hidden_layers=[32, 16])
        x = torch.randn(5, 10)

        output = model(x)

        assert output.shape == (5, 1)
        assert all(0 <= v <= 1 for v in output.numpy().flatten())

    # ============= API Tests =============

    """Tests for FastAPI endpoints"""

    @pytest.fixture
    async def client(self):
        """Create test client"""
        import os

        os.environ["API_RATE_LIMIT"] = "2/minute"
        from src.api.main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
        os.environ.pop("API_RATE_LIMIT", None)

    @pytest.fixture
    def auth_headers(self):
        """Generate auth headers with valid token"""
        from src.api.main import create_access_token

        token = create_access_token({"sub": "testuser", "scopes": ["read", "write"]})
        return {"Authorization": f"Bearer {token}"}

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint"""
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_auth_success(self, client):
        """Test successful authentication"""
        response = await client.post(
            "/api/auth/token", json={"username": "admin", "password": "password"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_auth_failure(self, client):
        """Test failed authentication"""
        response = await client.post(
            "/api/auth/token", json={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_recommendations_unauthorized(self, client):
        """Test recommendations endpoint without auth"""
        response = await client.get("/api/recommendations")
        assert response.status_code == 403  # Forbidden without token

    @pytest.mark.asyncio
    async def test_get_recommendations_authorized(self, client, auth_headers):
        """Test recommendations endpoint with auth"""
        with patch("src.api.main.get_db") as mock_db:
            # Mock database response
            mock_session = AsyncMock()
            mock_result = AsyncMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_db.return_value = mock_session

            response = await client.get("/api/recommendations", headers=auth_headers)
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_score_opportunity(self, client, auth_headers):
        """Test opportunity scoring endpoint"""
        with patch("src.api.main.orchestrator") as mock_orch:
            mock_orch.feature_engineer.create_cross_org_features.return_value = np.array([0.5] * 6)
            mock_orch.scorer.is_trained = True
            mock_orch.scorer.predict.return_value = (np.array([0.75]), {})

            request_data = {
                "account1": {"Name": "Test1", "Industry": "Tech"},
                "account2": {"Name": "Test2", "Industry": "Finance"},
                "include_explanation": True,
            }

            response = await client.post("/api/score", json=request_data, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert "score" in data
            assert 0 <= data["score"] <= 1
            assert "confidence_level" in data

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, client, auth_headers):
        """Requests beyond limit should return 429"""
        for _ in range(2):
            response = await client.get("/api/recommendations", headers=auth_headers)
            assert response.status_code in (200, 429)
        # Third request exceeds 2/minute limit
        response = await client.get("/api/recommendations", headers=auth_headers)
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_health_exempt_from_rate_limit(self, client):
        """Health endpoints are not rate limited"""
        for _ in range(5):
            r1 = await client.get("/")
            r2 = await client.get("/api/health")
            assert r1.status_code == 200
            assert r2.status_code == 200


# ============= Integration Tests =============


class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_with_mocked_salesforce(self, tmp_path):
        """Test full pipeline with mocked Salesforce"""
        from src.connectors.base import ConnectorRegistry
        from src.orchestrator import CrossSellOrchestrator

        # Create test config
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

        # Mock Salesforce responses
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


# ============= Test Configuration =============

# pytest.ini content:
"""
[pytest]
minversion = 6.0
addopts = -ra -q --strict-markers
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
asyncio_mode = auto
"""

# conftest.py content:
"""
import pytest
import asyncio
import os

# Set test environment
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///:memory:'

@pytest.fixture(scope="session")
def event_loop():
    '''Create an instance of the default event loop for the test session.'''
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
"""
