import numpy as np
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from src.api.main import app, create_access_token


class TestAPI:
    """Tests for FastAPI endpoints."""

    @pytest_asyncio.fixture
    async def client(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def auth_headers(self):
        token = create_access_token({"sub": "testuser", "scopes": ["read", "write"]})
        return {"Authorization": f"Bearer {token}"}

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_auth_success(self, client):
        response = await client.post(
            "/api/auth/token", json={"username": "admin", "password": "password"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    @pytest.mark.asyncio
    async def test_auth_failure(self, client):
        response = await client.post(
            "/api/auth/token", json={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_recommendations_unauthorized(self, client):
        response = await client.get("/api/recommendations")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_recommendations_authorized(self, client, auth_headers):
        with patch("src.api.main.get_db") as mock_db:
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
