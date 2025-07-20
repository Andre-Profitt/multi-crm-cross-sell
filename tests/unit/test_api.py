"""Unit tests for API endpoints"""
import pytest
from httpx import AsyncClient

from src.api.main import app


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_root_health(self):
        """Test root health endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
        assert response.status_code == 200


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_login_success(self):
        """Valid credentials should return a token"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/token",
                json={"username": "admin", "password": "password"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid(self):
        """Invalid credentials should return 401"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/token",
                json={"username": "bad", "password": "wrong"},
            )

        assert resp.status_code == 401
