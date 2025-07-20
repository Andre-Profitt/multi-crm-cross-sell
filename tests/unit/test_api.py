"""Unit tests for API endpoints."""
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
    async def test_login(self):
        """Test login endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/token",
                json={"username": "admin", "password": "password"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
