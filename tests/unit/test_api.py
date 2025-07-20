"""Unit tests for API endpoints"""
import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from src.api.main import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    app,
    create_access_token,
    get_db,
)


@asynccontextmanager
async def _no_lifespan(app):
    yield


app.router.lifespan_context = _no_lifespan


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_root_health(self):
        """Test root health endpoint"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
        assert response.status_code == 200


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_login_success(self):
        """Valid credentials should return a token"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/token",
                json={"username": "admin", "password": "password"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

        payload = jwt.decode(data["access_token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == "admin"

    @pytest.mark.asyncio
    async def test_login_invalid(self):
        """Invalid credentials should return 401"""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/token",
                json={"username": "bad", "password": "wrong"},
            )

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Incorrect username or password"

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_token(self):
        """Requests without a token should be rejected"""
        mock_session = AsyncMock()
        transport = ASGITransport(app=app)

        async def override_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_db
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recommendations")
        app.dependency_overrides.clear()
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_token(self):
        """Valid token should allow access"""
        token = create_access_token({"sub": "admin"})
        headers = {"Authorization": f"Bearer {token}"}

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        transport = ASGITransport(app=app)

        async def override_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_db
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recommendations", headers=headers)
        app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert resp.json() == []
