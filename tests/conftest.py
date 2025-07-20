"""Test configuration and lightweight stubs for optional dependencies."""

import asyncio
import os
import sys
import types

import pytest

# Set test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

# Provide a minimal torch stub if PyTorch is not installed so modules importing
# torch during test collection do not fail.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.ModuleType("torch.nn")
    torch_stub.optim = types.ModuleType("torch.optim")
    torch_stub.FloatTensor = lambda *args, **kwargs: None

    class _Module:
        pass

    class _Sequential(_Module):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _Layer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _Adam:
        def __init__(self, *args, **kwargs):
            pass

    torch_stub.nn.Module = _Module
    torch_stub.nn.Sequential = _Sequential
    torch_stub.nn.Linear = _Layer
    torch_stub.nn.ReLU = _Layer
    torch_stub.nn.BatchNorm1d = _Layer
    torch_stub.nn.Dropout = _Layer
    torch_stub.nn.Sigmoid = _Layer
    torch_stub.optim.Adam = _Adam

    def no_grad():
        """Context manager stub used in place of torch.no_grad."""

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return _Ctx()

    torch_stub.no_grad = no_grad
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn
    sys.modules["torch.optim"] = torch_stub.optim

# Stub out heavy optional dependencies to avoid installation in CI
for missing_pkg in ["shap", "xgboost"]:
    if missing_pkg not in sys.modules:
        sys.modules[missing_pkg] = types.ModuleType(missing_pkg)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
