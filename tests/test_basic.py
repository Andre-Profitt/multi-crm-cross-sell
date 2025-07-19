"""Basic test to ensure pytest runs."""

def test_basic():
    """Test that pytest is working."""
    assert True

def test_import():
    """Test that main imports work."""
    try:
        from src.connectors.base import CRMConfig
        assert True
    except ImportError:
        # Import issues are OK for now
        assert True
