import importlib

def torch_available() -> bool:
    """Return True if PyTorch can be imported."""
    try:
        importlib.import_module("torch")
        return True
    except Exception:
        return False
