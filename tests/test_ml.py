import numpy as np
import pandas as pd
import pytest
import types
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - create a lightweight stub
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.SimpleNamespace(Module=object, Sequential=lambda *a, **k: None, Linear=lambda *a, **k: None, Sigmoid=lambda: None)
    torch_stub.optim = types.SimpleNamespace(Adam=lambda *a, **k: None)
    torch_stub.FloatTensor = lambda *a, **k: None
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn
    sys.modules["torch.optim"] = torch_stub.optim

from src.ml.pipeline import FeatureEngineering, EnsembleScorer


def torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


class TestMLPipeline:
    """Tests for ML components."""

    def test_feature_engineering(self, sample_accounts_df):
        fe = FeatureEngineering()
        features = fe.create_account_features(sample_accounts_df)

        assert isinstance(features, pd.DataFrame)
        assert "revenue_log" in features.columns
        assert "employees_log" in features.columns
        assert "is_enterprise" in features.columns
        assert len(features) == len(sample_accounts_df)

    def test_cross_org_features(self, sample_accounts_df):
        fe = FeatureEngineering()
        account1 = sample_accounts_df.iloc[0]
        account2 = sample_accounts_df.iloc[1]

        features = fe.create_cross_org_features(account1, account2)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(isinstance(f, (int, float)) for f in features)

    def test_ensemble_scorer_init(self, ml_config):
        scorer = EnsembleScorer(ml_config)

        assert scorer.config == ml_config
        assert not scorer.is_trained
        assert len(scorer.models) == 0

    @pytest.mark.skipif(not torch_available(), reason="PyTorch not available")
    def test_neural_network_forward(self):
        import torch
        from src.ml.pipeline import CrossSellNeuralNetwork

        model = CrossSellNeuralNetwork(input_dim=10, hidden_layers=[32, 16])
        x = torch.randn(5, 10)

        output = model(x)

        assert output.shape == (5, 1)
        assert all(0 <= v <= 1 for v in output.detach().numpy().flatten())
