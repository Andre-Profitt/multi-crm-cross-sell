"""
Machine Learning Pipeline for Cross-Sell Opportunity Scoring
Implements ensemble models for identifying high-value cross-sell opportunities
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .nlp_features import DescriptionEmbedder

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models"""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Neural Network
    nn_hidden_layers: List[int] = None
    nn_dropout_rate: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 50

    # Ensemble weights
    ensemble_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [64, 32, 16]
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "neural_net": 0.3,
                "xgboost": 0.3,
                "random_forest": 0.2,
                "gradient_boost": 0.2,
            }


class FeatureEngineering:
    """Feature engineering for cross-sell opportunities"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.embedder = DescriptionEmbedder()
        self.description_dim = self.embedder.dim

    def create_account_features(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from account data"""
        features = pd.DataFrame()

        # Revenue features
        features["revenue_log"] = np.log1p(accounts_df["AnnualRevenue"].fillna(0))
        features["revenue_per_employee"] = (
            accounts_df["AnnualRevenue"] / accounts_df["NumberOfEmployees"].replace(0, 1)
        ).fillna(0)

        # Size features
        features["employees_log"] = np.log1p(accounts_df["NumberOfEmployees"].fillna(0))
        features["is_enterprise"] = (accounts_df["NumberOfEmployees"] > 1000).astype(int)

        # Time-based features
        features["company_age_days"] = (
            datetime.now() - pd.to_datetime(accounts_df["CreatedDate"])
        ).dt.days

        features["days_since_last_activity"] = (
            datetime.now() - pd.to_datetime(accounts_df["LastActivityDate"])
        ).dt.days.fillna(365)

        # Activity features
        features["activity_recency_score"] = 1 / (1 + features["days_since_last_activity"] / 30)

        # Industry encoding
        if "Industry" in accounts_df.columns:
            industry_dummies = pd.get_dummies(
                accounts_df["Industry"].fillna("Unknown"), prefix="industry"
            )
            features = pd.concat([features, industry_dummies], axis=1)

        # Geographic features
        if "BillingCountry" in accounts_df.columns:
            country_dummies = pd.get_dummies(
                accounts_df["BillingCountry"].fillna("Unknown"), prefix="country"
            )
            features = pd.concat([features, country_dummies], axis=1)

        # Description embeddings
        if "Description" in accounts_df.columns:
            desc_texts = accounts_df["Description"].fillna("").astype(str).tolist()
            embeddings = self.embedder.encode_batch(desc_texts)
        else:
            embeddings = np.zeros((len(accounts_df), self.description_dim))

        for i in range(self.description_dim):
            features[f"desc_emb_{i}"] = embeddings[:, i]

        self.feature_names = features.columns.tolist()
        return features

    def create_cross_org_features(
        self, account1: pd.Series, account2: pd.Series, additional_data: Dict = None
    ) -> np.ndarray:
        """Create features for cross-org opportunity scoring"""
        features = []

        # Industry similarity
        features.append(1.0 if account1.get("Industry") == account2.get("Industry") else 0.0)

        # Size compatibility
        size_ratio = account1.get("NumberOfEmployees", 1) / account2.get("NumberOfEmployees", 1)
        features.append(1.0 / (1.0 + abs(np.log(size_ratio))))

        # Geographic proximity
        same_country = account1.get("BillingCountry") == account2.get("BillingCountry")
        features.append(1.0 if same_country else 0.3)

        # Product complementarity
        if additional_data and "products_org1" in additional_data:
            products1 = set(additional_data.get("products_org1", []))
            products2 = set(additional_data.get("products_org2", []))

            overlap = len(products1.intersection(products2))
            unique = len(products1.symmetric_difference(products2))

            features.append(unique / (overlap + unique + 1))
        else:
            features.append(0.5)

        # Customer maturity alignment
        age1 = (datetime.now() - pd.to_datetime(account1.get("CreatedDate", datetime.now()))).days
        age2 = (datetime.now() - pd.to_datetime(account2.get("CreatedDate", datetime.now()))).days

        age_ratio = min(age1, age2) / max(age1, age2)
        features.append(age_ratio)

        # Activity alignment
        activity1 = (
            datetime.now()
            - pd.to_datetime(account1.get("LastActivityDate", datetime.now() - timedelta(days=365)))
        ).days
        activity2 = (
            datetime.now()
            - pd.to_datetime(account2.get("LastActivityDate", datetime.now() - timedelta(days=365)))
        ).days

        activity_score = 2.0 / (1.0 + activity1 / 30 + activity2 / 30)
        features.append(activity_score)

        return np.array(features)


class CrossSellNeuralNetwork(nn.Module):
    """Neural network for cross-sell scoring"""

    def __init__(self, input_dim: int, hidden_layers: List[int] = None, dropout_rate: float = 0.3):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EnsembleScorer:
    """Ensemble model for cross-sell opportunity scoring"""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train all models in the ensemble"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        metrics = {}

        # Train Neural Network
        logger.info("Training Neural Network...")
        self.models["neural_net"] = self._train_neural_net(X_train, y_train, X_val, y_val)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self.models["xgboost"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.config.random_state
        )
        self.models["xgboost"].fit(X_train, y_train)
        metrics["xgboost_val_score"] = self.models["xgboost"].score(X_val, y_val)

        # Train Random Forest
        logger.info("Training Random Forest...")
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=self.config.random_state
        )
        self.models["random_forest"].fit(X_train, y_train)
        metrics["rf_val_score"] = self.models["random_forest"].score(X_val, y_val)

        # Train Gradient Boosting
        logger.info("Training Gradient Boosting...")
        self.models["gradient_boost"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.config.random_state
        )
        self.models["gradient_boost"].fit(X_train, y_train)
        metrics["gb_val_score"] = self.models["gradient_boost"].score(X_val, y_val)

        self.is_trained = True
        logger.info(f"Training complete. Metrics: {metrics}")

        return metrics

    def _train_neural_net(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> nn.Module:
        """Train neural network model"""
        input_dim = X_train.shape[1]
        model = CrossSellNeuralNetwork(
            input_dim, self.config.nn_hidden_layers, self.config.nn_dropout_rate
        )

        optimizer = optim.Adam(model.parameters(), lr=self.config.nn_learning_rate)
        criterion = nn.BCELoss()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        # Training loop (simplified for brevity)
        model.train()
        for epoch in range(self.config.nn_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions using the ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = {}

        # Get predictions from each model
        with torch.no_grad():
            nn_input = torch.FloatTensor(X)
            predictions["neural_net"] = self.models["neural_net"](nn_input).numpy().flatten()

        predictions["xgboost"] = self.models["xgboost"].predict_proba(X)[:, 1]
        predictions["random_forest"] = self.models["random_forest"].predict_proba(X)[:, 1]
        predictions["gradient_boost"] = self.models["gradient_boost"].predict_proba(X)[:, 1]

        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for model_name, weight in self.config.ensemble_weights.items():
            ensemble_pred += weight * predictions[model_name]

        return ensemble_pred, predictions


class CrossSellRecommendationEngine:
    """Main recommendation engine combining all components"""

    def __init__(self, scorer: EnsembleScorer):
        self.scorer = scorer
        self.feature_engineer = FeatureEngineering()

    def generate_recommendations(self, org_data: Dict[str, Dict]) -> pd.DataFrame:
        """Generate cross-sell recommendations across organizations"""
        recommendations = []

        # Get all org pairs
        org_ids = list(org_data.keys())
        for i in range(len(org_ids)):
            for j in range(i + 1, len(org_ids)):
                org1_id, org2_id = org_ids[i], org_ids[j]

                # Generate recommendations between these orgs
                org_recommendations = self._generate_org_pair_recommendations(
                    org1_id, org_data[org1_id], org2_id, org_data[org2_id]
                )

                recommendations.extend(org_recommendations)

        # Convert to DataFrame and rank
        recommendations_df = pd.DataFrame(recommendations)
        if not recommendations_df.empty:
            recommendations_df = recommendations_df.sort_values("score", ascending=False)
            recommendations_df["rank"] = range(1, len(recommendations_df) + 1)

        return recommendations_df

    def _generate_org_pair_recommendations(
        self, org1_id: str, org1_data: Dict, org2_id: str, org2_data: Dict
    ) -> List[Dict]:
        """Generate recommendations between two organizations"""
        recommendations = []

        accounts1 = org1_data["accounts"]
        accounts2 = org2_data["accounts"]

        # Limit to top accounts for performance
        top_accounts1 = accounts1.nlargest(100, "AnnualRevenue", "all")
        top_accounts2 = accounts2.nlargest(100, "AnnualRevenue", "all")

        for _, acc1 in top_accounts1.iterrows():
            for _, acc2 in top_accounts2.iterrows():
                # Create features
                features = self.feature_engineer.create_cross_org_features(
                    acc1,
                    acc2,
                    {
                        "products_org1": org1_data.get("products", []),
                        "products_org2": org2_data.get("products", []),
                    },
                )

                # Score opportunity
                score, individual_scores = self.scorer.predict(features.reshape(1, -1))

                if score[0] > 0.5:  # Only keep promising opportunities
                    recommendation = {
                        "org1_id": org1_id,
                        "org1_account_id": acc1["Id"],
                        "org1_account_name": acc1["Name"],
                        "org2_id": org2_id,
                        "org2_account_id": acc2["Id"],
                        "org2_account_name": acc2["Name"],
                        "score": score[0],
                        "confidence_level": self._calculate_confidence_level(
                            score[0], individual_scores
                        ),
                        "recommendation_type": self._determine_recommendation_type(acc1, acc2),
                        "estimated_value": self._estimate_opportunity_value(acc1, acc2),
                        "next_best_action": self._suggest_next_action(acc1, acc2, score[0]),
                        "created_at": datetime.now(),
                    }

                    recommendations.append(recommendation)

        return recommendations

    def _calculate_confidence_level(
        self, score: float, individual_scores: Dict[str, np.ndarray]
    ) -> str:
        """Calculate confidence level based on score consensus"""
        scores = [s[0] for s in individual_scores.values()]
        std_dev = np.std(scores)

        if score > 0.8 and std_dev < 0.1:
            return "Very High"
        elif score > 0.7 and std_dev < 0.15:
            return "High"
        elif score > 0.6:
            return "Medium"
        else:
            return "Low"

    def _determine_recommendation_type(self, acc1: pd.Series, acc2: pd.Series) -> str:
        """Determine the type of cross-sell recommendation"""
        if acc1.get("Industry") == acc2.get("Industry"):
            return "Industry Expansion"
        elif acc1.get("Type") == "Partner" or acc2.get("Type") == "Partner":
            return "Partner Referral"
        else:
            return "Market Development"

    def _estimate_opportunity_value(self, acc1: pd.Series, acc2: pd.Series) -> float:
        """Estimate potential opportunity value"""
        # Simple heuristic: 10% of average revenue
        avg_revenue = (acc1.get("AnnualRevenue", 0) + acc2.get("AnnualRevenue", 0)) / 2
        return avg_revenue * 0.1

    def _suggest_next_action(self, acc1: pd.Series, acc2: pd.Series, score: float) -> str:
        """Suggest next best action"""
        if score > 0.8:
            return "Schedule executive introduction call immediately"
        elif score > 0.7:
            return "Prepare joint value proposition and reach out"
        elif score > 0.6:
            return "Add to nurture campaign and monitor engagement"
        else:
            return "Review quarterly for status changes"
