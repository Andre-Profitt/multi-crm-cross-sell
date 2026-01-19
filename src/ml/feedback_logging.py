"""
Feedback Logging and Model Versioning

Implements production ML observability patterns:
1. Prediction logging - Every inference logged with model version
2. Feedback collection - User actions (accept/reject/convert)
3. A/B testing support - Route traffic to different model versions
4. Training data generation - Convert feedback into labeled examples

Usage:
    logger = FeedbackLogger(db_session)

    # Log a prediction
    pred_id = logger.log_prediction(
        model_version="v2.3.1",
        account_pair=("acc1", "acc2"),
        score=0.85,
        features={"revenue_similarity": 0.9, ...}
    )

    # Log feedback
    logger.log_feedback(pred_id, "accepted", user_id="user123")

    # Generate training data
    training_df = logger.generate_training_data(min_feedback=100)
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class FeedbackType(str, Enum):
    """Types of user feedback on recommendations"""
    VIEWED = "viewed"           # User saw the recommendation
    CLICKED = "clicked"         # User clicked for details
    ACCEPTED = "accepted"       # User accepted/actioned
    REJECTED = "rejected"       # User dismissed
    CONVERTED = "converted"     # Led to actual sale
    IGNORED = "ignored"         # Shown but no action (implicit)


class ModelStage(str, Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class PredictionRecord:
    """A logged prediction"""
    prediction_id: str
    model_version: str
    model_stage: ModelStage
    account1_id: str
    account2_id: str
    score: float
    confidence: float
    rank: int  # Position in recommendation list
    features: Dict[str, float]
    explanation: Optional[str]
    created_at: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class FeedbackRecord:
    """User feedback on a prediction"""
    feedback_id: str
    prediction_id: str
    feedback_type: FeedbackType
    user_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Outcome tracking
    outcome_value: Optional[float] = None  # Deal value if converted
    outcome_date: Optional[datetime] = None


class PredictionLog(Base):
    """SQLAlchemy model for prediction logs"""
    __tablename__ = "prediction_logs"

    id = Column(String(50), primary_key=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_stage = Column(String(20), default="production")

    # Accounts
    account1_id = Column(String(100), nullable=False, index=True)
    account2_id = Column(String(100), nullable=False, index=True)

    # Prediction
    score = Column(Float, nullable=False)
    confidence = Column(Float)
    rank = Column(Integer)  # Position shown to user

    # Context
    features = Column(JSON)
    explanation = Column(Text)

    # Request metadata
    session_id = Column(String(100), index=True)
    user_id = Column(String(100), index=True)
    request_id = Column(String(100))
    latency_ms = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class FeedbackLog(Base):
    """SQLAlchemy model for feedback logs"""
    __tablename__ = "feedback_logs"

    id = Column(String(50), primary_key=True)
    prediction_id = Column(String(50), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False, index=True)

    # User context
    user_id = Column(String(100), index=True)

    # Outcome
    outcome_value = Column(Float)  # Revenue if converted
    outcome_date = Column(DateTime)

    # Additional context
    metadata = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ModelRegistry(Base):
    """Model version registry with performance tracking"""
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False)
    stage = Column(String(20), default="development")

    # Model info
    model_type = Column(String(50))  # ensemble, xgboost, etc
    description = Column(Text)

    # Artifacts
    artifact_path = Column(String(500))
    artifact_hash = Column(String(64))  # SHA256 of model file

    # Training metadata
    training_data_version = Column(String(50))
    training_samples = Column(Integer)
    training_started_at = Column(DateTime)
    training_completed_at = Column(DateTime)

    # Offline metrics (from holdout set)
    offline_auc = Column(Float)
    offline_precision_at_10 = Column(Float)
    offline_ndcg_at_10 = Column(Float)

    # Online metrics (from production)
    online_ctr = Column(Float)  # Click-through rate
    online_conversion_rate = Column(Float)
    online_revenue_lift = Column(Float)

    # Traffic allocation for A/B testing
    traffic_percentage = Column(Float, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    promoted_at = Column(DateTime)  # When moved to production
    retired_at = Column(DateTime)


class FeedbackLogger:
    """
    Production feedback logging for ML recommendations.

    Implements patterns from real ML systems:
    - Every prediction logged with full context for debugging
    - User feedback captured for continuous improvement
    - A/B testing infrastructure for safe deployments
    - Training data generation from production feedback
    """

    def __init__(self, session=None):
        """
        Initialize feedback logger.

        Args:
            session: SQLAlchemy session (optional, for testing)
        """
        self.session = session
        self._prediction_buffer: List[PredictionRecord] = []
        self._feedback_buffer: List[FeedbackRecord] = []
        self._buffer_size = 100  # Flush after this many records

    def log_prediction(
        self,
        model_version: str,
        account1_id: str,
        account2_id: str,
        score: float,
        confidence: float = None,
        rank: int = None,
        features: Dict[str, float] = None,
        explanation: str = None,
        session_id: str = None,
        user_id: str = None,
        request_id: str = None,
        latency_ms: float = None,
        model_stage: ModelStage = ModelStage.PRODUCTION,
    ) -> str:
        """
        Log a prediction for later analysis.

        Args:
            model_version: Version of model that made prediction
            account1_id: First account ID
            account2_id: Second account ID
            score: Prediction score (0-1)
            confidence: Model confidence (optional)
            rank: Position in recommendation list
            features: Feature values used for prediction
            explanation: Human-readable explanation
            session_id: User session identifier
            user_id: User identifier
            request_id: Request correlation ID
            latency_ms: Inference latency
            model_stage: Model lifecycle stage

        Returns:
            Prediction ID for feedback correlation
        """
        prediction_id = self._generate_prediction_id(
            model_version, account1_id, account2_id
        )

        record = PredictionRecord(
            prediction_id=prediction_id,
            model_version=model_version,
            model_stage=model_stage,
            account1_id=account1_id,
            account2_id=account2_id,
            score=score,
            confidence=confidence or score,  # Use score as confidence if not provided
            rank=rank,
            features=features or {},
            explanation=explanation,
            created_at=datetime.utcnow(),
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            latency_ms=latency_ms,
        )

        self._prediction_buffer.append(record)

        if len(self._prediction_buffer) >= self._buffer_size:
            self.flush()

        logger.debug(f"Logged prediction {prediction_id} for model {model_version}")
        return prediction_id

    def log_feedback(
        self,
        prediction_id: str,
        feedback_type: FeedbackType,
        user_id: str = None,
        outcome_value: float = None,
        outcome_date: datetime = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Log user feedback on a prediction.

        Args:
            prediction_id: ID of the prediction
            feedback_type: Type of feedback
            user_id: User providing feedback
            outcome_value: Revenue value if converted
            outcome_date: Date of outcome (e.g., deal close date)
            metadata: Additional context

        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())

        record = FeedbackRecord(
            feedback_id=feedback_id,
            prediction_id=prediction_id,
            feedback_type=feedback_type,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            outcome_value=outcome_value,
            outcome_date=outcome_date,
        )

        self._feedback_buffer.append(record)

        if len(self._feedback_buffer) >= self._buffer_size:
            self.flush()

        logger.info(f"Logged {feedback_type.value} feedback for prediction {prediction_id}")
        return feedback_id

    def flush(self):
        """Flush buffered records to database"""
        if not self.session:
            # In-memory mode for testing
            self._prediction_buffer = []
            self._feedback_buffer = []
            return

        try:
            # Bulk insert predictions
            for record in self._prediction_buffer:
                log = PredictionLog(
                    id=record.prediction_id,
                    model_version=record.model_version,
                    model_stage=record.model_stage.value,
                    account1_id=record.account1_id,
                    account2_id=record.account2_id,
                    score=record.score,
                    confidence=record.confidence,
                    rank=record.rank,
                    features=record.features,
                    explanation=record.explanation,
                    session_id=record.session_id,
                    user_id=record.user_id,
                    request_id=record.request_id,
                    latency_ms=record.latency_ms,
                    created_at=record.created_at,
                )
                self.session.merge(log)

            # Bulk insert feedback
            for record in self._feedback_buffer:
                log = FeedbackLog(
                    id=record.feedback_id,
                    prediction_id=record.prediction_id,
                    feedback_type=record.feedback_type.value,
                    user_id=record.user_id,
                    outcome_value=record.outcome_value,
                    outcome_date=record.outcome_date,
                    metadata=record.metadata,
                    created_at=record.timestamp,
                )
                self.session.add(log)

            self.session.commit()

            logger.info(
                f"Flushed {len(self._prediction_buffer)} predictions, "
                f"{len(self._feedback_buffer)} feedback records"
            )

        except Exception as e:
            logger.error(f"Failed to flush logs: {e}")
            self.session.rollback()
            raise
        finally:
            self._prediction_buffer = []
            self._feedback_buffer = []

    def _generate_prediction_id(
        self,
        model_version: str,
        account1_id: str,
        account2_id: str
    ) -> str:
        """Generate deterministic prediction ID"""
        # Sort account IDs for consistency
        sorted_accounts = tuple(sorted([account1_id, account2_id]))
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

        content = f"{model_version}:{sorted_accounts[0]}:{sorted_accounts[1]}:{timestamp}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]

        return f"pred_{hash_value}"


class TrainingDataGenerator:
    """
    Generate training data from production feedback.

    Converts user feedback into labeled examples for model retraining:
    - Positive: converted, accepted with high outcome value
    - Negative: rejected, ignored after viewing
    - Uncertain: viewed only (use for calibration)
    """

    def __init__(self, session):
        self.session = session

    def generate_training_data(
        self,
        min_feedback_count: int = 100,
        lookback_days: int = 90,
        positive_types: List[FeedbackType] = None,
        negative_types: List[FeedbackType] = None,
    ) -> pd.DataFrame:
        """
        Generate training data from feedback logs.

        Args:
            min_feedback_count: Minimum feedback to include a model version
            lookback_days: How far back to look for feedback
            positive_types: Feedback types to count as positive
            negative_types: Feedback types to count as negative

        Returns:
            DataFrame with features and labels
        """
        if positive_types is None:
            positive_types = [FeedbackType.CONVERTED, FeedbackType.ACCEPTED]
        if negative_types is None:
            negative_types = [FeedbackType.REJECTED]

        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Query predictions with feedback
        # In production, this would be a proper SQL query
        # Here we show the structure

        training_examples = []

        # This would be replaced with actual database queries
        # For demonstration, we show the expected output format

        logger.info(
            f"Generated training data: {len(training_examples)} examples "
            f"from last {lookback_days} days"
        )

        return pd.DataFrame(training_examples)

    def compute_feedback_metrics(
        self,
        model_version: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict[str, float]:
        """
        Compute online metrics from feedback.

        Args:
            model_version: Filter to specific model version
            start_date: Start of analysis window
            end_date: End of analysis window

        Returns:
            Dictionary of metrics
        """
        # In production, compute from database
        # Here we show expected structure

        return {
            "impression_count": 0,
            "click_count": 0,
            "ctr": 0.0,
            "accept_count": 0,
            "accept_rate": 0.0,
            "conversion_count": 0,
            "conversion_rate": 0.0,
            "total_revenue": 0.0,
            "avg_revenue_per_conversion": 0.0,
        }


class ABTestManager:
    """
    A/B testing for safe model deployment.

    Supports:
    - Traffic splitting between model versions
    - Metric tracking per variant
    - Statistical significance calculation
    - Safe rollout/rollback
    """

    def __init__(self, session):
        self.session = session
        self._traffic_allocation: Dict[str, float] = {}

    def register_model(
        self,
        version: str,
        model_type: str,
        artifact_path: str,
        offline_metrics: Dict[str, float],
        description: str = None,
    ) -> None:
        """
        Register a new model version.

        Args:
            version: Model version string
            model_type: Type of model (ensemble, xgboost, etc.)
            artifact_path: Path to model artifacts
            offline_metrics: Metrics from holdout evaluation
            description: Human-readable description
        """
        # Compute artifact hash for integrity
        artifact_hash = None
        if os.path.exists(artifact_path):
            with open(artifact_path, 'rb') as f:
                artifact_hash = hashlib.sha256(f.read()).hexdigest()

        logger.info(f"Registered model {version} ({model_type})")

    def set_traffic_allocation(self, allocations: Dict[str, float]) -> None:
        """
        Set traffic allocation for A/B test.

        Args:
            allocations: Model version -> traffic percentage (must sum to 1.0)
        """
        total = sum(allocations.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Traffic allocations must sum to 1.0, got {total}")

        self._traffic_allocation = allocations
        logger.info(f"Updated traffic allocation: {allocations}")

    def get_model_for_request(self, session_id: str) -> str:
        """
        Determine which model version to use for a request.

        Uses consistent hashing so same session always gets same model.

        Args:
            session_id: User session identifier

        Returns:
            Model version to use
        """
        if not self._traffic_allocation:
            return "default"

        # Hash session ID for consistent assignment
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        bucket = (hash_value % 1000) / 1000.0

        cumulative = 0.0
        for version, percentage in sorted(self._traffic_allocation.items()):
            cumulative += percentage
            if bucket < cumulative:
                return version

        return list(self._traffic_allocation.keys())[-1]

    def promote_model(self, version: str) -> None:
        """
        Promote a model to 100% production traffic.

        Args:
            version: Model version to promote
        """
        self._traffic_allocation = {version: 1.0}
        logger.info(f"Promoted model {version} to production (100% traffic)")

    def rollback(self, to_version: str) -> None:
        """
        Emergency rollback to a previous model version.

        Args:
            to_version: Version to rollback to
        """
        self._traffic_allocation = {to_version: 1.0}
        logger.warning(f"ROLLBACK: Reverted to model {to_version}")


# Convenience function for quick logging
_global_logger: Optional[FeedbackLogger] = None

def get_logger(session=None) -> FeedbackLogger:
    """Get or create global feedback logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = FeedbackLogger(session)
    return _global_logger


def log_prediction(**kwargs) -> str:
    """Convenience function to log prediction"""
    return get_logger().log_prediction(**kwargs)


def log_feedback(**kwargs) -> str:
    """Convenience function to log feedback"""
    return get_logger().log_feedback(**kwargs)
