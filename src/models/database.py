"""
SQLAlchemy database models for Cross-Sell Intelligence Platform
"""

import os
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Organization(Base):
    """Salesforce organization configuration"""

    __tablename__ = "organizations"

    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    instance_url = Column(String(500))
    auth_type = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    accounts = relationship("Account", back_populates="organization")
    sync_logs = relationship("SyncLog", back_populates="organization")


class Account(Base):
    """Unified account data from Salesforce"""

    __tablename__ = "accounts"

    id = Column(String(100), primary_key=True)  # org_id + salesforce_id
    salesforce_id = Column(String(50), nullable=False)
    org_id = Column(String(50), ForeignKey("organizations.id"))
    name = Column(String(500), nullable=False)
    industry = Column(String(200))
    industry_standardized = Column(String(200))
    annual_revenue = Column(Float)
    number_of_employees = Column(Integer)
    billing_country = Column(String(100))
    billing_state = Column(String(100))
    billing_city = Column(String(200))
    website = Column(String(500))
    type = Column(String(50))
    rating = Column(String(50))
    created_date = Column(DateTime)
    last_activity_date = Column(DateTime)
    last_modified_date = Column(DateTime)

    # Metadata
    synced_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="accounts")
    opportunities = relationship("Opportunity", back_populates="account")


class Opportunity(Base):
    """Opportunity data from Salesforce"""

    __tablename__ = "opportunities"

    id = Column(String(100), primary_key=True)
    salesforce_id = Column(String(50), nullable=False)
    account_id = Column(String(100), ForeignKey("accounts.id"))
    name = Column(String(500))
    amount = Column(Float)
    stage_name = Column(String(200))
    close_date = Column(DateTime)
    is_won = Column(Boolean)
    is_closed = Column(Boolean)
    probability = Column(Float)
    type = Column(String(200))
    lead_source = Column(String(200))
    created_date = Column(DateTime)

    # Relationships
    account = relationship("Account", back_populates="opportunities")


class CrossSellRecommendation(Base):
    """Generated cross-sell recommendations"""

    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Account 1
    account1_id = Column(String(100), ForeignKey("accounts.id"))
    account1_name = Column(String(500))
    account1_org = Column(String(200))
    org1_id = Column(String(50))
    org1_industry = Column(String(200))

    # Account 2
    account2_id = Column(String(100), ForeignKey("accounts.id"))
    account2_name = Column(String(500))
    account2_org = Column(String(200))
    org2_id = Column(String(50))
    org2_industry = Column(String(200))

    # Scoring
    score = Column(Float, nullable=False)
    score_neural_net = Column(Float)
    score_xgboost = Column(Float)
    score_random_forest = Column(Float)
    score_gradient_boost = Column(Float)

    # Recommendation details
    confidence_level = Column(String(50))
    recommendation_type = Column(String(100))
    estimated_value = Column(Float)
    next_best_action = Column(Text)

    # Status tracking
    status = Column(String(50), default="new")  # new, in_progress, converted, dismissed
    assigned_to = Column(String(200))
    notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    actioned_at = Column(DateTime)

    # Feature values (for explainability)
    feature_values = Column(JSON)


class ModelMetadata(Base):
    """ML model metadata and performance tracking"""

    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50))  # ensemble, neural_net, etc.

    # Training info
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)

    # Performance metrics
    validation_auc = Column(Float)
    validation_accuracy = Column(Float)
    test_auc = Column(Float)
    test_accuracy = Column(Float)

    # Feature importance
    feature_importance = Column(JSON)

    # Model parameters
    parameters = Column(JSON)

    # File paths
    model_path = Column(String(500))

    # Status
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime)


class SyncLog(Base):
    """Track data synchronization history"""

    __tablename__ = "sync_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    org_id = Column(String(50), ForeignKey("organizations.id"))
    sync_type = Column(String(50))  # accounts, opportunities, products, etc.

    # Sync details
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(50))  # running, completed, failed

    # Metrics
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)

    # Error tracking
    error_message = Column(Text)

    # Relationships
    organization = relationship("Organization", back_populates="sync_logs")


class User(Base):
    """Application users for authentication"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)
    role = Column(String(50), default="viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database initialization
def init_db(database_url: str = None):
    """Initialize database with tables"""
    if database_url is None:
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://crosssell_user:password@localhost/crosssell_db"
        )

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

    return engine


def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()
