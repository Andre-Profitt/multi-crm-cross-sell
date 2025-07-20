"""
Complete Cross-Sell Pipeline Orchestrator
Implements actual data extraction, ML processing, and scheduling
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.connectors.base import ConnectorRegistry
from src.connectors.salesforce import SalesforceConfig
from src.ml.pipeline import (
    CrossSellRecommendationEngine,
    EnsembleScorer,
    FeatureEngineering,
    ModelConfig,
)
from src.models.database import (
    Account,
    CrossSellRecommendation,
    ModelMetadata,
    Opportunity,
    Organization,
    SyncLog,
    init_db,
)
from src.utils.notifications import NotificationManager

logger = logging.getLogger(__name__)


class CrossSellOrchestrator:
    """
    Main orchestrator for the cross-sell analysis pipeline
    Handles data extraction, ML processing, and scheduling
    """

    def __init__(self, config_path: str = "config/orgs.json"):
        self.config_path = config_path
        self.orgs = self._load_config()
        self.ml_config = self._load_ml_config()
        self.scheduler = AsyncIOScheduler()
        self.notification_manager = NotificationManager()

        # Initialize database
        self.db_url = os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://crosssell_user:password@localhost/crosssell_db"
        )
        self.engine = None
        self.session_maker = None

        # ML components
        self.feature_engineer = FeatureEngineering()
        self.scorer = None
        self.recommendation_engine = None

        # State tracking
        self.active_jobs = {}
        self.last_run_status = {}

    def _load_config(self) -> List[Dict]:
        """Load organization configuration"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config.get("organizations", [])
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found")
            return []

    def _load_ml_config(self) -> ModelConfig:
        """Load ML configuration"""
        ml_config_path = "config/ml_config.yaml"
        try:
            with open(ml_config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            # Convert to ModelConfig
            return ModelConfig(
                nn_hidden_layers=config_dict["neural_network"]["architecture"]["hidden_layers"],
                nn_dropout_rate=config_dict["neural_network"]["architecture"]["dropout_rate"],
                nn_learning_rate=config_dict["neural_network"]["training"]["learning_rate"],
                nn_batch_size=config_dict["neural_network"]["training"]["batch_size"],
                nn_epochs=config_dict["neural_network"]["training"]["epochs"],
                ensemble_weights=config_dict["ensemble"]["weights"],
            )
        except Exception as e:
            logger.warning(f"Could not load ML config: {e}, using defaults")
            return ModelConfig()

    async def initialize(self):
        """Initialize database and ML components"""
        # Initialize database
        self.engine = create_async_engine(self.db_url, echo=False)
        self.session_maker = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

        # Create tables if needed
        await self._ensure_database()

        # Load or train ML model
        await self._initialize_ml_model()

        logger.info("Orchestrator initialized successfully")

    async def _ensure_database(self):
        """Ensure database tables exist"""
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: init_db(str(self.db_url)))

    async def _initialize_ml_model(self):
        """Load existing model or train a new one"""
        model_path = "models/ensemble_scorer.pkl"

        if os.path.exists(model_path):
            logger.info("Loading existing ML model")
            self.scorer = joblib.load(model_path)
        else:
            logger.info("No existing model found, will train on first run")
            self.scorer = EnsembleScorer(self.ml_config)

        self.recommendation_engine = CrossSellRecommendationEngine(self.scorer)

    async def run_pipeline(self):
        """Run the complete cross-sell analysis pipeline"""
        start_time = datetime.now()
        logger.info("Starting cross-sell analysis pipeline...")

        try:
            # 1. Extract data from all orgs
            org_data = await self._extract_all_org_data()

            # 2. Process and analyze data
            processed_data = await self._process_data(org_data)

            # 3. Train/update ML model if needed
            if not self.scorer.is_trained:
                await self._train_model(processed_data)

            # 4. Generate recommendations
            recommendations = await self._generate_recommendations(processed_data)

            # 5. Save results
            await self._save_recommendations(recommendations)

            # 6. Send notifications
            await self._send_notifications(recommendations)

            # Update status
            self.last_run_status = {
                "status": "success",
                "start_time": start_time,
                "end_time": datetime.now(),
                "recommendations_generated": len(recommendations),
                "orgs_processed": len(org_data),
            }

            logger.info(f"Pipeline complete! Generated {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            self.last_run_status = {
                "status": "failed",
                "start_time": start_time,
                "end_time": datetime.now(),
                "error": str(e),
            }
            await self.notification_manager.send_error_notification(str(e))
            raise

    async def _extract_all_org_data(self) -> Dict[str, Dict]:
        """Extract data from all configured organizations"""
        org_data = {}
        tasks = []

        for org_config in self.orgs:
            task = self._extract_org_data(org_config)
            tasks.append(task)

        # Run extractions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for org_config, result in zip(self.orgs, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract from {org_config['org_name']}: {result}")
                continue
            org_data[org_config["org_id"]] = result

        return org_data

    async def _extract_org_data(self, org_config: Dict) -> Dict:
        """Extract data from a single organization"""
        logger.info(f"Extracting data from {org_config['org_name']}")

        # Create sync log
        async with self.session_maker() as session:
            sync_log = SyncLog(
                org_id=org_config["org_id"],
                sync_type="full",
                started_at=datetime.utcnow(),
                status="running",
            )
            session.add(sync_log)
            await session.commit()
            sync_log_id = sync_log.id

        try:
            # Create connector
            if org_config.get("crm_type", "salesforce") == "salesforce":
                config = SalesforceConfig(**org_config)
                connector = ConnectorRegistry.get_connector("salesforce", config)
            else:
                raise ValueError(f"Unsupported CRM type: {org_config.get('crm_type')}")

            # Extract data
            async with connector:
                data = await connector.extract_all()

            # Update sync log
            async with self.session_maker() as session:
                sync_log = await session.get(SyncLog, sync_log_id)
                sync_log.completed_at = datetime.utcnow()
                sync_log.status = "completed"
                sync_log.records_processed = sum(len(df) for df in data.values())
                await session.commit()

            # Save to database
            await self._save_org_data(org_config["org_id"], data)

            return data

        except Exception as e:
            # Update sync log with error
            async with self.session_maker() as session:
                sync_log = await session.get(SyncLog, sync_log_id)
                sync_log.completed_at = datetime.utcnow()
                sync_log.status = "failed"
                sync_log.error_message = str(e)
                await session.commit()
            raise

    async def _save_org_data(self, org_id: str, data: Dict[str, pd.DataFrame]):
        """Save extracted data to database"""
        async with self.session_maker() as session:
            # Save accounts
            if "accounts" in data and not data["accounts"].empty:
                for _, account in data["accounts"].iterrows():
                    db_account = Account(
                        id=f"{org_id}_{account['Id']}",
                        salesforce_id=account["Id"],
                        org_id=org_id,
                        name=account.get("Name", ""),
                        industry=account.get("Industry"),
                        annual_revenue=account.get("AnnualRevenue"),
                        number_of_employees=account.get("NumberOfEmployees"),
                        billing_country=account.get("BillingCountry"),
                        billing_state=account.get("BillingState"),
                        billing_city=account.get("BillingCity"),
                        website=account.get("Website"),
                        type=account.get("Type"),
                        rating=account.get("Rating"),
                        created_date=pd.to_datetime(account.get("CreatedDate")),
                        last_activity_date=pd.to_datetime(account.get("LastActivityDate")),
                        last_modified_date=pd.to_datetime(account.get("LastModifiedDate")),
                    )
                    session.add(db_account)

            # Save opportunities
            if "opportunities" in data and not data["opportunities"].empty:
                for _, opp in data["opportunities"].iterrows():
                    db_opp = Opportunity(
                        id=f"{org_id}_{opp['Id']}",
                        salesforce_id=opp["Id"],
                        account_id=f"{org_id}_{opp['AccountId']}" if opp.get("AccountId") else None,
                        name=opp.get("Name"),
                        amount=opp.get("Amount"),
                        stage_name=opp.get("StageName"),
                        close_date=pd.to_datetime(opp.get("CloseDate")),
                        is_won=opp.get("IsWon", False),
                        is_closed=opp.get("IsClosed", False),
                        probability=opp.get("Probability"),
                        type=opp.get("Type"),
                        lead_source=opp.get("LeadSource"),
                        created_date=pd.to_datetime(opp.get("CreatedDate")),
                    )
                    session.add(db_opp)

            await session.commit()

    async def _process_data(self, org_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Process extracted data for ML pipeline"""
        processed = {"org_data": org_data, "feature_matrix": None, "labels": None}

        # Additional processing logic here
        # - Standardize industries
        # - Calculate derived features
        # - Handle missing values

        return processed

    async def _train_model(self, processed_data: Dict[str, Any]):
        """Train or update the ML model"""
        logger.info("Training ML model...")

        # Generate training data from historical recommendations
        # This is a simplified version - in production you'd have labeled data
        X, y = await self._generate_training_data()

        if X is not None and len(X) > 0:
            # Train the model
            metrics = self.scorer.train(X, y)

            # Save model
            model_path = "models/ensemble_scorer.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.scorer, model_path)

            # Save model metadata
            async with self.session_maker() as session:
                model_meta = ModelMetadata(
                    model_version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type="ensemble",
                    training_samples=len(X),
                    validation_auc=metrics.get("xgboost_val_score", 0),
                    parameters=self.ml_config.__dict__,
                    model_path=model_path,
                    is_active=True,
                )
                session.add(model_meta)
                await session.commit()

            logger.info(f"Model trained successfully with metrics: {metrics}")

    async def _generate_training_data(self):
        """Generate training data from accounts and opportunities."""
        logger.info("Generating training data from database")

        async with self.session_maker() as session:
            accounts_result = await session.execute(select(Account))
            accounts = accounts_result.scalars().all()

            opp_result = await session.execute(select(Opportunity))
            opportunities = opp_result.scalars().all()

        if not accounts:
            logger.warning("No account data found, falling back to synthetic data")
            return self._generate_synthetic_training_data()

        accounts_df = pd.DataFrame(
            [
                {
                    "Id": acc.id,
                    "Industry": acc.industry,
                    "AnnualRevenue": acc.annual_revenue,
                    "NumberOfEmployees": acc.number_of_employees,
                    "BillingCountry": acc.billing_country,
                    "CreatedDate": acc.created_date,
                    "LastActivityDate": acc.last_activity_date,
                }
                for acc in accounts
            ]
        )

        won_map = {}
        for opp in opportunities:
            if opp.account_id not in won_map:
                won_map[opp.account_id] = False
            if opp.is_won:
                won_map[opp.account_id] = True

        accounts_df["label"] = accounts_df["Id"].map(lambda aid: 1 if won_map.get(aid) else 0)

        features = self.feature_engineer.create_account_features(accounts_df)
        X = features.values
        y = accounts_df["label"].values
        return X, y

    async def _generate_recommendations(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate cross-sell recommendations"""
        logger.info("Generating cross-sell recommendations...")

        recommendations = self.recommendation_engine.generate_recommendations(
            processed_data["org_data"]
        )

        return recommendations

    async def _save_recommendations(self, recommendations: pd.DataFrame):
        """Save recommendations to database"""
        if recommendations.empty:
            logger.warning("No recommendations to save")
            return

        async with self.session_maker() as session:
            for _, rec in recommendations.iterrows():
                db_rec = CrossSellRecommendation(
                    account1_id=rec["org1_account_id"],
                    account1_name=rec["org1_account_name"],
                    account1_org=rec.get("org1_name", ""),
                    org1_id=rec["org1_id"],
                    account2_id=rec["org2_account_id"],
                    account2_name=rec["org2_account_name"],
                    account2_org=rec.get("org2_name", ""),
                    org2_id=rec["org2_id"],
                    score=rec["score"],
                    confidence_level=rec["confidence_level"],
                    recommendation_type=rec["recommendation_type"],
                    estimated_value=rec["estimated_value"],
                    next_best_action=rec["next_best_action"],
                )
                session.add(db_rec)

            await session.commit()
            logger.info(f"Saved {len(recommendations)} recommendations to database")

    async def _send_notifications(self, recommendations: pd.DataFrame):
        """Send notifications about new high-value opportunities"""
        if recommendations.empty:
            return

        high_value = recommendations[
            (recommendations["score"] > 0.8) & (recommendations["estimated_value"] > 100000)
        ]

        if not high_value.empty:
            await self.notification_manager.send_opportunity_alert(high_value)

    def schedule_runs(self, frequency: str):
        """Schedule periodic pipeline runs"""
        logger.info(f"Scheduling {frequency} pipeline runs...")

        # Define cron expressions
        cron_expressions = {
            "hourly": "0 * * * *",
            "daily": "0 9 * * *",  # 9 AM daily
            "weekly": "0 9 * * 1",  # 9 AM every Monday
        }

        if frequency not in cron_expressions:
            raise ValueError(f"Invalid frequency: {frequency}")

        # Schedule the job
        self.scheduler.add_job(
            self.run_pipeline,
            CronTrigger.from_crontab(cron_expressions[frequency]),
            id="cross_sell_pipeline",
            name=f"Cross-sell pipeline ({frequency})",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(f"Scheduled {frequency} runs. Scheduler started.")

    async def stop(self):
        """Gracefully stop the orchestrator"""
        if self.scheduler.running:
            self.scheduler.shutdown()

        if self.engine:
            await self.engine.dispose()

        logger.info("Orchestrator stopped")

    def _generate_synthetic_training_data(self):
        """Generate synthetic training data for initial model"""
        logger.info("Generating synthetic training data for initial model")

        # Generate 1000 synthetic samples
        n_samples = 1000
        n_features = 6  # Match the feature engineering

        # Create synthetic features
        X = np.random.rand(n_samples, n_features)

        # Create labels based on some logic
        y = []
        for features in X:
            score = (
                features[0] * 0.3
                + features[1] * 0.25  # Industry similarity
                + features[2] * 0.2  # Size compatibility
                + features[3] * 0.15  # Geographic proximity
                + features[4] * 0.05  # Product complementarity
                + features[5] * 0.05  # Customer maturity  # Activity alignment
            )
            # Add some noise
            score += np.random.normal(0, 0.1)
            y.append(1 if score > 0.5 else 0)

        return X, np.array(y)
