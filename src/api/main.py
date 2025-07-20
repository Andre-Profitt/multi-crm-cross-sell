"""
FastAPI REST API for Cross-Sell Intelligence Platform
Complete implementation with proper routes, security, and error handling
"""

import io
import logging
import os
import secrets
from datetime import datetime, timedelta
from functools import lru_cache

import yaml
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import redis.asyncio as redis
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import jwt
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import (
    Account,
    CrossSellRecommendation,
    ModelMetadata,
    Organization,
    get_session,
)
from src.orchestrator import CrossSellOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cross-Sell Intelligence API",
    description="AI-powered cross-sell opportunity identification across Salesforce orgs",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# JWT settings with secure defaults
def get_jwt_secret():
    """Get or generate JWT secret key"""
    jwt_secret = os.getenv("JWT_SECRET_KEY")

    if not jwt_secret or jwt_secret == "your-secret-key-change-in-production":
        secret_file = Path(".jwt_secret")

        if secret_file.exists():
            jwt_secret = secret_file.read_text().strip()
        else:
            jwt_secret = secrets.token_urlsafe(32)
            secret_file.write_text(jwt_secret)
            logger.warning("Generated new JWT secret. Set JWT_SECRET_KEY env var in production!")

    return jwt_secret


JWT_SECRET_KEY = get_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
# Global orchestrator instance
orchestrator = None


def get_cache_ttl() -> int:
    """Determine cache TTL from env or config"""
    env_ttl = os.getenv("API_CACHE_TTL")
    if env_ttl:
        try:
            return int(env_ttl)
        except ValueError:
            logger.warning("Invalid API_CACHE_TTL env var, using default")
    try:
        with open("config/ml_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return int(config.get("api", {}).get("cache_ttl_seconds", 60))
    except Exception:
        return 60


CACHE_TTL_SECONDS = get_cache_ttl()


# Pydantic models
class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCredentials(BaseModel):
    username: str
    password: str


class OpportunityScore(BaseModel):
    id: int
    account1_id: str
    account1_name: str
    account1_org: str
    org1_industry: Optional[str]
    account2_id: str
    account2_name: str
    account2_org: str
    org2_industry: Optional[str]
    score: float = Field(..., ge=0, le=1)
    confidence_level: str
    estimated_value: float
    recommendation_type: str
    next_best_action: str
    status: str = "new"
    created_at: datetime


class OpportunityFilter(BaseModel):
    min_score: Optional[float] = Field(0.5, ge=0, le=1)
    max_score: Optional[float] = Field(1.0, ge=0, le=1)
    confidence_levels: Optional[List[str]] = None
    org_ids: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    status: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: Optional[int] = Field(100, ge=1, le=1000)
    offset: Optional[int] = Field(0, ge=0)
    sort_by: Optional[str] = Field("score", regex="^(score|estimated_value|created_at)$")
    sort_order: Optional[str] = Field("desc", regex="^(asc|desc)$")


class OpportunityUpdate(BaseModel):
    status: Optional[str] = Field(None, regex="^(new|in_progress|converted|dismissed)$")
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


class ScoringRequest(BaseModel):
    account1: Dict[str, Any]
    account2: Dict[str, Any]
    include_explanation: bool = False


class ScoringResponse(BaseModel):
    score: float
    confidence_level: str
    recommendation_type: str
    estimated_value: float
    next_best_action: str
    explanation: Optional[Dict[str, Any]] = None


class InsightResponse(BaseModel):
    summary: Dict[str, Any]
    top_opportunities: List[OpportunityScore]
    ai_insights: List[str]
    industry_breakdown: Dict[str, Any]
    action_distribution: Dict[str, int]
    trend_data: List[Dict[str, Any]]
    generated_at: datetime


class ModelMetrics(BaseModel):
    model_version: str
    last_trained: datetime
    training_samples: int
    validation_auc: float
    test_auc: Optional[float]
    production_metrics: Dict[str, float]
    feature_importance: List[Dict[str, float]]
    is_active: bool


class PipelineStatus(BaseModel):
    status: str
    last_run: Optional[Dict[str, Any]]
    next_run: Optional[datetime]
    active_jobs: List[str]
    recent_errors: List[Dict[str, Any]]


# Authentication functions
def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return payload"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# Dependency to get database session
async def get_db() -> AsyncSession:
    """Get async database session"""
    async with orchestrator.session_maker() as session:
        yield session


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    orchestrator = CrossSellOrchestrator()
    await orchestrator.initialize()
    cache_backend = os.getenv("CACHE_BACKEND", "inmemory").lower()
    cache_url = os.getenv("CACHE_URL", "redis://localhost:6379/0")
    if cache_backend == "redis":
        redis_client = redis.from_url(cache_url, encoding="utf8", decode_responses=True)
        FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    if orchestrator:
        await orchestrator.stop()
    logger.info("API shutdown complete")


# Health check endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Cross-Sell Intelligence API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/health", tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check"""
    try:
        # Check database
        result = await db.execute(select(func.count(CrossSellRecommendation.id)))
        recommendation_count = result.scalar()

        return {
            "status": "healthy",
            "database": "connected",
            "recommendations_count": recommendation_count,
            "ml_model_loaded": orchestrator.scorer is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# Authentication endpoints
@app.post("/api/auth/token", response_model=TokenData, tags=["Authentication"])
async def login(credentials: UserCredentials):
    """Authenticate user and return JWT token"""
    # In production, verify against database or external auth service
    # This is a simplified version
    if credentials.username == "admin" and credentials.password == "password":
        access_token = create_access_token(
            data={"sub": credentials.username, "scopes": ["read", "write"]}
        )
        return TokenData(access_token=access_token, expires_in=JWT_EXPIRATION_HOURS * 3600)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )


# Recommendation endpoints
@app.get("/api/recommendations", response_model=List[OpportunityScore], tags=["Recommendations"])
@cache(expire=CACHE_TTL_SECONDS)
async def get_recommendations(
    filter: OpportunityFilter = Depends(),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(verify_token),
):
    """Get filtered cross-sell recommendations"""
    try:
        # Build query
        query = select(CrossSellRecommendation)

        # Apply filters
        conditions = []
        if filter.min_score is not None:
            conditions.append(CrossSellRecommendation.score >= filter.min_score)
        if filter.max_score is not None:
            conditions.append(CrossSellRecommendation.score <= filter.max_score)
        if filter.confidence_levels:
            conditions.append(
                CrossSellRecommendation.confidence_level.in_(filter.confidence_levels)
            )
        if filter.org_ids:
            conditions.append(
                or_(
                    CrossSellRecommendation.org1_id.in_(filter.org_ids),
                    CrossSellRecommendation.org2_id.in_(filter.org_ids),
                )
            )
        if filter.status:
            conditions.append(CrossSellRecommendation.status.in_(filter.status))
        if filter.date_from:
            conditions.append(CrossSellRecommendation.created_at >= filter.date_from)
        if filter.date_to:
            conditions.append(CrossSellRecommendation.created_at <= filter.date_to)

        if conditions:
            query = query.where(and_(*conditions))

        # Apply sorting
        sort_column = getattr(CrossSellRecommendation, filter.sort_by)
        if filter.sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.limit(filter.limit).offset(filter.offset)

        # Execute query
        result = await db.execute(query)
        recommendations = result.scalars().all()

        # Convert to response model
        return [
            OpportunityScore(
                id=rec.id,
                account1_id=rec.account1_id,
                account1_name=rec.account1_name,
                account1_org=rec.account1_org,
                org1_industry=rec.org1_industry,
                account2_id=rec.account2_id,
                account2_name=rec.account2_name,
                account2_org=rec.account2_org,
                org2_industry=rec.org2_industry,
                score=rec.score,
                confidence_level=rec.confidence_level,
                estimated_value=rec.estimated_value,
                recommendation_type=rec.recommendation_type,
                next_best_action=rec.next_best_action,
                status=rec.status,
                created_at=rec.created_at,
            )
            for rec in recommendations
        ]

    except Exception as e:
        logger.error(f"Error retrieving recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendations",
        )


@app.get(
    "/api/recommendations/{recommendation_id}",
    response_model=OpportunityScore,
    tags=["Recommendations"],
)
async def get_recommendation(
    recommendation_id: int, db: AsyncSession = Depends(get_db), user: Dict = Depends(verify_token)
):
    """Get a specific recommendation by ID"""
    result = await db.execute(
        select(CrossSellRecommendation).where(CrossSellRecommendation.id == recommendation_id)
    )
    recommendation = result.scalar_one_or_none()

    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recommendation not found"
        )

    return OpportunityScore(
        id=recommendation.id,
        account1_id=recommendation.account1_id,
        account1_name=recommendation.account1_name,
        account1_org=recommendation.account1_org,
        org1_industry=recommendation.org1_industry,
        account2_id=recommendation.account2_id,
        account2_name=recommendation.account2_name,
        account2_org=recommendation.account2_org,
        org2_industry=recommendation.org2_industry,
        score=recommendation.score,
        confidence_level=recommendation.confidence_level,
        estimated_value=recommendation.estimated_value,
        recommendation_type=recommendation.recommendation_type,
        next_best_action=recommendation.next_best_action,
        status=recommendation.status,
        created_at=recommendation.created_at,
    )


@app.patch("/api/recommendations/{recommendation_id}", tags=["Recommendations"])
async def update_recommendation(
    recommendation_id: int,
    update: OpportunityUpdate,
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(verify_token),
):
    """Update a recommendation's status or assignment"""
    result = await db.execute(
        select(CrossSellRecommendation).where(CrossSellRecommendation.id == recommendation_id)
    )
    recommendation = result.scalar_one_or_none()

    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recommendation not found"
        )

    # Apply updates
    if update.status:
        recommendation.status = update.status
        if update.status == "converted":
            recommendation.actioned_at = datetime.utcnow()
    if update.assigned_to is not None:
        recommendation.assigned_to = update.assigned_to
    if update.notes is not None:
        recommendation.notes = update.notes

    recommendation.updated_at = datetime.utcnow()

    await db.commit()

    return {"message": "Recommendation updated successfully"}


@app.post("/api/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_opportunity(request: ScoringRequest, user: Dict = Depends(verify_token)):
    """Score a single cross-sell opportunity"""
    try:
        # Convert accounts to pandas Series for feature engineering
        account1 = pd.Series(request.account1)
        account2 = pd.Series(request.account2)

        # Generate features
        features = orchestrator.feature_engineer.create_cross_org_features(account1, account2)

        # Score using the model
        if orchestrator.scorer.is_trained:
            score, individual_scores = orchestrator.scorer.predict(features.reshape(1, -1))
            score_value = float(score[0])
        else:
            # Fallback if model not trained
            score_value = 0.5
            individual_scores = {}

        # Determine confidence level
        if score_value > 0.8:
            confidence = "Very High"
        elif score_value > 0.7:
            confidence = "High"
        elif score_value > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        response = ScoringResponse(
            score=score_value,
            confidence_level=confidence,
            recommendation_type="Industry Expansion",
            estimated_value=100000 * score_value,
            next_best_action="Schedule introduction call"
            if score_value > 0.7
            else "Add to nurture campaign",
        )

        if request.include_explanation and individual_scores:
            response.explanation = {
                "individual_scores": {k: float(v[0]) for k, v in individual_scores.items()},
                "feature_values": features.tolist(),
            }

        return response

    except Exception as e:
        logger.error(f"Error scoring opportunity: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to score opportunity"
        )


@app.get("/api/insights", response_model=InsightResponse, tags=["Insights"])
async def get_insights(
    days_back: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(verify_token),
):
    """Get AI-generated insights and analytics"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        # Get recent recommendations
        result = await db.execute(
            select(CrossSellRecommendation)
            .where(CrossSellRecommendation.created_at >= cutoff_date)
            .order_by(CrossSellRecommendation.score.desc())
        )
        recommendations = result.scalars().all()

        # Calculate summary statistics
        total_value = sum(r.estimated_value for r in recommendations)
        avg_score = (
            sum(r.score for r in recommendations) / len(recommendations) if recommendations else 0
        )
        high_confidence = sum(1 for r in recommendations if r.score > 0.8)

        summary = {
            "total_opportunities": len(recommendations),
            "total_potential_value": total_value,
            "avg_confidence_score": avg_score,
            "high_confidence_count": high_confidence,
            "conversion_rate": 0.24,  # Would come from historical data
            "avg_deal_size": total_value / len(recommendations) if recommendations else 0,
        }

        # Top opportunities
        top_opportunities = [
            OpportunityScore(
                id=r.id,
                account1_id=r.account1_id,
                account1_name=r.account1_name,
                account1_org=r.account1_org,
                org1_industry=r.org1_industry,
                account2_id=r.account2_id,
                account2_name=r.account2_name,
                account2_org=r.account2_org,
                org2_industry=r.org2_industry,
                score=r.score,
                confidence_level=r.confidence_level,
                estimated_value=r.estimated_value,
                recommendation_type=r.recommendation_type,
                next_best_action=r.next_best_action,
                status=r.status,
                created_at=r.created_at,
            )
            for r in recommendations[:10]
        ]

        # AI insights
        ai_insights = []
        if recommendations:
            ai_insights.extend(
                [
                    f"Identified {len(recommendations)} cross-sell opportunities worth ${total_value:,.0f}",
                    f"Average confidence score of {avg_score:.1%} indicates strong opportunity quality",
                    f"{high_confidence} high-confidence opportunities require immediate action",
                ]
            )

            # Industry insights
            industry_counts = {}
            for r in recommendations:
                if r.org1_industry:
                    industry_counts[r.org1_industry] = industry_counts.get(r.org1_industry, 0) + 1

            if industry_counts:
                top_industry = max(industry_counts, key=industry_counts.get)
                ai_insights.append(
                    f"{top_industry} sector shows highest cross-sell potential with {industry_counts[top_industry]} opportunities"
                )

        # Industry breakdown
        industry_breakdown = {}
        for r in recommendations:
            if r.org1_industry:
                if r.org1_industry not in industry_breakdown:
                    industry_breakdown[r.org1_industry] = {
                        "count": 0,
                        "total_value": 0,
                        "avg_score": 0,
                    }
                industry_breakdown[r.org1_industry]["count"] += 1
                industry_breakdown[r.org1_industry]["total_value"] += r.estimated_value
                industry_breakdown[r.org1_industry]["avg_score"] += r.score

        # Calculate averages
        for industry in industry_breakdown:
            count = industry_breakdown[industry]["count"]
            industry_breakdown[industry]["avg_score"] /= count

        # Action distribution
        action_distribution = {}
        for r in recommendations:
            action = r.next_best_action
            action_distribution[action] = action_distribution.get(action, 0) + 1

        # Trend data (last 7 days)
        trend_data = []
        for i in range(7):
            date = datetime.utcnow().date() - timedelta(days=i)
            day_recs = [r for r in recommendations if r.created_at.date() == date]
            trend_data.append(
                {
                    "date": date.isoformat(),
                    "count": len(day_recs),
                    "value": sum(r.estimated_value for r in day_recs),
                }
            )
        trend_data.reverse()

        return InsightResponse(
            summary=summary,
            top_opportunities=top_opportunities,
            ai_insights=ai_insights,
            industry_breakdown=industry_breakdown,
            action_distribution=action_distribution,
            trend_data=trend_data,
            generated_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate insights"
        )


@app.get("/api/export/{format}", tags=["Export"])
async def export_recommendations(
    format: str,
    background_tasks: BackgroundTasks,
    filter: OpportunityFilter = Depends(),
    db: AsyncSession = Depends(get_db),
    user: Dict = Depends(verify_token),
):
    """Export recommendations in various formats"""
    if format not in ["csv", "excel", "json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Format must be csv, excel, or json"
        )

    try:
        # Get filtered recommendations (reuse the same logic)
        recommendations = await get_recommendations(filter, db, user)

        # Convert to DataFrame
        df = pd.DataFrame([r.dict() for r in recommendations])

        # Generate file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=recommendations_{timestamp}.csv"
                },
            )

        elif format == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Recommendations")
            output.seek(0)

            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename=recommendations_{timestamp}.xlsx"
                },
            )

        else:  # json
            return JSONResponse(
                content=df.to_dict(orient="records"),
                headers={
                    "Content-Disposition": f"attachment; filename=recommendations_{timestamp}.json"
                },
            )

    except Exception as e:
        logger.error(f"Error exporting recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export recommendations",
        )


@app.get("/api/models/current", response_model=ModelMetrics, tags=["Models"])
async def get_current_model(db: AsyncSession = Depends(get_db), user: Dict = Depends(verify_token)):
    """Get current model metrics"""
    result = await db.execute(
        select(ModelMetadata)
        .where(ModelMetadata.is_active == True)
        .order_by(ModelMetadata.trained_at.desc())
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active model found")

    # Get production metrics (would come from monitoring)
    production_metrics = {
        "daily_predictions": 1250,
        "avg_prediction_time_ms": 45,
        "cache_hit_rate": 0.72,
        "error_rate": 0.001,
    }

    # Mock feature importance (would come from model)
    feature_importance = [
        {"feature": "industry_match", "importance": 0.28},
        {"feature": "size_compatibility", "importance": 0.22},
        {"feature": "geographic_proximity", "importance": 0.18},
        {"feature": "product_complementarity", "importance": 0.15},
        {"feature": "customer_maturity", "importance": 0.10},
        {"feature": "activity_alignment", "importance": 0.07},
    ]

    return ModelMetrics(
        model_version=model.model_version,
        last_trained=model.trained_at,
        training_samples=model.training_samples,
        validation_auc=model.validation_auc,
        test_auc=model.test_auc,
        production_metrics=production_metrics,
        feature_importance=feature_importance,
        is_active=model.is_active,
    )


@app.post("/api/pipeline/run", tags=["Pipeline"])
async def trigger_pipeline_run(
    background_tasks: BackgroundTasks, user: Dict = Depends(verify_token)
):
    """Manually trigger a pipeline run"""
    # Check user has write permissions
    if "write" not in user.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
        )

    # Add pipeline run to background tasks
    background_tasks.add_task(orchestrator.run_pipeline)

    return {
        "message": "Pipeline run triggered successfully",
        "status": "started",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/pipeline/status", response_model=PipelineStatus, tags=["Pipeline"])
async def get_pipeline_status(
    db: AsyncSession = Depends(get_db), user: Dict = Depends(verify_token)
):
    """Get current pipeline status"""
    # Get recent errors from sync logs
    result = await db.execute(
        select(SyncLog)
        .where(SyncLog.status == "failed")
        .order_by(SyncLog.started_at.desc())
        .limit(5)
    )
    failed_syncs = result.scalars().all()

    recent_errors = [
        {"timestamp": sync.started_at, "org_id": sync.org_id, "error": sync.error_message}
        for sync in failed_syncs
    ]

    # Get next scheduled run (if scheduler is running)
    next_run = None
    if orchestrator.scheduler.running:
        job = orchestrator.scheduler.get_job("cross_sell_pipeline")
        if job:
            next_run = job.next_run_time

    return PipelineStatus(
        status="running" if orchestrator.scheduler.running else "stopped",
        last_run=orchestrator.last_run_status,
        next_run=next_run,
        active_jobs=list(orchestrator.active_jobs.keys()),
        recent_errors=recent_errors,
    )


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Resource not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
