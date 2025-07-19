"""
FastAPI REST API for Cross-Sell Intelligence Platform
Provides programmatic access to recommendations and insights
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import jwt
import os
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cross-Sell Intelligence API",
    description="AI-powered cross-sell opportunity identification across Salesforce orgs",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/crosssell_db")
engine = create_engine(DATABASE_URL)

# Pydantic models
class OpportunityScore(BaseModel):
    account1_id: str
    account1_name: str
    account1_org: str
    account2_id: str
    account2_name: str
    account2_org: str
    score: float = Field(..., ge=0, le=1)
    confidence_level: str
    estimated_value: float
    recommendation_type: str
    next_best_action: str

class OpportunityFilter(BaseModel):
    min_score: Optional[float] = Field(0.5, ge=0, le=1)
    max_score: Optional[float] = Field(1.0, ge=0, le=1)
    confidence_levels: Optional[List[str]] = None
    org_ids: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    limit: Optional[int] = Field(100, ge=1, le=1000)
    offset: Optional[int] = Field(0, ge=0)

class ScoringRequest(BaseModel):
    account1: Dict[str, Any]
    account2: Dict[str, Any]
    include_explanation: bool = False

class InsightResponse(BaseModel):
    summary: Dict[str, Any]
    top_opportunities: List[OpportunityScore]
    ai_insights: List[str]
    industry_breakdown: Dict[str, Any]
    action_distribution: Dict[str, int]
    generated_at: datetime

class ModelMetrics(BaseModel):
    model_version: str
    last_trained: datetime
    training_samples: int
    validation_auc: float
    production_metrics: Dict[str, float]
    feature_importance: List[Dict[str, float]]

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            os.getenv("JWT_SECRET_KEY", "your-secret-key"), 
            algorithms=["HS256"]
        )
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Cross-Sell Intelligence API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/recommendations", response_model=List[OpportunityScore], tags=["Recommendations"])
async def get_recommendations(
    filter: OpportunityFilter = Depends(),
    user: str = Depends(verify_token)
):
    """Get filtered cross-sell recommendations"""
    try:
        # Load recommendations from database
        query = f"""
        SELECT * FROM recommendations
        WHERE score >= {filter.min_score}
        AND score <= {filter.max_score}
        """
        
        # Apply additional filters
        if filter.confidence_levels:
            levels = "','".join(filter.confidence_levels)
            query += f" AND confidence_level IN ('{levels}')"
        
        if filter.org_ids:
            org_ids = "','".join(filter.org_ids)
            query += f" AND (org1_id IN ('{org_ids}') OR org2_id IN ('{org_ids}'))"
        
        query += f" ORDER BY score DESC LIMIT {filter.limit} OFFSET {filter.offset}"
        
        df = pd.read_sql(query, engine)
        
        # Convert to response model
        recommendations = [
            OpportunityScore(**row) for _, row in df.iterrows()
        ]
        
        logger.info(f"User {user} retrieved {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")

@app.post("/api/score", response_model=Dict[str, Any], tags=["Scoring"])
async def score_opportunity(
    request: ScoringRequest,
    user: str = Depends(verify_token)
):
    """Score a single cross-sell opportunity"""
    # Simplified implementation - in production, load actual model
    score = np.random.random()
    
    return {
        "score": score,
        "confidence_level": "High" if score > 0.7 else "Medium",
        "recommendation_type": "Industry Expansion",
        "estimated_value": 100000 * score,
        "next_best_action": "Schedule introduction call"
    }

@app.get("/api/insights", response_model=InsightResponse, tags=["Insights"])
async def get_insights(
    days_back: int = Query(30, ge=1, le=365),
    user: str = Depends(verify_token)
):
    """Get AI-generated insights and analytics"""
    try:
        # Query recent recommendations
        query = f"""
        SELECT * FROM recommendations
        WHERE created_at >= NOW() - INTERVAL '{days_back} days'
        """
        df = pd.read_sql(query, engine)
        
        # Generate insights
        summary = {
            "total_opportunities": len(df),
            "total_potential_value": float(df['estimated_value'].sum()) if len(df) > 0 else 0,
            "avg_confidence_score": float(df['score'].mean()) if len(df) > 0 else 0,
            "high_confidence_count": len(df[df['score'] > 0.8]) if len(df) > 0 else 0
        }
        
        # Top opportunities
        top_opps = df.nlargest(10, 'score').to_dict('records') if len(df) > 0 else []
        top_opportunities = [OpportunityScore(**opp) for opp in top_opps]
        
        # AI insights
        ai_insights = [
            f"Identified {summary['total_opportunities']} cross-sell opportunities worth ${summary['total_potential_value']:,.0f}",
            f"Average confidence score: {summary['avg_confidence_score']:.2%}",
            f"{summary['high_confidence_count']} high-confidence opportunities require immediate action"
        ]
        
        # Industry breakdown
        industry_breakdown = {}
        if len(df) > 0:
            industry_breakdown = df.groupby('org1_industry').agg({
                'score': ['count', 'mean'],
                'estimated_value': 'sum'
            }).to_dict()
        
        # Action distribution
        action_distribution = {}
        if len(df) > 0:
            action_distribution = df['next_best_action'].value_counts().to_dict()
        
        return InsightResponse(
            summary=summary,
            top_opportunities=top_opportunities,
            ai_insights=ai_insights,
            industry_breakdown=industry_breakdown,
            action_distribution=action_distribution,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")

@app.get("/api/export/{format}", tags=["Export"])
async def export_recommendations(
    format: str,
    filter: OpportunityFilter = Depends(),
    user: str = Depends(verify_token)
):
    """Export recommendations in various formats"""
    if format not in ["csv", "excel", "json"]:
        raise HTTPException(status_code=400, detail="Format must be csv, excel, or json")
    
    try:
        # Get filtered recommendations
        recommendations = await get_recommendations(filter, user)
        df = pd.DataFrame([r.dict() for r in recommendations])
        
        # Generate file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "csv":
            filename = f"exports/recommendations_{timestamp}.csv"
            df.to_csv(filename, index=False)
            media_type = "text/csv"
        elif format == "excel":
            filename = f"exports/recommendations_{timestamp}.xlsx"
            df.to_excel(filename, index=False, engine='openpyxl')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:  # json
            filename = f"exports/recommendations_{timestamp}.json"
            df.to_json(filename, orient='records', indent=2)
            media_type = "application/json"
        
        return FileResponse(
            filename,
            media_type=media_type,
            filename=os.path.basename(filename)
        )
        
    except Exception as e:
        logger.error(f"Error exporting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export recommendations")

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
