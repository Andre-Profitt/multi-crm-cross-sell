# API Reference

## Overview

The Cross-Sell Intelligence API provides programmatic access to recommendations, scoring, and data management functionality.

**Base URL**: `http://localhost:8000`  
**API Documentation**: `http://localhost:8000/api/docs`

## Authentication

All API endpoints (except health checks) require JWT authentication.

### Get Access Token

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "password"
  }'
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

Use the token in subsequent requests:
```bash
-H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Endpoints

### 1. Sync Endpoint

Trigger data synchronization from Salesforce orgs.

```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json"
```

**Response**:
```json
{
  "message": "Pipeline run triggered successfully",
  "status": "started",
  "timestamp": "2025-01-19T10:30:00Z"
}
```

### 2. Recommendations Endpoint

Get cross-sell recommendations.

```bash
curl -X GET "http://localhost:8000/api/recommendations?min_score=0.7&limit=10" \
  -H "Authorization: Bearer $API_TOKEN"
```

**Query Parameters**:
- `min_score` (float): Minimum confidence score (0.0-1.0)
- `max_score` (float): Maximum confidence score
- `confidence_levels` (array): Filter by confidence levels
- `org_ids` (array): Filter by organization IDs
- `status` (array): Filter by status
- `limit` (int): Number of results (default: 100)
- `offset` (int): Pagination offset

**Python Example**:
```python
import requests

url = "http://localhost:8000/api/recommendations"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
params = {
    "min_score": 0.7,
    "limit": 10,
    "sort_by": "score",
    "sort_order": "desc"
}

resp = requests.get(url, headers=headers, params=params)
for r in resp.json():
    print(f"{r['account1_name']} <-> {r['account2_name']}: {r['score']:.2f}")
```

### 3. Score Opportunity

Score a specific cross-sell opportunity.

```bash
curl -X POST http://localhost:8000/api/score \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account1": {
      "Name": "TechCorp Solutions",
      "Industry": "Technology",
      "AnnualRevenue": 5000000,
      "NumberOfEmployees": 250
    },
    "account2": {
      "Name": "Finance Innovations Inc",
      "Industry": "Financial Services",
      "AnnualRevenue": 10000000,
      "NumberOfEmployees": 500
    },
    "include_explanation": true
  }'
```

**Response**:
```json
{
  "score": 0.85,
  "confidence_level": "High",
  "recommendation_type": "Industry Expansion",
  "estimated_value": 250000,
  "next_best_action": "Schedule executive introduction call",
  "explanation": {
    "individual_scores": {
      "neural_net": 0.87,
      "xgboost": 0.84,
      "random_forest": 0.83,
      "gradient_boost": 0.86
    },
    "feature_values": [0.0, 0.8, 1.0, 0.6, 0.7, 0.9]
  }
}
```

### 4. Get Insights

Get AI-generated insights and analytics.

```bash
curl -X GET "http://localhost:8000/api/insights?days_back=30" \
  -H "Authorization: Bearer $API_TOKEN"
```

**Response Structure**:
```json
{
  "summary": {
    "total_opportunities": 150,
    "total_potential_value": 5000000,
    "avg_confidence_score": 0.72,
    "high_confidence_count": 45
  },
  "top_opportunities": [...],
  "ai_insights": [
    "Technology sector shows highest cross-sell potential",
    "45 high-confidence opportunities require immediate action"
  ],
  "industry_breakdown": {...},
  "trend_data": [...]
}
```

### 5. Export Data

Export recommendations in various formats.

```bash
# CSV Export
curl -X GET "http://localhost:8000/api/export/csv?min_score=0.7" \
  -H "Authorization: Bearer $API_TOKEN" \
  -o recommendations.csv

# Excel Export
curl -X GET "http://localhost:8000/api/export/excel" \
  -H "Authorization: Bearer $API_TOKEN" \
  -o recommendations.xlsx

# JSON Export
curl -X GET "http://localhost:8000/api/export/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -o recommendations.json
```

### 6. Update Recommendation

Update the status of a recommendation.

```bash
curl -X PATCH http://localhost:8000/api/recommendations/123 \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_progress",
    "assigned_to": "sales.team@company.com",
    "notes": "Initial contact made, scheduling follow-up"
  }'
```

## Error Handling

All endpoints follow standard HTTP status codes:

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Missing or invalid token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Rate Limiting

API endpoints are rate-limited to:
- 100 requests per minute per API key
- 1000 requests per hour per API key

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642598400
```

## Webhooks (Coming Soon)

Subscribe to events:
- New high-score recommendations
- Model retrained
- Sync completed

## SDK Examples

### Python SDK

```python
from crosssell_client import CrossSellAPI

# Initialize client
client = CrossSellAPI(
    base_url="http://localhost:8000",
    api_token="YOUR_TOKEN"
)

# Get recommendations
recommendations = client.get_recommendations(min_score=0.8)

# Score opportunity
score = client.score_opportunity(
    account1={"Name": "Company A", "Industry": "Tech"},
    account2={"Name": "Company B", "Industry": "Finance"}
)
```

### JavaScript/Node.js

```javascript
const CrossSellClient = require('crosssell-client');

const client = new CrossSellClient({
  baseUrl: 'http://localhost:8000',
  apiToken: process.env.API_TOKEN
});

// Get recommendations
const recommendations = await client.getRecommendations({
  minScore: 0.8,
  limit: 10
});

// Score opportunity
const score = await client.scoreOpportunity({
  account1: { name: 'Company A', industry: 'Tech' },
  account2: { name: 'Company B', industry: 'Finance' }
});
```
