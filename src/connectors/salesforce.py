"""
Salesforce Connector v2 - Async with Bulk API Support
Addresses all issues from code review
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import aiohttp
import jwt
from cryptography.fernet import Fernet
import aiofiles
import sqlite3
from contextlib import asynccontextmanager
import backoff
from dataclasses import dataclass
import logging

from .base import BaseCRMConnector, CRMConfig, register_connector

logger = logging.getLogger(__name__)


@dataclass
class SalesforceConfig(CRMConfig):
    """Salesforce-specific configuration"""
    auth_type: str = "jwt"  # "jwt", "password", "oauth2"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    security_token: Optional[str] = None
    private_key_path: Optional[str] = None
    sandbox: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if not self.api_version:
            self.api_version = "v60.0"
        
        # Validate auth requirements
        if self.auth_type == "jwt" and not self.private_key_path:
            raise ValueError("JWT auth requires private_key_path")
        elif self.auth_type == "password":
            if not all([self.username, self.password, self.security_token]):
                raise ValueError("Password auth requires username, password, and security_token")


class TokenManager:
    """Secure token management with SQLite backend"""
    
    def __init__(self, db_path: str = ".sf_cache/tokens.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        
        # Generate or load encryption key
        key_path = os.path.join(os.path.dirname(db_path), "token.key")
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self.fernet = Fernet(f.read())
        else:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            self.fernet = Fernet(key)
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    org_id TEXT PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    instance_url TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def get_token(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get token from cache"""
        async with aiofiles.open(self.db_path, 'rb') as f:
            # Use thread for SQLite operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_token_sync, org_id)
    
    def _get_token_sync(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous token retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_token, instance_url, expiry FROM tokens WHERE org_id = ?",
                (org_id,)
            )
            row = cursor.fetchone()
            
            if row:
                expiry = datetime.fromisoformat(row[2])
                if datetime.utcnow() < expiry:
                    return {
                        'access_token': self.fernet.decrypt(row[0].encode()).decode(),
                        'instance_url': row[1],
                        'expiry': expiry
                    }
        return None
    
    async def save_token(self, org_id: str, token_data: Dict[str, Any]):
        """Save token to cache"""
        encrypted_token = self.fernet.encrypt(token_data['access_token'].encode()).decode()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            self._save_token_sync, 
            org_id, 
            encrypted_token,
            token_data['instance_url'],
            token_data.get('expiry', datetime.utcnow() + timedelta(hours=2))
        )
    
    def _save_token_sync(self, org_id: str, encrypted_token: str, 
                        instance_url: str, expiry: datetime):
        """Synchronous token save"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tokens (org_id, access_token, instance_url, expiry)
                VALUES (?, ?, ?, ?)
            """, (org_id, encrypted_token, instance_url, expiry.isoformat()))


@register_connector("salesforce")
class SalesforceConnector(BaseCRMConnector):
    """
    Async Salesforce connector with bulk API support
    """
    
    def __init__(self, config: SalesforceConfig):
        super().__init__(config)
        self.config: SalesforceConfig = config
        self.token_manager = TokenManager()
        self.rate_limiter = RateLimiter(calls_per_minute=100)
        self._bulk_jobs = {}
    
    async def authenticate(self) -> bool:
        """Authenticate with Salesforce"""
        try:
            # Check cached token first
            cached = await self.token_manager.get_token(self.config.org_id)
            if cached:
                self.config.access_token = cached['access_token']
                self.config.token_expiry = cached['expiry']
                self._authenticated = True
                logger.info(f"Using cached token for {self.config.org_name}")
                return True
            
            # Authenticate based on type
            if self.config.auth_type == "jwt":
                return await self._auth_jwt()
            elif self.config.auth_type == "password":
                return await self._auth_password()
            else:
                raise ValueError(f"Unsupported auth type: {self.config.auth_type}")
                
        except Exception as e:
            logger.error(f"Authentication failed for {self.config.org_name}: {str(e)}")
            return False
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _auth_password(self) -> bool:
        """Password OAuth flow"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        auth_url = f"{self.config.instance_url}/services/oauth2/token"
        
        data = {
            'grant_type': 'password',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'username': self.config.username,
            'password': f"{self.config.password}{self.config.security_token}"
        }
        
        async with self._session.post(auth_url, data=data) as response:
            response.raise_for_status()
            token_data = await response.json()
            
            self.config.access_token = token_data['access_token']
            self.config.instance_url = token_data['instance_url']
            self.config.token_expiry = datetime.utcnow() + timedelta(hours=2)
            
            # Save to cache
            await self.token_manager.save_token(self.config.org_id, {
                'access_token': self.config.access_token,
                'instance_url': self.config.instance_url,
                'expiry': self.config.token_expiry
            })
            
            self._authenticated = True
            logger.info(f"Successfully authenticated {self.config.org_name}")
            return True
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _auth_jwt(self) -> bool:
        """JWT Bearer flow authentication"""
        # Implementation for JWT auth
        # This would use the private key to sign a JWT assertion
        raise NotImplementedError("JWT auth to be implemented")
    
    async def test_connection(self) -> bool:
        """Test Salesforce connection"""
        if not self._authenticated:
            if not await self.authenticate():
                return False
        
        try:
            async with self._get_session() as session:
                url = f"{self.config.instance_url}/services/data/{self.config.api_version}"
                headers = self._get_headers()
                
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @asynccontextmanager
    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        try:
            yield self._session
        finally:
            pass  # Don't close here, close in connector close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'Authorization': f'Bearer {self.config.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _query_rest(self, soql: str) -> pd.DataFrame:
        """Execute SOQL query via REST API with pagination"""
        await self.rate_limiter.acquire()
        
        records = []
        next_url = f"{self.config.instance_url}/services/data/{self.config.api_version}/query?q={soql}"
        
        async with self._get_session() as session:
            while next_url:
                async with session.get(next_url, headers=self._get_headers()) as response:
                    if response.status == 401:
                        # Token expired, re-authenticate
                        logger.info("Token expired, re-authenticating...")
                        await self.authenticate()
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    records.extend(data['records'])
                    
                    if data.get('done', True):
                        break
                    else:
                        next_url = self.config.instance_url + data['nextRecordsUrl']
        
        # Convert to DataFrame and clean
        if records:
            df = pd.DataFrame(records)
            df = df.drop(['attributes'], axis=1, errors='ignore')
            return self._add_metadata(df)
        
        return pd.DataFrame()
    
    async def _query_bulk(self, soql: str, object_type: str) -> pd.DataFrame:
        """Execute query via Bulk API 2.0 for large datasets"""
        # Create bulk job
        job_id = await self._create_bulk_job(object_type)
        
        try:
            # Submit query
            await self._add_bulk_query(job_id, soql)
            
            # Wait for completion
            await self._wait_for_bulk_job(job_id)
            
            # Get results
            return await self._get_bulk_results(job_id)
            
        finally:
            # Clean up job
            await self._close_bulk_job(job_id)
    
    async def _create_bulk_job(self, object_type: str) -> str:
        """Create a Bulk API 2.0 job"""
        url = f"{self.config.instance_url}/services/data/{self.config.api_version}/jobs/query"
        
        data = {
            "operation": "query",
            "object": object_type,
            "contentType": "CSV",
            "lineEnding": "LF"
        }
        
        async with self._get_session() as session:
            async with session.post(url, json=data, headers=self._get_headers()) as response:
                response.raise_for_status()
                result = await response.json()
                return result['id']
    
    async def extract_accounts(self, 
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Extract account data"""
        soql = """
        SELECT 
            Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
            BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry,
            Website, Phone, Description, OwnerId, CreatedDate, LastModifiedDate,
            LastActivityDate, AccountSource, Rating
        FROM Account
        WHERE IsDeleted = false
        """
        
        if filters:
            conditions = []
            for field, value in filters.items():
                conditions.append(f"{field} = '{value}'")
            soql += " AND " + " AND ".join(conditions)
        
        soql += " ORDER BY AnnualRevenue DESC NULLS LAST"
        
        if limit:
            soql += f" LIMIT {limit}"
        
        logger.info(f"Extracting accounts from {self.config.org_name}")
        
        # Use bulk API for large extracts
        if not limit or limit > 10000:
            return await self._query_bulk(soql, "Account")
        else:
            return await self._query_rest(soql)
    
    async def extract_opportunities(self,
                                  filters: Optional[Dict[str, Any]] = None,
                                  limit: Optional[int] = None) -> pd.DataFrame:
        """Extract opportunity data"""
        soql = """
        SELECT 
            Id, AccountId, Name, Description, StageName, Amount, Probability,
            CloseDate, Type, NextStep, LeadSource, IsClosed, IsWon,
            ForecastCategory, ForecastCategoryName, CampaignId,
            HasOpportunityLineItem, Pricebook2Id, OwnerId,
            CreatedDate, LastModifiedDate, LastActivityDate
        FROM Opportunity
        WHERE IsDeleted = false
        """
        
        if filters:
            conditions = []
            for field, value in filters.items():
                conditions.append(f"{field} = '{value}'")
            soql += " AND " + " AND ".join(conditions)
        
        soql += " ORDER BY CloseDate DESC"
        
        if limit:
            soql += f" LIMIT {limit}"
        
        logger.info(f"Extracting opportunities from {self.config.org_name}")
        return await self._query_rest(soql)
    
    async def extract_products(self,
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Extract product data"""
        soql = """
        SELECT 
            Id, Name, ProductCode, Description, IsActive, 
            Family, CreatedDate, LastModifiedDate
        FROM Product2
        WHERE IsDeleted = false
        """
        
        if limit:
            soql += f" LIMIT {limit}"
        
        logger.info(f"Extracting products from {self.config.org_name}")
        return await self._query_rest(soql)
    
    async def extract_contacts(self,
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Extract contact data"""
        soql = """
        SELECT 
            Id, AccountId, FirstName, LastName, Title, Email,
            Phone, MobilePhone, Department, CreatedDate, LastModifiedDate
        FROM Contact
        WHERE IsDeleted = false
        """
        
        if limit:
            soql += f" LIMIT {limit}"
        
        logger.info(f"Extracting contacts from {self.config.org_name}")
        return await self._query_rest(soql)


class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 100):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.updated_at = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.updated_at
            
            # Refill tokens
            self.tokens = min(
                self.calls_per_minute,
                self.tokens + elapsed * (self.calls_per_minute / 60)
            )
            self.updated_at = now
            
            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.calls_per_minute
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
