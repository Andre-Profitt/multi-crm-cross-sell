"""
Multi-Salesforce Connector for Cross-Selling Opportunities
Handles authentication and data extraction from multiple Salesforce orgs
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import pandas as pd
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(calls_per_minute: int = 60):
    min_interval = 60.0 / calls_per_minute
    last_called = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{args[0].__class__.__name__}_{id(args[0])}"
            current_time = time.time()
            
            if key in last_called:
                elapsed = current_time - last_called[key]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            last_called[key] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class SalesforceOrgConfig:
    """Configuration for a Salesforce organization"""
    org_id: str
    org_name: str
    instance_url: str
    auth_type: str = "password"  # "jwt" or "password"
    
    # Common fields
    client_id: Optional[str] = None
    username: Optional[str] = None
    
    # JWT Bearer Flow
    private_key_path: Optional[str] = None
    
    # Username-Password Flow
    client_secret: Optional[str] = None
    password: Optional[str] = None
    security_token: Optional[str] = None
    
    # Token management
    access_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    
    # Metadata
    api_version: str = "v60.0"

class SalesforceAuthManager:
    """Manages authentication for multiple Salesforce orgs"""
    
    def __init__(self, cache_dir: str = ".sf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.token_cache_file = os.path.join(cache_dir, "tokens.json")
        self._load_token_cache()
    
    def _load_token_cache(self):
        """Load cached tokens from file"""
        try:
            with open(self.token_cache_file, 'r') as f:
                self.token_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.token_cache = {}
    
    def _save_token_cache(self):
        """Save tokens to cache file"""
        with open(self.token_cache_file, 'w') as f:
            json.dump(self.token_cache, f)
    
    def authenticate_password(self, config: SalesforceOrgConfig) -> str:
        """Authenticate using Username-Password Flow"""
        # Check cache first
        if self._is_token_valid(config):
            logger.info(f"Using cached token for {config.org_name}")
            return config.access_token
        
        logger.info(f"Authenticating {config.org_name} via password flow")
        
        response = requests.post(
            f"{config.instance_url}/services/oauth2/token",
            data={
                'grant_type': 'password',
                'client_id': config.client_id,
                'client_secret': config.client_secret,
                'username': config.username,
                'password': f"{config.password}{config.security_token}"
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            config.access_token = token_data['access_token']
            config.token_expiry = datetime.now() + timedelta(hours=2)
            
            # Cache token
            self.token_cache[config.org_id] = {
                'access_token': config.access_token,
                'instance_url': config.instance_url,
                'expiry': config.token_expiry.isoformat()
            }
            self._save_token_cache()
            
            return config.access_token
        else:
            raise Exception(f"Password auth failed for {config.org_name}: {response.text}")
    
    def authenticate(self, config: SalesforceOrgConfig) -> str:
        """Authenticate based on configured auth type"""
        if config.auth_type == "password":
            return self.authenticate_password(config)
        else:
            raise ValueError(f"Unknown auth type: {config.auth_type}")
    
    def _is_token_valid(self, config: SalesforceOrgConfig) -> bool:
        """Check if cached token is still valid"""
        if config.org_id in self.token_cache:
            cached = self.token_cache[config.org_id]
            expiry = datetime.fromisoformat(cached['expiry'])
            
            if datetime.now() < expiry:
                config.access_token = cached['access_token']
                config.instance_url = cached['instance_url']
                config.token_expiry = expiry
                return True
        
        return config.access_token and config.token_expiry and datetime.now() < config.token_expiry

class SalesforceDataExtractor:
    """Extract data from Salesforce with error handling and retries"""
    
    def __init__(self, auth_manager: SalesforceAuthManager):
        self.auth_manager = auth_manager
        self.batch_size = 2000  # Salesforce query limit
    
    @rate_limit(calls_per_minute=100)  # Respect API limits
    def query(self, config: SalesforceOrgConfig, soql: str) -> pd.DataFrame:
        """Execute SOQL query with pagination support"""
        # Ensure authenticated
        token = self.auth_manager.authenticate(config)
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        all_records = []
        next_url = f"{config.instance_url}/services/data/{config.api_version}/query?q={soql}"
        
        while next_url:
            response = requests.get(next_url, headers=headers)
            
            if response.status_code == 401:  # Token expired
                logger.info("Token expired, re-authenticating...")
                token = self.auth_manager.authenticate(config)
                headers['Authorization'] = f'Bearer {token}'
                response = requests.get(next_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                all_records.extend(data['records'])
                
                # Check for more records
                if data.get('done', True):
                    break
                else:
                    next_url = config.instance_url + data['nextRecordsUrl']
            else:
                raise Exception(f"Query failed: {response.text}")
        
        # Convert to DataFrame
        if all_records:
            df = pd.DataFrame(all_records)
            # Remove Salesforce metadata
            df = df.drop(['attributes'], axis=1, errors='ignore')
            df['_org_id'] = config.org_id
            df['_org_name'] = config.org_name
            return df
        else:
            return pd.DataFrame()
    
    def extract_accounts(self, config: SalesforceOrgConfig) -> pd.DataFrame:
        """Extract account data with all relevant fields"""
        soql = """
        SELECT 
            Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
            BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry,
            Website, Phone, Description, OwnerId, CreatedDate, LastModifiedDate,
            LastActivityDate, AccountSource, Rating
        FROM Account
        WHERE IsDeleted = false
        ORDER BY AnnualRevenue DESC NULLS LAST
        """
        
        logger.info(f"Extracting accounts from {config.org_name}")
        return self.query(config, soql)
    
    def extract_opportunities(self, config: SalesforceOrgConfig) -> pd.DataFrame:
        """Extract opportunity data with all relevant fields"""
        soql = """
        SELECT 
            Id, AccountId, Name, Description, StageName, Amount, Probability,
            CloseDate, Type, NextStep, LeadSource, IsClosed, IsWon,
            ForecastCategory, ForecastCategoryName, CampaignId,
            HasOpportunityLineItem, Pricebook2Id, OwnerId,
            CreatedDate, LastModifiedDate, LastActivityDate
        FROM Opportunity
        WHERE IsDeleted = false
        ORDER BY CloseDate DESC
        """
        
        logger.info(f"Extracting opportunities from {config.org_name}")
        return self.query(config, soql)

class CrossOrgDataProcessor:
    """Process and unify data across multiple Salesforce orgs"""
    
    def __init__(self):
        self.unified_data = {}
        
    def standardize_industries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize industry classifications across orgs"""
        industry_mapping = {
            'Technology': ['Technology', 'Tech', 'Software', 'IT', 'Information Technology'],
            'Healthcare': ['Healthcare', 'Health', 'Medical', 'Pharma', 'Pharmaceutical'],
            'Finance': ['Finance', 'Financial', 'Banking', 'Insurance', 'FinTech'],
            'Retail': ['Retail', 'E-commerce', 'Ecommerce', 'Consumer Goods'],
            'Manufacturing': ['Manufacturing', 'Industrial', 'Automotive', 'Machinery']
        }
        
        # Create reverse mapping
        reverse_map = {}
        for standard, variations in industry_mapping.items():
            for var in variations:
                reverse_map[var.lower()] = standard
        
        # Apply mapping
        df['Industry_Standardized'] = df['Industry'].str.lower().map(reverse_map)
        df['Industry_Standardized'].fillna(df['Industry'], inplace=True)
        
        return df
