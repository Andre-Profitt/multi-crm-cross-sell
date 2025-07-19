"""
Base CRM Connector Interface
Defines the contract all CRM connectors must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class CRMConfig:
    """Base configuration for CRM connections"""
    org_id: str
    org_name: str
    instance_url: str
    api_version: Optional[str] = None
    
    def __post_init__(self):
        if not self.org_id:
            raise ValueError("org_id is required")
        if not self.instance_url:
            raise ValueError("instance_url is required")


class DataExtractor(Protocol):
    """Protocol for data extraction methods"""
    async def extract_accounts(self) -> pd.DataFrame:
        ...
    
    async def extract_opportunities(self) -> pd.DataFrame:
        ...
    
    async def extract_products(self) -> pd.DataFrame:
        ...


class BaseCRMConnector(ABC):
    """
    Abstract base class for CRM connectors.
    All CRM connectors must inherit from this class.
    """
    
    def __init__(self, config: CRMConfig):
        self.config = config
        self._authenticated = False
        self._session = None
        
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the CRM system.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection is working"""
        pass
    
    @abstractmethod
    async def extract_accounts(self, 
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract account data from the CRM.
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with account data
        """
        pass
    
    @abstractmethod
    async def extract_opportunities(self,
                                  filters: Optional[Dict[str, Any]] = None,
                                  limit: Optional[int] = None) -> pd.DataFrame:
        """Extract opportunity data from the CRM"""
        pass
    
    @abstractmethod
    async def extract_products(self,
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Extract product data from the CRM"""
        pass
    
    @abstractmethod
    async def extract_contacts(self,
                             filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Extract contact data from the CRM"""
        pass
    
    async def extract_all(self) -> Dict[str, pd.DataFrame]:
        """
        Extract all relevant data from the CRM.
        Default implementation calls all extract methods.
        """
        if not self._authenticated:
            await self.authenticate()
        
        tasks = {
            'accounts': self.extract_accounts(),
            'opportunities': self.extract_opportunities(),
            'products': self.extract_products(),
            'contacts': self.extract_contacts()
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                logger.info(f"Extracting {name} from {self.config.org_name}")
                results[name] = await task
                logger.info(f"Extracted {len(results[name])} {name} records")
            except Exception as e:
                logger.error(f"Failed to extract {name}: {str(e)}")
                results[name] = pd.DataFrame()
        
        return results
    
    async def close(self):
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None
        self._authenticated = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns to extracted data"""
        df['_org_id'] = self.config.org_id
        df['_org_name'] = self.config.org_name
        df['_extracted_at'] = datetime.utcnow()
        return df


class ConnectorRegistry:
    """Registry for available CRM connectors"""
    
    _connectors: Dict[str, type[BaseCRMConnector]] = {}
    
    @classmethod
    def register(cls, name: str, connector_class: type[BaseCRMConnector]):
        """Register a new connector type"""
        if not issubclass(connector_class, BaseCRMConnector):
            raise TypeError(f"{connector_class} must inherit from BaseCRMConnector")
        cls._connectors[name.lower()] = connector_class
        logger.info(f"Registered connector: {name}")
    
    @classmethod
    def get_connector(cls, name: str, config: CRMConfig) -> BaseCRMConnector:
        """Get a connector instance by name"""
        connector_class = cls._connectors.get(name.lower())
        if not connector_class:
            raise ValueError(f"Unknown connector type: {name}")
        return connector_class(config)
    
    @classmethod
    def list_connectors(cls) -> List[str]:
        """List all registered connector types"""
        return list(cls._connectors.keys())


# Decorator for auto-registration
def register_connector(name: str):
    """Decorator to register a connector class"""
    def decorator(cls):
        ConnectorRegistry.register(name, cls)
        return cls
    return decorator
