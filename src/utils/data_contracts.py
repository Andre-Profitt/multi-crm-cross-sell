"""
Data Contracts for Cross-Sell Platform

Defines expected schemas for ingested data with validation.
Ensures data quality at system boundaries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnContract:
    """Contract for a single column"""
    name: str
    dtype: str  # 'string', 'numeric', 'datetime', 'boolean'
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[str]] = None


@dataclass
class DataContract:
    """Contract for a DataFrame"""
    name: str
    columns: List[ColumnContract]
    required_columns: List[str]
    min_rows: int = 0
    max_rows: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of contract validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    null_counts: Dict[str, int]


# Define contracts for core entities
ACCOUNT_CONTRACT = DataContract(
    name="accounts",
    required_columns=["Id", "Name"],
    columns=[
        ColumnContract("Id", "string", nullable=False, unique=True),
        ColumnContract("Name", "string", nullable=False),
        ColumnContract("Industry", "string", nullable=True),
        ColumnContract("AnnualRevenue", "numeric", nullable=True, min_value=0),
        ColumnContract("NumberOfEmployees", "numeric", nullable=True, min_value=0),
        ColumnContract("BillingCountry", "string", nullable=True),
        ColumnContract("Type", "string", nullable=True,
                      allowed_values={"Customer", "Prospect", "Partner", "Other"}),
        ColumnContract("Rating", "string", nullable=True,
                      allowed_values={"Hot", "Warm", "Cold"}),
        ColumnContract("CreatedDate", "datetime", nullable=True),
        ColumnContract("LastActivityDate", "datetime", nullable=True),
    ],
    min_rows=1,
)

OPPORTUNITY_CONTRACT = DataContract(
    name="opportunities",
    required_columns=["Id", "AccountId", "Name"],
    columns=[
        ColumnContract("Id", "string", nullable=False, unique=True),
        ColumnContract("AccountId", "string", nullable=False),
        ColumnContract("Name", "string", nullable=False),
        ColumnContract("Amount", "numeric", nullable=True, min_value=0),
        ColumnContract("StageName", "string", nullable=True),
        ColumnContract("CloseDate", "datetime", nullable=True),
        ColumnContract("Probability", "numeric", nullable=True, min_value=0, max_value=100),
        ColumnContract("IsClosed", "boolean", nullable=True),
        ColumnContract("IsWon", "boolean", nullable=True),
    ],
    min_rows=0,
)

RECOMMENDATION_CONTRACT = DataContract(
    name="recommendations",
    required_columns=["account1_id", "account2_id", "score"],
    columns=[
        ColumnContract("account1_id", "string", nullable=False),
        ColumnContract("account2_id", "string", nullable=False),
        ColumnContract("score", "numeric", nullable=False, min_value=0, max_value=1),
        ColumnContract("confidence_level", "string", nullable=True,
                      allowed_values={"Very High", "High", "Medium", "Low"}),
        ColumnContract("estimated_value", "numeric", nullable=True, min_value=0),
    ],
    min_rows=0,
)


def validate_dataframe(df: pd.DataFrame, contract: DataContract) -> ValidationResult:
    """
    Validate a DataFrame against a data contract.

    Args:
        df: DataFrame to validate
        contract: Contract to validate against

    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    null_counts = {}

    # Check row count
    if len(df) < contract.min_rows:
        errors.append(f"Row count {len(df)} below minimum {contract.min_rows}")

    if contract.max_rows and len(df) > contract.max_rows:
        warnings.append(f"Row count {len(df)} exceeds maximum {contract.max_rows}")

    # Check required columns exist
    for col in contract.required_columns:
        if col not in df.columns:
            errors.append(f"Required column '{col}' missing")

    # Validate each column
    for col_contract in contract.columns:
        if col_contract.name not in df.columns:
            if col_contract.name in contract.required_columns:
                # Already reported above
                continue
            warnings.append(f"Expected column '{col_contract.name}' not found")
            continue

        col_data = df[col_contract.name]
        null_count = col_data.isna().sum()
        null_counts[col_contract.name] = int(null_count)

        # Check nullability
        if not col_contract.nullable and null_count > 0:
            errors.append(f"Column '{col_contract.name}' has {null_count} null values but is not nullable")

        # Check uniqueness
        if col_contract.unique:
            duplicates = col_data.dropna().duplicated().sum()
            if duplicates > 0:
                errors.append(f"Column '{col_contract.name}' has {duplicates} duplicate values but should be unique")

        # Type-specific validations
        non_null_data = col_data.dropna()
        if len(non_null_data) == 0:
            continue

        if col_contract.dtype == "numeric":
            if col_contract.min_value is not None:
                below_min = (non_null_data < col_contract.min_value).sum()
                if below_min > 0:
                    errors.append(f"Column '{col_contract.name}' has {below_min} values below minimum {col_contract.min_value}")

            if col_contract.max_value is not None:
                above_max = (non_null_data > col_contract.max_value).sum()
                if above_max > 0:
                    errors.append(f"Column '{col_contract.name}' has {above_max} values above maximum {col_contract.max_value}")

        elif col_contract.dtype == "string" and col_contract.allowed_values:
            invalid = ~non_null_data.isin(col_contract.allowed_values)
            invalid_count = invalid.sum()
            if invalid_count > 0:
                invalid_examples = non_null_data[invalid].head(3).tolist()
                warnings.append(
                    f"Column '{col_contract.name}' has {invalid_count} values outside allowed set. "
                    f"Examples: {invalid_examples}"
                )

    is_valid = len(errors) == 0

    if errors:
        logger.error(f"Validation failed for {contract.name}: {errors}")
    if warnings:
        logger.warning(f"Validation warnings for {contract.name}: {warnings}")

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
        null_counts=null_counts,
    )


class DataQualityMonitor:
    """
    Monitor data quality across pipeline runs.
    Tracks metrics over time for anomaly detection.
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record(self, contract_name: str, result: ValidationResult, timestamp: Optional[datetime] = None):
        """Record a validation result"""
        self._history.append({
            "contract": contract_name,
            "timestamp": timestamp or datetime.utcnow(),
            "is_valid": result.is_valid,
            "row_count": result.row_count,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "null_counts": result.null_counts,
        })

    def get_summary(self, contract_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics"""
        history = self._history
        if contract_name:
            history = [h for h in history if h["contract"] == contract_name]

        if not history:
            return {"total_validations": 0}

        return {
            "total_validations": len(history),
            "success_rate": sum(1 for h in history if h["is_valid"]) / len(history),
            "avg_row_count": sum(h["row_count"] for h in history) / len(history),
            "total_errors": sum(h["error_count"] for h in history),
            "total_warnings": sum(h["warning_count"] for h in history),
        }


# Watermark tracking for incremental syncs
class SyncWatermark:
    """
    Track sync watermarks for incremental data extraction.

    Usage:
        watermark = SyncWatermark("org1", "accounts")
        last_sync = watermark.get()  # Returns last sync timestamp

        # After successful sync:
        watermark.update(new_timestamp)
    """

    def __init__(self, org_id: str, entity_type: str, storage_path: str = ".sync_watermarks"):
        self.org_id = org_id
        self.entity_type = entity_type
        self.storage_path = storage_path
        self._watermarks: Dict[str, datetime] = {}

    def _key(self) -> str:
        return f"{self.org_id}:{self.entity_type}"

    def get(self) -> Optional[datetime]:
        """Get the last sync watermark"""
        return self._watermarks.get(self._key())

    def update(self, timestamp: datetime):
        """Update the watermark after successful sync"""
        self._watermarks[self._key()] = timestamp
        logger.info(f"Updated watermark for {self._key()}: {timestamp}")

    def get_incremental_filter(self) -> Optional[str]:
        """
        Get SOQL filter clause for incremental extraction.

        Returns:
            SOQL WHERE clause fragment or None if no watermark exists
        """
        last_sync = self.get()
        if last_sync:
            return f"LastModifiedDate > {last_sync.isoformat()}Z"
        return None
