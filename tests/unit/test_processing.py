import pandas as pd
import pytest

from src.orchestrator import CrossSellOrchestrator

async def test_industry_normalization_and_feature_matrix():
    orch = CrossSellOrchestrator()
    df = pd.DataFrame({
        "Id": ["1", "2"],
        "Name": ["A", "B"],
        "Industry": ["tech", "Technology"],
        "AnnualRevenue": [100000, 200000],
        "NumberOfEmployees": [10, 20],
        "BillingCountry": ["us", "United States"],
        "CreatedDate": ["2020-01-01", "2021-01-01"],
        "LastActivityDate": ["2021-06-01", "2021-06-02"],
    })
    processed = await orch._process_data({"org1": {"accounts": df}})
    cleaned = processed["org_data"]["org1"]["accounts"]
    assert list(cleaned["Industry"]) == ["Technology", "Technology"]
    assert list(cleaned["BillingCountry"]) == ["USA", "USA"]
    # Feature matrix should have two rows after dedup
    fm = processed["feature_matrix"]
    assert len(fm) == 2

async def test_deduplication():
    orch = CrossSellOrchestrator()
    df = pd.DataFrame({
        "Id": ["1", "1"],
        "Name": ["A", "A"],
        "Industry": ["tech", "tech"],
        "AnnualRevenue": [100000, 100000],
        "NumberOfEmployees": [10, 10],
        "BillingCountry": ["us", "us"],
        "CreatedDate": ["2020-01-01", "2020-01-01"],
        "LastActivityDate": ["2021-06-01", "2021-06-01"],
    })
    processed = await orch._process_data({"org1": {"accounts": df}})
    cleaned = processed["org_data"]["org1"]["accounts"]
    assert len(cleaned) == 1
    fm = processed["feature_matrix"]
    assert len(fm) == 1
