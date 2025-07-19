#!/usr/bin/env python3
"""Generate sample data for testing the cross-sell platform."""

import asyncio
import random
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import Base, Organization, Account, Opportunity, init_db

fake = Faker()

def generate_accounts(org_id: str, count: int = 100):
    """Generate sample accounts."""
    accounts = []
    industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
    types = ["Customer", "Prospect", "Partner"]
    ratings = ["Hot", "Warm", "Cold"]
    
    for i in range(count):
        account = {
            "id": f"{org_id}_acc_{i}",
            "salesforce_id": f"001{fake.random_number(digits=15)}",
            "org_id": org_id,
            "name": fake.company(),
            "industry": random.choice(industries),
            "annual_revenue": random.randint(100000, 10000000),
            "number_of_employees": random.randint(10, 5000),
            "billing_country": fake.country(),
            "billing_state": fake.state(),
            "billing_city": fake.city(),
            "website": fake.url(),
            "type": random.choice(types),
            "rating": random.choice(ratings),
            "created_date": fake.date_time_between(start_date="-2y", end_date="now"),
            "last_activity_date": fake.date_time_between(start_date="-30d", end_date="now")
        }
        accounts.append(account)
    
    return accounts

def main():
    """Generate sample data."""
    print("🚀 Generating sample data...")
    
    # Initialize database
    engine = init_db()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create organizations
    orgs = [
        {"id": "org1", "name": "TechCorp Solutions", "instance_url": "https://techcorp.salesforce.com"},
        {"id": "org2", "name": "Global Finance Inc", "instance_url": "https://globalfinance.salesforce.com"},
        {"id": "org3", "name": "Healthcare Plus", "instance_url": "https://healthplus.salesforce.com"}
    ]
    
    for org_data in orgs:
        org = Organization(**org_data)
        session.merge(org)
    
    session.commit()
    print(f"✅ Created {len(orgs)} organizations")
    
    # Generate accounts for each org
    total_accounts = 0
    for org in orgs:
        accounts = generate_accounts(org["id"], count=50)
        for acc_data in accounts:
            account = Account(**acc_data)
            session.merge(account)
        total_accounts += len(accounts)
    
    session.commit()
    print(f"✅ Created {total_accounts} accounts")
    
    # Add some opportunities
    opportunities_count = 0
    accounts = session.query(Account).all()
    for account in random.sample(accounts, min(30, len(accounts))):
        for i in range(random.randint(1, 3)):
            opp = Opportunity(
                id=f"{account.org_id}_opp_{opportunities_count}",
                salesforce_id=f"006{fake.random_number(digits=15)}",
                account_id=account.id,
                name=f"{account.name} - {fake.catch_phrase()}",
                amount=random.randint(10000, 500000),
                stage_name=random.choice(["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won"]),
                close_date=fake.future_date(end_date="+90d"),
                probability=random.randint(10, 90),
                type=random.choice(["New Business", "Existing Business", "Renewal"]),
                created_date=datetime.now()
            )
            session.add(opp)
            opportunities_count += 1
    
    session.commit()
    print(f"✅ Created {opportunities_count} opportunities")
    
    print("\n📊 Sample data generated successfully!")
    print("\nNext steps:")
    print("1. Run the pipeline: python main.py --run-once")
    print("2. Check the API: http://localhost:8000/api/recommendations")
    print("3. View dashboard: http://localhost:8501")

if __name__ == "__main__":
    main()
