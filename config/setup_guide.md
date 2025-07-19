# Salesforce Setup Guide

## 1. Create a Connected App in Salesforce
1. Go to Setup → Apps → App Manager
2. Click "New Connected App"
3. Fill in:
   - Connected App Name: "Cross-Sell Intelligence"
   - API Name: "Cross_Sell_Intelligence"
   - Contact Email: your-email@example.com
4. Enable OAuth Settings:
   - Callback URL: http://localhost:8000/callback
   - Selected OAuth Scopes:
     - Access and manage your data (api)
     - Perform requests on your behalf at any time (refresh_token, offline_access)
5. Save and note the Consumer Key (Client ID) and Consumer Secret

## 2. Update config/orgs.json
Replace the example with your actual Salesforce org:

```json
{
    "organizations": [
        {
            "org_id": "your_company",
            "org_name": "Your Company Name",
            "instance_url": "https://yourcompany.my.salesforce.com",
            "auth_type": "password",
            "client_id": "YOUR_CONSUMER_KEY",
            "client_secret": "YOUR_CONSUMER_SECRET",
            "username": "your.email@company.com",
            "password": "your_password",
            "security_token": "your_security_token"
        }
    ]
}
```

## 3. Get Your Security Token
1. In Salesforce: Your Name → My Settings → Personal → Reset My Security Token
2. Check your email for the token
3. Add it to the config

## 4. Test the Connection
Run: python -c "from src.orchestrator import CrossSellOrchestrator; o = CrossSellOrchestrator(); import asyncio; asyncio.run(o.run_pipeline())"
