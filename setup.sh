#!/bin/bash
set -e

echo '🚀 Setting up Multi-CRM Cross-Sell Intelligence Platform'
echo '========================================================'

# Create directory structure
echo '📁 Creating project structure...'
mkdir -p src/connectors src/ml src/api src/visualization src/utils
mkdir -p config data/sample docs k8s keys logs models notebooks outputs reports scripts
mkdir -p tests/unit tests/integration tests/ml tests/performance
mkdir -p .github/workflows exports

# Create Python package files
echo '🐍 Creating Python packages...'
echo '__version__ = "1.0.0"' > src/__init__.py
touch src/connectors/__init__.py
touch src/ml/__init__.py
touch src/api/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

# Create config example
echo '⚙️ Creating configuration files...'
cat > .env.example << 'ENVEOF'
DATABASE_URL=postgresql://crosssell_user:password@localhost:5432/crosssell_db
REDIS_URL=redis://localhost:6379/0
SALESFORCE_CLIENT_ID=your_client_id
SALESFORCE_CLIENT_SECRET=your_client_secret
JWT_SECRET_KEY=your_secret_key
LOG_LEVEL=INFO
ENVEOF

echo '✅ Setup complete!'
