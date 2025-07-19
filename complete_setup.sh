#!/bin/bash
# complete_setup.sh - Complete setup and organization for Multi-CRM Cross-Sell project

set -e

echo '🚀 Complete Multi-CRM Cross-Sell Project Setup'
echo '=============================================='

# Create all required directories
echo '📁 Creating directory structure...'
mkdir -p src/{connectors,ml,api,visualization,utils}
mkdir -p {config,docs,k8s,keys,logs,models,notebooks,outputs,reports,scripts,exports}
mkdir -p data/sample
mkdir -p tests/{unit,integration,ml,performance}
mkdir -p .github/workflows

# Create __init__.py files
echo ''
echo '🐍 Creating Python package files...'
echo '__version__ = "1.0.0"' > src/__init__.py
touch src/{connectors,ml,api,visualization,utils}/__init__.py
touch tests/{__init__.py,unit/__init__.py,integration/__init__.py,ml/__init__.py,performance/__init__.py}

# Move files to correct locations
echo ''
echo '📦 Organizing files...'

# Move Python files if they exist
[ -f 'enhanced-salesforce-connector.py' ] && mv 'enhanced-salesforce-connector.py' src/connectors/salesforce.py
[ -f 'salesforce-connector-starter.py' ] && mv 'salesforce-connector-starter.py' src/connectors/salesforce_simple.py
[ -f 'fastapi-endpoints.py' ] && mv 'fastapi-endpoints.py' src/api/main.py
[ -f 'streamlit-dashboard.py' ] && mv 'streamlit-dashboard.py' src/visualization/dashboard.py
[ -f 'testing-framework.py' ] && mv 'testing-framework.py' tests/integration/test_integration.py
[ -f 'ml_pipeline.py' ] && mv 'ml_pipeline.py' src/ml/pipeline.py
[ -f 'orchestrator.py' ] && [ ! -f 'src/orchestrator.py' ] && mv orchestrator.py src/orchestrator.py

echo '✅ Setup organizing complete!'
