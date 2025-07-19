#!/usr/bin/env python3
import sys
import os

print('🔍 Multi-CRM Cross-Sell Setup Verification')
print('=' * 40)

# Check Python version
print(f'\n✅ Python version: {sys.version.split()[0]}')

# Check virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print('✅ Virtual environment is active')
else:
    print('❌ Virtual environment not active - run: source venv/bin/activate')

# Check imports
try:
    import pandas
    print('✅ pandas installed')
except ImportError:
    print('❌ pandas not installed')

try:
    import streamlit
    print('✅ streamlit installed')
except ImportError:
    print('❌ streamlit not installed')

try:
    import fastapi
    print('✅ fastapi installed')
except ImportError:
    print('❌ fastapi not installed')

# Check configuration files
print('\n📁 Configuration files:')
if os.path.exists('config/orgs.json'):
    print('✅ config/orgs.json exists')
    if os.path.getsize('config/orgs.json') > 100:
        print('   ℹ️  File has content - check if credentials are added')
else:
    print('❌ config/orgs.json missing')

if os.path.exists('.env'):
    print('✅ .env exists')
else:
    print('❌ .env missing')

# Check source files
print('\n📄 Source files:')
if os.path.exists('src/orchestrator.py'):
    print('✅ src/orchestrator.py found')
else:
    print('❌ src/orchestrator.py missing')

if os.path.exists('src/visualization/dashboard.py'):
    print('✅ src/visualization/dashboard.py found')
else:
    print('❌ src/visualization/dashboard.py missing')

print('\n✨ Setup verification complete!')
