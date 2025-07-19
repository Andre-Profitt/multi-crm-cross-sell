#!/usr/bin/env python3
import subprocess
import sys
import os

print('🔍 Streamlit Diagnostic Tool')
print('=' * 40)

# Check Python and pip
print(f'Python: {sys.executable}')
print(f'Python version: {sys.version.split()[0]}')

# Check if in virtual environment
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f'Virtual environment active: {in_venv}')

# Check streamlit installation
try:
    import streamlit as st
    print(f'✅ Streamlit version: {st.__version__}')
except ImportError:
    print('❌ Streamlit not installed')
    print('Run: pip install streamlit')

# Check other dependencies
deps = ['pandas', 'numpy', 'plotly', 'psycopg2']
for dep in deps:
    try:
        __import__(dep)
        print(f'✅ {dep} installed')
    except ImportError:
        print(f'❌ {dep} not installed')

# Try to find dashboard file
dashboard_path = 'src/visualization/dashboard.py'
if os.path.exists(dashboard_path):
    print(f'✅ Dashboard file exists: {dashboard_path}')
else:
    print(f'❌ Dashboard file not found: {dashboard_path}')

# Check if port 8501 is in use
try:
    result = subprocess.run(['lsof', '-i', ':8501'], capture_output=True, text=True)
    if result.stdout:
        print('⚠️  Port 8501 is in use:')
        print(result.stdout)
    else:
        print('✅ Port 8501 is available')
except:
    print('Could not check port status')

print('\nTo run dashboard:')
print('streamlit run test_dashboard.py')
print('or')
print('streamlit run src/visualization/dashboard.py')
