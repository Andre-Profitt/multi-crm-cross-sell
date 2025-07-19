#!/usr/bin/env python3
import os
import json

print('\n✨ Multi-CRM Cross-Sell Project Setup Validation')
print('=' * 50)

# Check critical files
critical_files = {
    'Root Files': [
        'README.md', 'requirements.txt', 'main.py', 'setup.py',
        'Dockerfile', 'docker-compose.yml', '.env.example', '.gitignore'
    ],
    'Source Code': [
        'src/__init__.py',
        'src/orchestrator.py',
        'src/connectors/salesforce.py',
        'src/ml/pipeline.py', 
        'src/api/main.py',
        'src/visualization/dashboard.py'
    ],
    'Configuration': [
        'config/orgs.example.json',
        'config/ml_config.yaml'
    ],
    'GitHub': [
        '.github/workflows/ci.yml'
    ]
}

all_good = True
for category, files in critical_files.items():
    print(f'\n{category}:')
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f'  ✅ {file} ({size} bytes)')
        else:
            print(f'  ❌ {file} - MISSING')
            all_good = False

# Check directories
print('\n📁 Directory Structure:')
dirs = ['src', 'config', 'docs', 'k8s', 'tests', 'scripts', 'data/sample']
for d in dirs:
    if os.path.exists(d):
        count = len([f for f in os.listdir(d) if not f.startswith('.')])
        print(f'  ✅ {d}/ ({count} items)')
    else:
        print(f'  ❌ {d}/ - MISSING')
        all_good = False

if all_good:
    print('\n🎉 All critical components are in place!')
else:
    print('\n⚠️  Some components are missing. Run complete_setup.sh to fix.')

print('\n📝 Next Steps:')
print('  1. Review and update personal info in LICENSE, README.md, setup.py')
print('  2. Create virtual environment: python -m venv venv')
print('  3. Activate it: source venv/bin/activate')
print('  4. Install dependencies: pip install -r requirements.txt')
print('  5. Configure Salesforce: cp config/orgs.example.json config/orgs.json')
print('  6. Set up git remote and push to GitHub')
