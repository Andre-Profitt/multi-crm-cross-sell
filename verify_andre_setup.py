#!/usr/bin/env python3
import os
import re

print('\n🔍 Verifying Personal Information Updates')
print('=' * 50)

files_to_check = {
    'LICENSE': 'Andre Profitt',
    'README.md': ['Andre-Profitt', 'andre-profitt', 'linkedin.com/in/andre-profitt'],
    'setup.py': ['Andre Profitt', 'Andre-Profitt/multi-crm-cross-sell'],
    'src/__init__.py': 'Andre Profitt'
}

all_good = True
for file, patterns in files_to_check.items():
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        
        if isinstance(patterns, list):
            for pattern in patterns:
                if pattern in content:
                    print(f'✅ {file}: Found "{pattern}"')
                else:
                    print(f'❌ {file}: Missing "{pattern}"')
                    all_good = False
        else:
            if patterns in content:
                print(f'✅ {file}: Found "{patterns}"')
            else:
                print(f'❌ {file}: Missing "{patterns}"')
                all_good = False
    else:
        print(f'❌ {file}: File not found')
        all_good = False

print('\n📊 Git Repository Status:')
os.system('git remote -v | head -2')

if all_good:
    print('\n✅ All personal information has been updated correctly!')
else:
    print('\n⚠️  Some updates may be missing.')

print('\n🚀 Ready to push to GitHub!')
print('\nNext steps:')
print('1. Create the repository on GitHub: https://github.com/new')
print('2. Repository name: multi-crm-cross-sell')  
print('3. Then run: git push -u origin main')
print('\n💡 Don\'t forget to update the email in setup.py!')
