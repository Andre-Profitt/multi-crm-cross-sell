#!/bin/bash
# push-to-github.sh - Run this after creating the repository on GitHub

echo '🚀 Pushing to GitHub...'
echo 'Enter your GitHub username:'
read GITHUB_USERNAME

# Add remote origin
git remote add origin https://github.com/$GITHUB_USERNAME/multi-crm-cross-sell.git

# Push to main branch
git branch -M main
git push -u origin main

echo '✅ Successfully pushed to GitHub!'
echo ''
echo 'Your repository is now live at:'
echo "https://github.com/$GITHUB_USERNAME/multi-crm-cross-sell"
echo ''
echo 'Next steps:'
echo '1. Add your Salesforce credentials to config/orgs.json'
echo '2. Create a virtual environment: python -m venv venv'
echo '3. Activate it: source venv/bin/activate'
echo '4. Install dependencies: pip install -r requirements.txt'
echo '5. Run the project: python main.py --run-once'
