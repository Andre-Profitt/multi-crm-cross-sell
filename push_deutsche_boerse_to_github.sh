#!/bin/bash
# Script to push Deutsche Barse enhancements to GitHub

echo 'Preparing Deutsche Barse enhancements for GitHub...'

# Add all new files
git add src/utils/dedupe.py src/utils/secrets_manager.py src/genai/ src/connectors/salesforce_bulk_concurrent.py src/orchestrator_enhanced.py demo_deutsche_boerse.sh *.md main.py src/api/main.py requirements.txt

# Commit
git commit -m 'feat: Add Deutsche Barse enterprise enhancements'

echo 'Ready to push! Run: git push origin main'
