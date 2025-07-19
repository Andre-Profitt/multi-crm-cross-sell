#!/bin/bash
# final_github_push.sh - Complete the GitHub setup for Andre Profitt

echo '🚀 Pushing Multi-CRM Cross-Sell to GitHub'
echo '========================================'

# Ensure we are on main branch
git branch -M main

# Push to GitHub
echo '📤 Pushing to GitHub...'
git push -u origin main

if [ $? -eq 0 ]; then
    echo ''
    echo '✅ Successfully pushed to GitHub!'
    echo ''
    echo '🎉 Your repository is now live at:'
    echo '   https://github.com/Andre-Profitt/multi-crm-cross-sell'
    echo ''
    open 'https://github.com/Andre-Profitt/multi-crm-cross-sell'
else
    echo ''
    echo '❌ Push failed. Please ensure you have created the repository on GitHub.'
fi
