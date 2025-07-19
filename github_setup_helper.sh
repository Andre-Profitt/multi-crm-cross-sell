#!/bin/bash

echo '🚀 GitHub Repository Setup Helper'
echo '================================'

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo '✅ GitHub CLI detected!'
    echo ''
    echo 'Creating repository using GitHub CLI...'
    gh repo create multi-crm-cross-sell --public --source=. --remote=origin --push --description "AI-powered cross-sell opportunity identification across multiple CRM systems" || echo 'Repository may already exist'
else
    echo '❌ GitHub CLI not found.'
    echo ''
    echo '📋 Manual Steps Required:'
    echo '1. Go to: https://github.com/new'
    echo '2. Create repository: multi-crm-cross-sell'
    echo '3. Make it PUBLIC'
    echo '4. DO NOT initialize with README'
    echo ''
    echo 'Then run: git push -u origin main'
    echo ''
    echo '🔑 If authentication fails:'
    echo '1. Create a Personal Access Token:'
    echo '   https://github.com/settings/tokens/new'
    echo '2. Select scopes: repo (all)'
    echo '3. Use token as password when pushing'
fi

echo ''
echo '🌐 Opening GitHub...'
open 'https://github.com/Andre-Profitt'
