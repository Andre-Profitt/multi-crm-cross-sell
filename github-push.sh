#!/bin/bash
cd ~/Desktop/multi-crm-cross-sell

echo '🚀 Pushing Multi-CRM Cross-Sell to GitHub'
echo '========================================'

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo '✅ GitHub CLI found'
    
    # Check auth status
    if gh auth status &> /dev/null; then
        echo '✅ Already authenticated with GitHub'
    else
        echo '📝 Please authenticate with GitHub:'
        gh auth login
    fi
    
    # Create and push repository
    echo '📤 Creating repository and pushing...'
    gh repo create multi-crm-cross-sell \
        --public \
        --description 'AI-powered cross-sell opportunity identification across multiple CRM systems' \
        --source=. \
        --remote=origin \
        --push
    
    if [ $? -eq 0 ]; then
        echo '✅ Successfully created and pushed to GitHub!'
        echo ''
        echo '🎉 Your repository is now live at:'
        gh repo view --web
    else
        echo '❌ Failed to create repository. It may already exist.'
        echo 'Trying to push to existing repository...'
        
        # Try to add remote and push
        git remote add origin https://github.com/$(gh api user -q .login)/multi-crm-cross-sell.git 2>/dev/null || true
        git branch -M main
        git push -u origin main
    fi
else
    echo '❌ GitHub CLI not found'
    echo 'Installing GitHub CLI...'
    brew install gh
    echo 'Please run this script again after installation'
fi
