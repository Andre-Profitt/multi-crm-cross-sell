#!/bin/bash
cd ~/Desktop/multi-crm-cross-sell

clear
echo '🚀 GITHUB PUSH HELPER'
echo '===================='
echo ''
echo 'This will push your project to GitHub.'
echo ''
echo 'What is your GitHub username?'
read -p 'GitHub username: ' USERNAME
echo ''
echo 'Have you already created a repository named "multi-crm-cross-sell" on GitHub? (y/n)'
read -p 'Answer: ' CREATED

if [ "$CREATED" = "n" ] || [ "$CREATED" = "N" ]; then
    echo ''
    echo '📝 Please do this NOW:'
    echo '1. Go to https://github.com/new'
    echo '2. Repository name: multi-crm-cross-sell'
    echo '3. Make it Public or Private'
    echo '4. DO NOT initialize with README'
    echo '5. Click Create Repository'
    echo ''
    echo 'Press ENTER when done...'
    read
fi

echo ''
echo '🔄 Adding remote and pushing...'

# Remove existing remote if any
git remote remove origin 2>/dev/null || true

# Add new remote
git remote add origin https://github.com/$USERNAME/multi-crm-cross-sell.git

# Push
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ''
    echo '✅ SUCCESS! Your project is now on GitHub!'
    echo ''
    echo '🎉 View your repository at:'
    echo "https://github.com/$USERNAME/multi-crm-cross-sell"
    echo ''
    open "https://github.com/$USERNAME/multi-crm-cross-sell"
else
    echo ''
    echo '❌ Push failed. This usually means:'
    echo '1. The repository wasn\'t created on GitHub yet'
    echo '2. Your GitHub credentials need to be set up'
    echo ''
    echo 'To set up credentials, run:'
    echo 'git config --global user.name "Your Name"'
    echo 'git config --global user.email "your-email@example.com"'
fi
