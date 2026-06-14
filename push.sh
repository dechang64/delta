#!/bin/bash
# Push delta_jfe to GitHub
# Usage: Set GITHUB_TOKEN environment variable first
#   export GITHUB_TOKEN=ghp_your_token_here
#   bash push.sh

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN not set"
    echo "Run: export GITHUB_TOKEN=ghp_your_token_here"
    exit 1
fi

cd "$(dirname "$0")"
git remote set-url origin "https://dechang64:${GITHUB_TOKEN}@github.com/dechang64/delta_jfe.git"
git push origin main
echo "Push complete!"
