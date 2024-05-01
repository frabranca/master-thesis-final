#!/bin/bash

# Check if there are changes to commit
if [[ -n $(git status -s) ]]; then
    # Add all changes
    git add .

    # Commit changes with a default message
    git commit -m "Auto-commit: $(date)"

    # Push changes to the remote repository
    git push # Replace 'master' with your branch name if needed
else
    echo "No changes to commit."
fi
