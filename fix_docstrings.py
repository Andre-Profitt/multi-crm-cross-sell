#!/usr/bin/env python3
"""Quick fix for missing docstrings."""

import os
import re

def fix_test_files():
    """Add missing docstrings to test files."""
    test_files = [
        'tests/unit/test_api.py',
        'tests/unit/test_connectors.py'
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Fix module docstring
            if content.startswith('"""'):
                lines = content.split('\n')
                if not lines[0].endswith('.'):
                    lines[0] = lines[0].rstrip('"') + '."""'
                    content = '\n'.join(lines)
            
            # Fix class docstrings
            content = re.sub(
                r'class (\w+).*:\n(\s+)def',
                r'class \1:\n\2"""Test class."""\n\2def',
                content
            )
            
            # Fix method docstrings
            content = re.sub(
                r'def test_(\w+)\(.*\):\n(\s+)"""([^.])([^"]*?)"""',
                r'def test_\1(self):\n\2"""\3\4."""',
                content
            )
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f'Fixed {filepath}')

if __name__ == '__main__':
    fix_test_files()
