#!/usr/bin/env python3
import os
import re

# Read the kubernetes deployment file if it exists
if os.path.exists('kubernetes-deployment.txt'):
    with open('kubernetes-deployment.txt', 'r') as f:
        content = f.read()
    
    # Split by '---' and '# filename.yaml' patterns
    sections = re.split(r'\n---\n|\n# (k8s/\w+\.yaml)\n', content)
    
    current_file = None
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if section.startswith('k8s/') and section.endswith('.yaml'):
            current_file = section
        elif current_file and section:
            # Write the content to the file
            os.makedirs('k8s', exist_ok=True)
            with open(current_file, 'w') as f:
                f.write(section + '\n')
            print(f'Created {current_file}')
            current_file = None
        elif 'apiVersion:' in section or 'kind:' in section:
            # Try to extract the resource type for filename
            kind_match = re.search(r'kind:\s+(\w+)', section)
            name_match = re.search(r'name:\s+(\w+)', section)
            if kind_match:
                kind = kind_match.group(1).lower()
                name = name_match.group(1) if name_match else kind
                filename = f'k8s/{name}.yaml'
                os.makedirs('k8s', exist_ok=True)
                with open(filename, 'w') as f:
                    f.write(section + '\n')
                print(f'Created {filename}')
    print('Kubernetes files split successfully!')
else:
    print('kubernetes-deployment.txt not found')
