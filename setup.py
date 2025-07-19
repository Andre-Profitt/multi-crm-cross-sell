from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_path = os.path.join("src", "__init__.py")
    with open(init_path) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')
    return "1.0.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="multi-crm-cross-sell",
    version=get_version(),
    author="Andre Profitt",
    author_email="andre@example.com",
    description="AI-powered cross-sell opportunity identification across multiple CRM systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andre-Profitt/multi-crm-cross-sell",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "cross-sell=main:main",
            "cross-sell-api=src.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.css"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="salesforce crm cross-sell ai machine-learning revops",
)