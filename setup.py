from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi-crm-cross-sell",
    version="1.0.0",
    author="Andre Profitt",
    author_email="andre@example.com",
    description="AI-powered cross-sell opportunity identification across multiple CRM systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andre-Profitt/multi-crm-cross-sell",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.103.0",
        "streamlit>=1.27.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "simple-salesforce>=1.12.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
