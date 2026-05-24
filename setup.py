"""
Setup configuration for Customer Churn Prediction package

This allows the package to be installed with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    description="Professional ML pipeline for predicting customer churn in telecom industry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/customer-churn-prediction",
    license="MIT",
    
    packages=find_packages(exclude=["tests", "notebook", "notebooks"]),
    
    python_requires=">=3.8",
    
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
        "scipy>=1.10.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "all": [
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "gunicorn>=21.2.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "churn-train=main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords="machine-learning churn-prediction classification telecom",
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/customer-churn-prediction/issues",
        "Source": "https://github.com/yourusername/customer-churn-prediction",
    },
)
