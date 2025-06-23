"""Setup script for the Codebase Indexing Solution."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="codebase-indexing-solution",
    version="1.0.0",
    author="Codebase Indexing Team",
    author_email="team@codebaseindexing.com",
    description="A comprehensive solution for indexing, searching, and visualizing codebases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/codebase-indexing-solution",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "tree-sitter>=0.20.4",
        "tree-sitter-python>=0.20.4",
        "tree-sitter-javascript>=0.20.4",
        "tree-sitter-typescript>=0.20.4",
        "qdrant-client>=1.7.0",
        "neo4j>=5.15.0",
        "openai>=1.3.7",
        "requests>=2.31.0",
        "numpy>=1.24.3",
        "sentence-transformers>=2.2.2",
        "ollama>=0.1.7",
        "click>=8.1.7",
        "tqdm>=4.66.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "codebase-indexer=src.indexer:main",
            "codebase-server=src.main:main",
        ],
    },
)
