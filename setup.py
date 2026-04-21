from setuptools import setup, find_packages

setup(
    name="adaptive-tutor-rl",
    version="1.0.0",
    description="Reinforcement Learning for Adaptive Tutorial Agents",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "networkx>=3.1",
        "scikit-learn>=1.3.0",
        "anthropic>=0.30.0",
    ],
)
