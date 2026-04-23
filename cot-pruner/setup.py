from setuptools import setup, find_packages

setup(
    name="cot-pruner",
    version="0.4.0",
    author="Your Name",
    description="CoT Pruner - MI + Perturbation based Chain of Thought Pruning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
    ],
)
