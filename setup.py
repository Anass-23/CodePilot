from setuptools import setup, find_packages

setup(
    name="codepilot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "faiss-cpu",
        "numpy<2",
        "requests",
        "torch",
        "transformers",
        "sentencepiece",
        "streamlit",
    ],
    entry_points={
        'console_scripts': [
            'codepilot=codepilot.cli:main',
            'codepilot-ui=codepilot.ui.launcher:main',
        ],
    },
    author="Anass Anahri, Eric Muthomi, Pol Vidal",
    author_email="upc@upc.edu",
    description="RAG system for Python codebases using code-specific embeddings and FAISS",
    keywords="rag, huggingface, faiss, code-search, codebert",
    python_requires=">=3.7",
)
