from setuptools import setup, find_packages

setup(
    name="sense-rag",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain",
        "langgraph-api",
    ],
) 