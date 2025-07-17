from setuptools import setup, find_packages

setup(
    name="sense-rag",
    version="0.2.0-LangGraph",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.5.3",
        "langchain>=0.1.0",
        "langgraph-api>=0.2.86",
        "aiofiles>=23.2.1",
    ],
) 