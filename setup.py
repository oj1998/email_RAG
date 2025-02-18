from setuptools import setup, find_packages

setup(
    name="email_rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "pydantic",
        "langchain-community",
        "huggingface-hub",
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "google-api-python-client",
        "pandas",
        "beautifulsoup4",
        "python-multipart",
        "supabase",
        "sqlalchemy",
        "chromadb",
        "pgvector"
    ],
)
