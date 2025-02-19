from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loaders.email_loader import EmailLoader

class SimpleEmailPipeline:
    """
    Simplified pipeline that only handles email ingestion and embedding
    """
    def __init__(
        self, 
        embeddings_model: Embeddings,
        vector_store: VectorStore
    ):
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store

    def ingest_emails(self, gmail_client, search_options) -> int:
        """
        Process emails and store their embeddings
        
        Returns:
            int: Number of documents processed
        """
        # Load and process emails
        loader = EmailLoader(gmail_client, search_options)
        documents = loader.load_and_split()
        
        # Extract texts and metadata separately to avoid ID conflicts
        texts = [doc.page_content for doc in documents]
        metadatas = [
            {k: v for k, v in doc.metadata.items() if k != 'id'}  # Exclude any potential id field
            for doc in documents
        ]
        
        # Store documents with embeddings, avoiding the add_documents method
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        
        return len(documents)

# Usage example:
"""
from langchain_community.embeddings import OpenAIEmbeddings
from gmail_client import GmailClient, EmailSearchOptions
from vector_store_loader import VectorStoreFactory

# Initialize components
gmail_client = GmailClient("credentials.json")
embeddings_model = OpenAIEmbeddings()
vector_store = VectorStoreFactory.create(
    embeddings_model,
    config={
        "type": "chroma",
        "collection_name": "email_store"
    }
)

# Set up pipeline
pipeline = SimpleEmailPipeline(embeddings_model, vector_store)

# Search parameters
search_options = EmailSearchOptions(
    query="after:2024/01/01",
    max_results=100
)

# Run ingestion
num_processed = pipeline.ingest_emails(gmail_client, search_options)
print(f"Processed {num_processed} emails")
"""
