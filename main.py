from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from loaders.simple_pipeline import SimpleEmailPipeline
import os

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 100
    include_attachments: Optional[bool] = False

@app.post("/process-emails")
async def process_emails(search_request: SearchRequest):
    try:
        # Initialize components
        gmail_client = GmailClient("credentials.json")
        embeddings_model = HuggingFaceEmbeddings()
        
        vector_store = VectorStoreFactory.create(
            embeddings_model,
            config={
                "supabase_url": os.getenv("SUPABASE_URL"),
                "supabase_key": os.getenv("SUPABASE_SERVICE_KEY"),  # Use service key for full access
                "table_name": "email_embeddings"
            }
        )

        # Set up pipeline
        pipeline = SimpleEmailPipeline(embeddings_model, vector_store)

        # Search parameters
        search_options = EmailSearchOptions(
            query=search_request.query,
            max_results=search_request.max_results,
            include_attachments=search_request.include_attachments
        )

        # Run ingestion
        num_processed = pipeline.ingest_emails(gmail_client, search_options)
        
        return {"status": "success", "emails_processed": num_processed}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
