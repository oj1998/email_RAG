from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict
import json
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from loaders.simple_pipeline import SimpleEmailPipeline

app = FastAPI()

# Store flow objects temporarily (in production, use a proper session store)
FLOWS = {}
# Store Gmail clients (in production, use a proper user session management)
GMAIL_CLIENTS = {}

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 100
    include_attachments: Optional[bool] = False

@app.get("/auth/gmail")
async def gmail_auth():
    """Start Gmail OAuth flow"""
    try:
        # Get credentials from environment
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        if not creds_json:
            raise HTTPException(status_code=500, detail="Google credentials not configured")
        
        credentials = json.loads(creds_json)
        
        # Generate auth URL
        flow, auth_url = GmailClient.get_auth_url(credentials)
        
        # Store flow object for callback
        FLOWS['gmail'] = flow
        
        # Redirect to Google's auth page
        return RedirectResponse(url=auth_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/oauth2callback")
async def oauth2callback(code: str, state: Optional[str] = None):
    """Handle OAuth callback"""
    try:
        if 'gmail' not in FLOWS:
            raise HTTPException(status_code=400, detail="No active OAuth flow")
            
        flow = FLOWS['gmail']
        
        # Initialize Gmail client with credentials
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        credentials = json.loads(creds_json)
        
        gmail_client = GmailClient(credentials=credentials)
        
        # Exchange code for tokens
        token = gmail_client.authorize_with_code(flow, code)
        
        # Store client for later use
        GMAIL_CLIENTS['user'] = gmail_client
        
        return {"message": "Authentication successful", "token": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-emails")
async def process_emails(search_request: SearchRequest):
    """Process and embed emails"""
    try:
        if 'user' not in GMAIL_CLIENTS:
            raise HTTPException(status_code=401, detail="Must authenticate with Gmail first")
            
        gmail_client = GMAIL_CLIENTS['user']
        
        # Use the correct model that produces 384-dimensional embeddings
        embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # This model produces 384-dimensional embeddings
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = VectorStoreFactory.create(
            embeddings_model,
            config={
                "type": "supabase",
                "supabase_url": os.getenv("SUPABASE_URL"),
                "supabase_key": os.getenv("SUPABASE_SERVICE_KEY"),
                "collection_name": "email_embeddings"
            }
        )

        pipeline = SimpleEmailPipeline(embeddings_model, vector_store)

        search_options = EmailSearchOptions(
            query=search_request.query,
            max_results=search_request.max_results,
            include_attachments=search_request.include_attachments
        )

        num_processed = pipeline.ingest_emails(gmail_client, search_options)
        
        return {"status": "success", "emails_processed": num_processed}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
