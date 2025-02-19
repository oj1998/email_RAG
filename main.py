from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import json
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from loaders.simple_pipeline import SimpleEmailPipeline
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

app = FastAPI()

# Store flow objects temporarily (in production, use a proper session store)
FLOWS = {}
# Store Gmail clients (in production, use a proper user session management)
GMAIL_CLIENTS = {}
# Store email QA systems
EMAIL_QA_SYSTEMS = {}

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 100
    include_attachments: Optional[bool] = False

class QueryRequest(BaseModel):
    question: str
    query_str: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    k: Optional[int] = 5

class EmailFilterRequest(BaseModel):
    after_date: Optional[str] = None
    before_date: Optional[str] = None
    from_email: Optional[str] = None
    to_email: Optional[str] = None
    subject_contains: Optional[str] = None
    has_attachment: Optional[bool] = None
    label: Optional[str] = None

async def get_vector_store():
    """Get or create vector store"""
    embeddings_model = HuggingFaceEmbeddings()
    vector_store = VectorStoreFactory.create(
        embeddings_model,
        config={
            "type": "supabase",
            "supabase_url": os.getenv("SUPABASE_URL"),
            "supabase_key": os.getenv("SUPABASE_SERVICE_KEY"),
            "collection_name": "email_embeddings"
        }
    )
    return vector_store, embeddings_model

async def get_qa_system():
    """Get or create email QA system"""
    if 'email_qa' not in EMAIL_QA_SYSTEMS:
        vector_store, embeddings_model = await get_vector_store()
        
        # Initialize language model
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create QA system
        EMAIL_QA_SYSTEMS['email_qa'] = EmailQASystem(
            vector_store=vector_store,
            embeddings_model=embeddings_model,
            llm=llm,
            k=5,
            use_reranker=True
        )
    
    return EMAIL_QA_SYSTEMS['email_qa']

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
        vector_store, embeddings_model = await get_vector_store()

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

@app.post("/query-emails")
async def query_emails(query_request: QueryRequest):
    """Query emails and get an answer"""
    try:
        qa_system = await get_qa_system()
        
        answer = qa_system.query(
            question=query_request.question,
            query_str=query_request.query_str,
            filters=query_request.filters,
            k=query_request.k
        )
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-relevant-emails")
async def get_relevant_emails(query_request: QueryRequest):
    """Get relevant emails without generating an answer"""
    try:
        qa_system = await get_qa_system()
        
        relevant_emails = qa_system.get_relevant_emails(
            query=query_request.query_str or query_request.question,
            filters=query_request.filters,
            k=query_request.k
        )
        
        # Convert documents to dict format
        results = []
        for doc in relevant_emails:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {"emails": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
