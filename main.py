from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import asyncio
import asyncpg
import logging
import os
import json
from datetime import datetime

# Import LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import local components
from client.gmail_client import GmailClient, EmailSearchOptions
from retrievers.email_retriever import EmailQASystem
from email_adapter import create_email_router, get_email_qa_system, is_email_query, process_email_query

import aiohttp
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bubble_backend import query_router, initialize_components


# Define models inline instead of importing
class Weather(BaseModel):
    conditions: Optional[str] = None
    temperature: Optional[float] = None

    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < -100 or v > 100):
            raise ValueError('Temperature must be between -100 and 100 degrees')
        return v

class QueryContext(BaseModel):
    location: Optional[str] = None
    weather: Optional[Weather] = None
    activeEquipment: Optional[List[str]] = None
    timeOfDay: Optional[str] = None
    deviceType: Optional[str] = None
    noiseLevel: Optional[float] = None
    previousQueries: Optional[List[str]] = None
    timeframe: Optional[str] = None

class EmailFilter(BaseModel):
    after_date: Optional[str] = None
    before_date: Optional[str] = None
    from_email: Optional[str] = None
    to_email: Optional[str] = None
    subject_contains: Optional[str] = None
    has_attachment: Optional[bool] = None
    label: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    context: Optional[QueryContext] = None
    document_ids: Optional[List[str]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    email_filters: Optional[EmailFilter] = None
    info_sources: Optional[List[str]] = None

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query too long (max 1000 characters)')
        return v.strip()

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables and validate
required_env_vars = [
    'POSTGRES_CONNECTION_STRING', 
    'OPENAI_API_KEY', 
    'SUPABASE_URL', 
    'SUPABASE_SERVICE_KEY'
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Global state
FLOWS = {}
GMAIL_CLIENTS = {}
pool: Optional[asyncpg.Pool] = None
vector_store: Optional[PGVector] = None
email_qa_system: Optional[EmailQASystem] = None

# Simple function to format sources
def format_sources(source_documents: List) -> str:
    """Format source documents into a readable string"""
    if not source_documents:
        return ""
        
    sources = []
    for doc in source_documents:
        source_text = f"• {doc.metadata.get('title', 'Unknown Document')}"
        if doc.metadata.get('page'):
            source_text += f" (Page {doc.metadata.get('page')})"
        sources.append(source_text)
        
    return "\n".join(sources)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global pool, vector_store, email_qa_system
    try:
        # Initialize database pool
        pool = await asyncpg.create_pool(
            os.getenv('POSTGRES_CONNECTION_STRING'),
            min_size=5,
            max_size=20,
            statement_cache_size=0
        )
        logger.info("Database pool created successfully")
        
        # Initialize vector stores
        vector_store = PGVector(
            collection_name="document_embeddings",
            connection_string=os.getenv('POSTGRES_CONNECTION_STRING'),
            embedding_function=OpenAIEmbeddings()
        )
        logger.info("Document vector store initialized successfully")
        
        # Initialize email QA system
        email_qa_system = await get_email_qa_system()
        
        # Initialize bubble_backend components
        bubble_components = await initialize_components()
        logger.info("Bubble backend components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        if pool:
            await pool.close()
            logger.info("Database pool closed")

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include email router
app.include_router(create_email_router())

app.include_router(query_router)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection with a timeout
        db_healthy = False
        if pool:
            try:
                async with asyncio.timeout(1.0):  # 1 second timeout
                    async with pool.acquire() as conn:
                        await conn.execute('SELECT 1')
                        db_healthy = True
            except Exception as e:
                logger.warning(f"Database health check issue: {e}")
        
        # Simple check if vector store exists (don't do operations)
        vs_exists = vector_store is not None
            
        # Simple check if email system has been initialized (don't call methods)
        email_status = "initialized" if email_qa_system is not None else "not available"
        
        return {
            "status": "healthy",
            "database": "connected" if db_healthy else "disconnected",
            "vector_store": "available" if vs_exists else "unavailable",
            "email_system": email_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Still return a 200 status so the healthcheck passes
        # Just indicate components that are unhealthy
        return {
            "status": "partial",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/auth/gmail")
async def gmail_auth():
    """Start Gmail OAuth flow"""
    try:
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        if not creds_json:
            raise HTTPException(status_code=500, detail="Google credentials not configured")
        
        credentials = json.loads(creds_json)
        flow, auth_url = GmailClient.get_auth_url(credentials)
        FLOWS['gmail'] = flow
        
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
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        credentials = json.loads(creds_json)
        
        gmail_client = GmailClient(credentials=credentials)
        token = gmail_client.authorize_with_code(flow, code)
        GMAIL_CLIENTS['user'] = gmail_client
        
        return {"message": "Authentication successful", "token": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def extract_source_attributions(response_content, source_documents):
    """
    Extract source attributions with confidence scores and relevant excerpts
    from the document sources that contributed to the response.
    """
    attributions = []
    
    for doc in source_documents:
        # Extract document content
        doc_content = doc.page_content if hasattr(doc, 'page_content') else ""
        
        # Skip empty documents
        if not doc_content:
            continue
            
        # Calculate relevance based on content overlap
        response_lower = response_content.lower()
        doc_lower = doc_content.lower()
        
        # Simple relevance: Calculate word overlap
        response_words = set(w for w in response_lower.split() if len(w) > 3)
        doc_words = set(w for w in doc_lower.split() if len(w) > 3)
        
        if not response_words:
            confidence = 0.5  # Default if no significant words in response
        else:
            overlap_count = len(response_words.intersection(doc_words))
            # Calculate confidence (basic implementation)
            confidence = min(0.95, max(0.3, overlap_count / (len(response_words) + 1)))
        
        # Find most relevant excerpt (simple implementation)
        # Look for sentences or paragraphs that contain the most overlapping words
        excerpt = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
        
        # Try to find a better excerpt by splitting into paragraphs
        paragraphs = [p.strip() for p in doc_content.split('\n\n') if p.strip()]
        if paragraphs:
            # Score each paragraph by word overlap
            para_scores = []
            for i, para in enumerate(paragraphs):
                para_words = set(w for w in para.lower().split() if len(w) > 3)
                overlap = len(para_words.intersection(response_words))
                score = overlap / (len(para_words) + 1) if para_words else 0
                para_scores.append((score, i, para))
            
            # Use the highest scoring paragraph as excerpt
            if para_scores:
                para_scores.sort(reverse=True)
                best_para = para_scores[0][2]
                if len(best_para) > 220:
                    excerpt = best_para[:220] + "..."
                else:
                    excerpt = best_para
        
        attributions.append({
            "source_id": doc.metadata.get("document_id"),
            "confidence": round(confidence, 2),
            "content": excerpt
        })
    
    # Sort by confidence
    return sorted(attributions, key=lambda x: x["confidence"], reverse=True)


# Add the ProcessRequest model if it's not already in main.py
class ProcessRequest(BaseModel):
    document_id: str
    file_url: str
    metadata: Optional[Dict[str, Any]] = None

    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()

# Add the download_file helper function
async def download_file(file_url: str) -> bytes:
    """Download file from URL with support for Supabase signed URLs and fallbacks"""
    logger.info(f"Attempting to download file from URL: {file_url}")
    
    # First, try the URL as-is (for signed URLs)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    logger.info(f"Successfully downloaded file directly from URL")
                    return await response.read()
                else:
                    error_text = await response.text()
                    logger.warning(f"Direct download failed. Status: {response.status}, Error: {error_text}")
                    
                    # Special handling for expired signed URLs
                    if "token" in file_url and (response.status == 401 or response.status == 403):
                        logger.warning("The signed URL appears to have expired")
    except Exception as e:
        logger.warning(f"Exception in direct download: {str(e)}")
    
    # If direct download fails, extract filename for fallback approaches
    # Handle URLs with query parameters (like signed URLs)
    filename = file_url.split("/")[-1].split("?")[0]  # Remove query parameters
    logger.info(f"Attempting fallback approaches for file: {filename}")
    
    # Try fallback approaches with Supabase
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    
    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500,
            detail="Supabase configuration missing for fallback approaches."
        )
    
    # Try multiple bucket configurations as fallbacks
    buckets = ["documents", "storage", ""]  # Empty string for no bucket prefix
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    }
    
    # First try the common path pattern we've seen in your URL
    if "/documents/documents/" in file_url:
        bucket_path = "documents/documents"
        for access_type in ["public", "authenticated"]:
            download_url = f"{supabase_url}/storage/v1/object/{access_type}/{bucket_path}/{filename}"
            
            logger.info(f"Trying specific bucket path URL: {download_url}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(download_url, headers=headers) as response:
                        if response.status == 200:
                            logger.info(f"Successfully downloaded using specific bucket path: {download_url}")
                            return await response.read()
                        else:
                            error_text = await response.text()
                            logger.warning(f"Failed specific path download from {download_url}. Status: {response.status}")
            except Exception as e:
                logger.warning(f"Exception trying specific path URL {download_url}: {str(e)}")
    
    # Try other bucket combinations
    for bucket in buckets:
        bucket_prefix = f"{bucket}/" if bucket else ""
        for access_type in ["public", "authenticated"]:
            download_url = f"{supabase_url}/storage/v1/object/{access_type}/{bucket_prefix}{filename}"
            
            logger.info(f"Trying fallback Supabase URL: {download_url}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(download_url, headers=headers) as response:
                        if response.status == 200:
                            logger.info(f"Successfully downloaded using fallback URL: {download_url}")
                            return await response.read()
                        else:
                            error_text = await response.text()
                            logger.warning(f"Failed fallback download from {download_url}. Status: {response.status}")
            except Exception as e:
                logger.warning(f"Exception trying fallback URL {download_url}: {str(e)}")
    
    # If we've come this far, try one more approach: reconstructing the signed URL path
    if "/sign/" in file_url:
        # Extract the bucket and path from the signed URL
        parts = file_url.split("/storage/v1/object/sign/")[1].split("?")[0]
        for access_type in ["public", "authenticated"]:
            download_url = f"{supabase_url}/storage/v1/object/{access_type}/{parts}"
            
            logger.info(f"Trying reconstructed URL: {download_url}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(download_url, headers=headers) as response:
                        if response.status == 200:
                            logger.info(f"Successfully downloaded using reconstructed URL: {download_url}")
                            return await response.read()
                        else:
                            error_text = await response.text()
                            logger.warning(f"Failed reconstructed download from {download_url}. Status: {response.status}")
            except Exception as e:
                logger.warning(f"Exception trying reconstructed URL {download_url}: {str(e)}")
    
    # If all attempts fail, raise an error
    logger.error("All download attempts failed")
    raise HTTPException(
        status_code=400,
        detail="Failed to download file after multiple attempts. The URL may be invalid or expired."
    )

# Add the document processing endpoint
@app.post("/process")
async def process_document(request: ProcessRequest):
    """Process a document and store its embeddings with enhanced metadata"""
    global vector_store
    
    if not vector_store:
        logger.error("Vector store not initialized during document processing attempt")
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. Please wait for system startup to complete and try again."
        )

    try:
        # Extract filename from URL
        filename = request.file_url.split("/")[-1]
        
        # Download and process file
        logger.info(f"Starting processing of document: {filename}")
        file_content = await download_file(request.file_url)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

            try:
                # Load and split document
                loader = PyMuPDFLoader(temp_path)
                documents = loader.load()
                
                # Enhanced document title logic
                title = request.metadata.get("title", "")
                if not title.strip():
                    # Extract title from filename by removing the extension and replacing underscores/hyphens
                    title = filename.split('.')[0].replace('_', ' ').replace('-', ' ').title()
                    # If title is still empty or just numbers, provide a fallback title
                    if not title.strip() or title.strip().isdigit():
                        title = f"Document {request.document_id}"
                
                # Get document creation date if available
                creation_date = None
                try:
                    if hasattr(documents[0], "metadata") and "creationDate" in documents[0].metadata:
                        creation_date = documents[0].metadata["creationDate"]
                except (IndexError, AttributeError, KeyError):
                    pass  # No creation date available
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)

                # Calculate total pages
                total_pages = max([doc.metadata.get("page", 0) for doc in documents], default=0) + 1
                
                # Log document info
                logger.info(f"Document '{title}' has {total_pages} pages, split into {len(texts)} chunks")

                # Update metadata for each chunk with enhanced information
                for i, text in enumerate(texts):
                    # Determine page number, defaulting to chunk index if not available
                    page_number = text.metadata.get("page", i)
                    
                    # Enhanced metadata with more useful fields
                    text.metadata.update({
                        "document_id": request.document_id,
                        "filename": filename,
                        "title": title,
                        "page": page_number,
                        "total_pages": total_pages,
                        "chunk_index": i,
                        "total_chunks": len(texts),
                        "document_type": request.metadata.get("document_type", "Unknown"),
                        "description": request.metadata.get("description", ""),
                        "upload_time": datetime.utcnow().isoformat(),
                        "creation_date": creation_date,
                        "content_length": len(text.page_content) if hasattr(text, "page_content") else 0,
                        **(request.metadata or {})
                    })

                # Add to vector store with explicit error handling
                try:
                    vector_store.add_documents(texts)
                except Exception as e:
                    logger.error(f"Failed to add documents to vector store: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to store document vectors: {str(e)}"
                    )

                # Log success
                logger.info(f"Successfully processed document {request.document_id} with {len(texts)} chunks")

                return {
                    "status": "success",
                    "document_id": request.document_id,
                    "filename": filename,
                    "title": title,
                    "chunks_processed": len(texts),
                    "total_pages": total_pages,
                    "metadata": {
                        "document_type": request.metadata.get("document_type", "Unknown"),
                        "description": request.metadata.get("description", ""),
                        "upload_time": datetime.utcnow().isoformat(),
                        "creation_date": creation_date,
                        "chunk_size": 1000,
                        "overlap": 200
                    }
                }

            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file {temp_path}")

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

