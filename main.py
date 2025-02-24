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

async def process_document_query(request: QueryRequest) -> Dict[str, Any]:
    """Process queries for document retrieval"""
    if not pool or not vector_store:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    try:
        # Configure search
        search_kwargs = {"filter": {}, "k": 5}
        if request.document_ids:
            search_kwargs["filter"]["document_id"] = {"$in": request.document_ids}
        if request.metadata_filter:
            search_kwargs["filter"].update(request.metadata_filter)

        # Set up retrieval chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

        # Execute query
        chain_response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: qa({"question": request.query, "chat_history": []})
        )

        # Get response and sources
        response_content = chain_response.get("answer", "No response generated")
        source_documents = chain_response.get("source_documents", [])
        
        # Format response with sources
        if source_documents:
            sources_text = format_sources(source_documents)
            formatted_response = f"{response_content}\n\nSources:\n{sources_text}"
        else:
            formatted_response = response_content

        # Store in database
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (role, content, conversation_id, metadata) VALUES ($1, $2, $3, $4)",
                'user', request.query, request.conversation_id,
                json.dumps({"context": request.context.dict() if request.context else {}})
            )
            await conn.execute(
                "INSERT INTO messages (role, content, conversation_id, metadata) VALUES ($1, $2, $3, $4)",
                'assistant', formatted_response, request.conversation_id,
                json.dumps({})
            )

        return {
            "status": "success",
            "answer": formatted_response,
            "sources": [
                {
                    "id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title"),
                    "page": doc.metadata.get("page")
                } for doc in source_documents
            ]
        }

    except Exception as e:
        logger.error(f"Error in document query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Main query endpoint that routes to appropriate handler"""
    try:
        # Convert Pydantic model to dictionary for email adapter functions
        request_dict = {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "context": request.context.dict() if request.context else {},
            "email_filters": request.email_filters.dict() if request.email_filters else None,
            "info_sources": request.info_sources
        }
        
        if email_qa_system and (request.email_filters or is_email_query(request_dict)):
            logger.info(f"Routing to email system: {request.query}")
            email_result = await process_email_query(
                query=request.query,
                conversation_id=request.conversation_id,
                context=request.context.dict() if request.context else {},
                email_filters=request.email_filters.dict() if request.email_filters else None
            )
            return email_result
        else:
            logger.info(f"Routing to document system: {request.query}")
            return await process_document_query(request)
    except Exception as e:
        logger.error(f"Error in query routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
    """Download file directly from Supabase with path correction"""
    logger.info(f"Original file URL: {file_url}")
    
    try:
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL").rstrip("/")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        # Attempt to extract the filename and path
        filename = file_url.split("/")[-1]
        
        # For the case where files are in a documents/documents/ nested structure
        if "documents/documents/" in file_url:
            logger.info("Detected nested document path structure")
            file_path = f"documents/{filename}"
        else:
            file_path = filename
            
        logger.info(f"Using file_path: {file_path}")
        
        # Create full download URL that uses the Supabase API directly
        download_url = f"{supabase_url}/storage/v1/object/public/documents/{file_path}"
        
        logger.info(f"Trying direct download URL: {download_url}")
        
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(download_url, headers=headers) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to download from {download_url}. Status: {response.status}, Error: {error_text}")
                    
                    # Try alternative path structures
                    alternative_paths = [
                        f"{supabase_url}/storage/v1/object/public/documents/{filename}",
                        f"{supabase_url}/storage/v1/object/authenticated/documents/{filename}",
                        file_url  # Original URL as fallback
                    ]
                    
                    for alt_url in alternative_paths:
                        logger.info(f"Trying alternative URL: {alt_url}")
                        async with session.get(alt_url, headers=headers) as alt_response:
                            if alt_response.status == 200:
                                logger.info(f"Success with alternative URL: {alt_url}")
                                return await alt_response.read()
                            else:
                                alt_error = await alt_response.text()
                                logger.error(f"Failed with alternative URL {alt_url}. Status: {alt_response.status}, Error: {alt_error}")
                    
                    # If all attempts fail, raise exception with the original error
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to download file: {error_text}"
                    )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document download failed: {str(e)}")

# Add the document processing endpoint
@app.post("/process")
async def process_document(request: ProcessRequest):
    """Process a document and store its embeddings"""
    global vector_store  # Use the vector_store from main.py
    
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
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)

                # Update metadata for each chunk
                for text in texts:
                    text.metadata.update({
                        "document_id": request.document_id,
                        "filename": filename,
                        "title": request.metadata.get("title", filename),  
                        "upload_time": datetime.utcnow().isoformat(),
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
                    "chunks_processed": len(texts),
                    "metadata": {
                        "upload_time": datetime.utcnow().isoformat(),
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
