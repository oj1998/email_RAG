from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
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
from email_adapter import create_email_router, get_email_qa_system, is_email_query
from models.request_models import QueryRequest, SearchRequest
from utils.nlp_transformer import NLPTransformer

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
        if not pool:
            raise HTTPException(status_code=500, detail="Database pool not initialized")
        async with pool.acquire() as conn:
            await conn.execute('SELECT 1')
            
        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": "initialized" if vector_store else "not initialized",
            "email_system": "initialized" if email_qa_system else "not available",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
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

        # Transform response
        transformer = NLPTransformer()
        formatted_response = await transformer.transform_content(
            query=request.query,
            raw_response=chain_response.get("answer", "No response generated"),
            context=request.context.dict(),
            source_documents=chain_response.get("source_documents", [])
        )

        # Store in database
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (role, content, conversation_id, metadata) VALUES ($1, $2, $3, $4)",
                'user', request.query, request.conversation_id,
                json.dumps({"context": request.context.dict()})
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
                } for doc in chain_response.get("source_documents", [])
            ]
        }

    except Exception as e:
        logger.error(f"Error in document query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Main query endpoint that routes to appropriate handler"""
    try:
        if email_qa_system and (request.email_filters or is_email_query(request)):
            logger.info(f"Routing to email system: {request.query}")
            return await process_email_query(request)
        else:
            logger.info(f"Routing to document system: {request.query}")
            return await process_document_query(request)
    except Exception as e:
        logger.error(f"Error in query routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
