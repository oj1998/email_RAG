from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import uvicorn
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import logging
import json
from datetime import datetime
import asyncpg
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager

# Import email-related components
from langchain_community.embeddings import HuggingFaceEmbeddings
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize environment variables with validation
required_env_vars = [
    'POSTGRES_CONNECTION_STRING', 
    'OPENAI_API_KEY', 
    'SUPABASE_URL', 
    'SUPABASE_SERVICE_KEY'
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

CONNECTION_STRING = os.getenv('POSTGRES_CONNECTION_STRING')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# Global variables for state management
pool: Optional[asyncpg.Pool] = None
vector_store: Optional[PGVector] = None
email_qa_system: Optional[EmailQASystem] = None

# Pydantic models
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
    context: QueryContext
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

# Startup and shutdown events manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, vector_store, email_qa_system
    try:
        # Initialize database pool with statement cache disabled
        pool = await asyncpg.create_pool(
            CONNECTION_STRING,
            min_size=5,
            max_size=20,
            statement_cache_size=0  # Only use this parameter
        )
        logger.info("Database pool created successfully")
        
        # Verify/create database tables
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        logger.info("Database tables verified/created successfully")
        
        # Initialize vector store for documents
        vector_store = PGVector(
            collection_name="document_embeddings",
            connection_string=CONNECTION_STRING,
            embedding_function=OpenAIEmbeddings()
        )
        logger.info("Document vector store initialized successfully")
        
        # Initialize email QA system
        try:
            # Initialize embedding model
            embeddings_model = HuggingFaceEmbeddings()
            
            # Initialize language model
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",
                temperature=0.2
            )
            
            # Initialize vector store for emails
            email_vector_store = VectorStoreFactory.create(
                embeddings_model,
                config={
                    "type": "supabase",
                    "supabase_url": SUPABASE_URL,
                    "supabase_key": SUPABASE_KEY,
                    "collection_name": "email_embeddings"
                }
            )
            
            # Create QA system
            email_qa_system = EmailQASystem(
                vector_store=email_vector_store,
                embeddings_model=embeddings_model,
                llm=llm,
                k=5,
                use_reranker=True
            )
            logger.info("Email QA system initialized successfully")
        except Exception as e:
            logger.warning(f"Email QA system initialization failed: {e}")
            email_qa_system = None
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Cleanup
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        if not pool:
            raise HTTPException(status_code=500, detail="Database pool not initialized")
            
        async with pool.acquire() as conn:
            await conn.execute('SELECT 1')
            
        # Check vector store
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
            
        # Check email system status
        email_status = "initialized" if email_qa_system else "not available"
            
        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": "initialized",
            "email_system": email_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

class NLPTransformer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Enhanced prompts with source handling
        self.analysis_prompt = PromptTemplate(
            template="""Analyze this content and suggest appropriate structure and formatting.

            Query: {query}
            Context: {context}
            Raw Response: {response}
            Sources: {sources}

            Consider:
            1. The natural logical flow of information
            2. The urgency and importance of different parts
            3. The query's intent and context
            4. How to make the information most actionable
            5. Which sources support which parts of the response

            Response must be valid JSON matching this structure:
            {
                "sections": [
                    {
                        "title": "section title",
                        "content": "section content",
                        "priority": priority_number,
                        "formatting": {formatting_instructions},
                        "sources": [source_references]
                    }
                ],
                "metadata": {
                    additional_metadata
                }
            }
            """,
            input_variables=["query", "context", "response", "sources"]
        )
        
        self.formatting_prompt = PromptTemplate(
            template="""Format this content according to the provided specifications.

            Content: {content}
            Section Type: {section_type}
            Formatting Instructions: {formatting}

            Format the content to be clear and actionable while maintaining its meaning.
            Add appropriate formatting elements (e.g., numbering, bullets, warnings).
            
            Formatted Content:""",
            input_variables=["content", "section_type", "formatting"]
        )

    async def transform_content(self, query: str, raw_response: str, context: Dict, source_documents: List) -> str:
        """Transform content using NLP-based analysis with source tracking"""
        try:
            # Format source information
            sources = []
            for doc in source_documents:
                source_info = {
                    "id": doc.metadata.get("document_id", "unknown"),
                    "title": doc.metadata.get("title", "Unknown Document"),
                    "page": doc.metadata.get("page", None)
                }
                sources.append(source_info)

            # Get content analysis
            analysis_prompt = self.analysis_prompt.format(
                query=query,
                context=json.dumps(context),
                response=raw_response,
                sources=json.dumps(sources)
            )
            
            analysis_response = self.llm.invoke(analysis_prompt)
            structured_format = json.loads(analysis_response.content)
            
            # Format sections with sources
            formatted_sections = []
            for section in structured_format["sections"]:
                formatting_result = await self._format_section(
                    section["content"],
                    section["title"],
                    section["formatting"]
                )
                formatted_sections.append({
                    "title": section["title"],
                    "content": formatting_result,
                    "priority": section["priority"],
                    "sources": section.get("sources", [])
                })
            
            # Combine sections and add source references
            final_content = self._combine_sections_with_sources(formatted_sections, sources)
            
            return final_content
            
        except Exception as e:
            logger.error(f"Error in content transformation: {e}")
            return raw_response

    async def _format_section(self, content: str, section_type: str, formatting: Dict) -> str:
        """Format an individual section"""
        try:
            formatting_prompt = self.formatting_prompt.format(
                content=content,
                section_type=section_type,
                formatting=json.dumps(formatting)
            )
            
            formatted_content = self.llm.invoke(formatting_prompt)
            return formatted_content.content.strip()
            
        except Exception as e:
            logger.error(f"Error formatting section {section_type}: {e}")
            return content

    def _combine_sections_with_sources(self, sections: List[Dict], all_sources: List[Dict]) -> str:
        """Combine formatted sections and add source references"""
        try:
            combined = []
            used_sources = set()
            
            # Add content sections
            for section in sections:
                title = section.get('title', 'Section')
                content = section.get('content', '')
                section_sources = section.get('sources', [])
                
                if content:
                    combined.extend([f"\n{title}", content])
                    used_sources.update(section_sources)
            
            # Add sources section if there are any sources
            if used_sources:
                combined.extend([
                    "\nSources:",
                    self._format_sources(used_sources, all_sources)
                ])
            
            return "\n".join(combined).strip()
            
        except Exception as e:
            logger.error(f"Error combining sections with sources: {e}")
            return "\n".join(s.get('content', '') for s in sections)

    def _format_sources(self, used_source_ids: set, all_sources: List[Dict]) -> str:
        """Format source references"""
        formatted_sources = []
        for source in all_sources:
            if source["id"] in used_sources:
                source_text = f"• {source['title']}"
                if source['page']:
                    source_text += f" (Page {source['page']})"
                formatted_sources.append(source_text)
        return "\n".join(formatted_sources)

# Email query handler
async def process_email_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a query specifically for email data
    
    Args:
        request: The query request
        
    Returns:
        Response with answer and metadata
    """
    try:
        if not email_qa_system:
            raise HTTPException(status_code=503, detail="Email system not available")
        
        # Convert email filters if provided
        filters = None
        if request.email_filters:
            filters = request.email_filters.dict(exclude_none=True)
        
        # Get answer from email QA system
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: email_qa_system.query(
                question=request.query, 
                filters=filters,
                k=5
            )
        )
        
        # Get source emails for reference
        relevant_emails = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: email_qa_system.get_relevant_emails(
                query=request.query, 
                filters=filters, 
                k=3
            )
        )
        
        # Format source information
        sources = []
        for email in relevant_emails:
            source_info = {
                "id": email.metadata.get("email_id", "unknown"),
                "title": email.metadata.get("subject", "Email"),
                "sender": email.metadata.get("sender", "Unknown"),
                "date": email.metadata.get("date", "")
            }
            sources.append(source_info)
        
        # Store in database
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (role, content, conversation_id, metadata)
                VALUES ($1, $2, $3, $4)
            """, 'user', request.query, request.conversation_id,
                json.dumps({"context": request.context.dict()}))

            await conn.execute("""
                INSERT INTO messages (role, content, conversation_id, metadata)
                VALUES ($1, $2, $3, $4)
            """, 'assistant', answer, request.conversation_id,
                json.dumps({"sources": sources}))
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error in email query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Email query processing failed: {str(e)}")

def is_email_query(request: QueryRequest) -> bool:
    """
    Determine if the query should be routed to email system
    based on info_sources or explicit flags
    """
    # If email_filters is explicitly provided, it's an email query
    if request.email_filters:
        return True
        
    # Check if emails are in the conversation's info_sources
    if request.info_sources:
        email_sources = ["emails", "email", "mail", "gmail"]
        if any(source.lower() in email_sources for source in request.info_sources):
            return True
            
    # Check query text for email-specific indicators
    query = request.query.lower()
    email_keywords = ["email", "mail", "gmail", "inbox", "message", "received", "sent"]
    if any(keyword in query for keyword in email_keywords):
        return True
        
    return False

# Document query handler (existing logic extracted to a separate function)
async def process_document_query(request: QueryRequest) -> Dict[str, Any]:
    if not pool or not vector_store:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    # Configure search parameters
    search_kwargs = {"filter": {}, "k": 5}
    if request.document_ids:
        search_kwargs["filter"]["document_id"] = {"$in": request.document_ids}
    if request.metadata_filter:
        search_kwargs["filter"].update(request.metadata_filter)

    # Format context string
    weather_str = "Unknown"
    if request.context.weather:
        conditions = request.context.weather.conditions or "Unknown"
        temp = request.context.weather.temperature or "Unknown"
        weather_str = f"{conditions}, {temp}°"

    context_str = f"""Current Context:
- Location: {request.context.location or 'Unknown'}
- Weather: {weather_str}
- Active Equipment: {', '.join(request.context.activeEquipment) if request.context.activeEquipment else 'None'}
- Time: {request.context.timeOfDay or 'Unknown'}
- Device: {request.context.deviceType or 'Unknown'}"""

    # Initialize memory and retriever
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # Create the chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        ),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    # Execute the chain with simpler input structure
    chain_response = await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: qa({
            "question": request.query,
            "chat_history": []
        })
    )

    response_content = chain_response.get("answer", "No response generated")
    source_documents = chain_response.get("source_documents", [])

    # Transform the response
    transformer = NLPTransformer()
    formatted_response = await transformer.transform_content(
        query=request.query,
        raw_response=response_content,
        context=request.context.dict(),
        source_documents=source_documents
    )

    # Store in database
    async with pool.acquire() as conn:
        # Use simple query protocol instead of prepared statements
        await conn.execute("""
            INSERT INTO messages (role, content, conversation_id, metadata)
            VALUES ($1, $2, $3, $4)
        """, 'user', request.query, request.conversation_id,
            json.dumps({"context": request.context.dict()}))

        await conn.execute("""
            INSERT INTO messages (role, content, conversation_id, metadata)
            VALUES ($1, $2, $3, $4)
        """, 'assistant', formatted_response, request.conversation_id,
            json.dumps({}))

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

# Main query endpoint that routes to appropriate handler
@app.post("/query")
async def query_documents(request: QueryRequest):
    """Main query endpoint that determines whether to use document or email retrieval"""
    try:
        # Route based on query type
        if is_email_query(request) and email_qa_system:
            logger.info(f"Routing query to email system: {request.query}")
            return await process_email_query(request)
        else:
            logger.info(f"Routing query to document system: {request.query}")
            return await process_document_query(request)
            
    except Exception as e:
        logger.error(f"Error in query routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("bubble_backend:app", host="0.0.0.0", port=port, reload=True)
