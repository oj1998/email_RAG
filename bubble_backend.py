# Part 1: Imports and Initial Setup
from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Tuple
import uvicorn
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import os
import logging
import json
from datetime import datetime, timedelta
import asyncpg
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager
from email_adapter import get_email_qa_system, process_email_query as adapter_process_email_query, is_email_query as adapter_is_email_query

# Import email-related components
from langchain_community.embeddings import HuggingFaceEmbeddings
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

# Import our custom components
from conversation_handler import ConversationHandler, ConversationContext
from construction_classifier import ConstructionClassifier, QuestionType
from weighted_memory import WeightedConversationMemory
from format_mapper import FormatMapper, CategoryFormat, SafetyLevel
from query_intent import QueryIntentAnalyzer, QueryIntent, IntentAnalysis

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
conversation_handler: Optional[ConversationHandler] = None
classifier: Optional[ConstructionClassifier] = None

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

class ResponseMetadata(BaseModel):
    category: str
    confidence: float
    source_count: int
    processing_time: float
    conversation_context_used: bool
    intent_analysis: Optional[Dict] = None

# Part 2: FastAPI Setup and NLP Transformer

# Startup and shutdown events manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, vector_store, conversation_handler, classifier
    try:
        # Initialize database pool with statement cache disabled
        pool = await asyncpg.create_pool(
            CONNECTION_STRING,
            min_size=5,
            max_size=20,
            statement_cache_size=0
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
        
        # Initialize conversation handler
        conversation_handler = ConversationHandler(pool)
        logger.info("Conversation handler initialized successfully")
        
        # Initialize classifier
        classifier = ConstructionClassifier()
        logger.info("Construction classifier initialized successfully")
        
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
        
        # Check components
        vs_exists = vector_store is not None
        ch_exists = conversation_handler is not None
        classifier_exists = classifier is not None
            
        # Check email system status without making calls
        try:
            email_status = "initialized" if await get_email_qa_system() is not None else "not available"
        except Exception:
            email_status = "not available"
            
        return {
            "status": "healthy",
            "database": "connected" if db_healthy else "disconnected",
            "vector_store": "available" if vs_exists else "unavailable",
            "conversation_handler": "available" if ch_exists else "unavailable",
            "classifier": "available" if classifier_exists else "unavailable",
            "email_system": email_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "partial",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

class NLPTransformer:
    def __init__(self):
        self.format_mapper = FormatMapper()
        self.intent_analyzer = QueryIntentAnalyzer()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2
        )

    async def transform_content(
        self, 
        query: str,
        raw_response: str,
        context: Dict,
        source_documents: List,
        classification: Dict
    ) -> Tuple[str, IntentAnalysis]:
        # Analyze query intent
        intent_analysis = self.intent_analyzer.analyze(query, context)
        
        # Get base category format
        category_format = self.format_mapper.get_format_for_category(
            classification['category']
        )

        # If it's a casual/discussion intent, use simplified format
        if intent_analysis.primary_intent in [QueryIntent.DISCUSSION, QueryIntent.CLARIFICATION]:
            formatted_content = await self._format_conversational(
                raw_response,
                classification['category'],
                intent_analysis
            )
            return formatted_content, intent_analysis

        # For emergency intent, always use structured safety format
        if intent_analysis.primary_intent == QueryIntent.EMERGENCY:
            formatted_content = await self._format_emergency(
                raw_response,
                classification,
                intent_analysis
            )
            return formatted_content, intent_analysis

        # For other cases, use standard structured format
        formatted_content = await self._format_structured(
            raw_response,
            category_format,
            classification,
            intent_analysis
        )
        return formatted_content, intent_analysis

    async def _format_conversational(
        self,
        content: str,
        category: str,
        intent_analysis: IntentAnalysis
    ) -> str:
        """Format content in a conversational style."""
        prompt = PromptTemplate(
            template="""Format this response conversationally while preserving key information.
            Original content: {content}
            Category: {category}
            Intent: {intent}
            
            If this is safety-related, include important warnings naturally in the conversation.
            Make it sound informal but informative.
            
            Response:"""
        )
        
        response = await self.llm.agenerate(prompt.format(
            content=content,
            category=category,
            intent=intent_analysis.primary_intent.value
        ))
        
        return response.generations[0].text

    async def _format_emergency(
        self,
        content: str,
        classification: Dict,
        intent_analysis: IntentAnalysis
    ) -> str:
        """Format emergency content with highest priority safety formatting."""
        # Always use safety format with emergency priority
        safety_format = self.format_mapper.get_safety_format(
            safety_level=SafetyLevel.CRITICAL
        )
        
        return await self._format_structured(
            content,
            safety_format,
            classification,
            intent_analysis
        )

    async def _format_structured(
        self,
        content: str,
        format_spec: CategoryFormat,
        classification: Dict,
        intent_analysis: IntentAnalysis
    ) -> str:
        """Apply full structured formatting based on category and intent."""
        # Validate content matches required sections
        validation_errors = self.format_mapper.validate_content(
            content,
            classification['category']
        )
        
        if validation_errors:
            # Handle missing required sections
            content = await self._fix_missing_sections(
                content,
                validation_errors,
                format_spec
            )

        # Apply formatting rules
        formatted_content = self.format_mapper.apply_formatting(
            content,
            classification['category'],
            'main'  # section name
        )

        return formatted_content

    async def _fix_missing_sections(
        self,
        content: str,
        validation_errors: List[str],
        format_spec: CategoryFormat
    ) -> str:
        """Fix content to include missing required sections."""
        prompt = PromptTemplate(
            template="""Restructure this content to include missing required sections:
            Content: {content}
            Missing sections: {missing_sections}
            
            Ensure all required sections are present while preserving existing information.
            
            Restructured content:"""
        )
        
        response = await self.llm.agenerate(prompt.format(
            content=content,
            missing_sections=", ".join(validation_errors)
        ))
        
        return response.generations[0].text

# Part 3: Query Processing and Main Endpoint

async def process_email_query(
    request: QueryRequest,
    conversation_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Process a query specifically for email data with conversation context"""
    try:
        # Enhance context with conversation history
        enhanced_context = {
            **request.context.dict(),
            "conversation_history": conversation_context.get("chat_history", []) if conversation_context else []
        }

        # Get answer using email adapter
        response = await adapter_process_email_query(
            query=request.query,
            conversation_id=request.conversation_id,
            context=enhanced_context,
            email_filters=request.email_filters.dict() if request.email_filters else None
        )
        
        # Store in database with enhanced metadata
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (role, content, conversation_id, metadata)
                VALUES ($1, $2, $3, $4)
            """, 'user', request.query, request.conversation_id,
                json.dumps({
                    "context": enhanced_context,
                    "query_type": "email"
                }))

            await conn.execute("""
                INSERT INTO messages (role, content, conversation_id, metadata)
                VALUES ($1, $2, $3, $4)
            """, 'assistant', response["answer"], request.conversation_id,
                json.dumps({
                    "sources": response.get("sources", []),
                    "query_type": "email",
                    "conversation_used": bool(conversation_context)
                }))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in email query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_query(
    request: QueryRequest,
    conversation_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Enhanced document query processing with conversation context and classification"""
    start_time = datetime.utcnow()
    
    if not pool or not vector_store:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    try:
        # Get question classification
        classification = await classifier.classify_question(
            question=request.query,
            conversation_context=conversation_context,
            current_context=request.context.dict()
        )
        
        # Configure search based on classification
        search_kwargs = {"filter": {}, "k": 5}
        if request.document_ids:
            search_kwargs["filter"]["document_id"] = {"$in": request.document_ids}
        if request.metadata_filter:
            search_kwargs["filter"].update(request.metadata_filter)

        # Adjust search based on classification
        if classification.confidence > 0.7:
            search_kwargs["k"] = 8 if classification.category == "SAFETY" else 5

        # Initialize memory with conversation history
        memory = WeightedConversationMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            decay_rate=0.1,
            time_weight_factor=0.6,
            relevance_weight_factor=0.4
        )

        if conversation_context and "chat_history" in conversation_context:
            for message in conversation_context["chat_history"]:
                if isinstance(message, dict):
                    msg_type = HumanMessage if message.get("role") == "user" else AIMessage
                    memory.chat_memory.add_message(msg_type(content=message.get("content", "")))

        # Get retriever
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        # Initialize QA chain with appropriate model based on classification
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model_name="gpt-4" if classification.category == "SAFETY" else "gpt-3.5-turbo",
                temperature=0.7
            ),
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        # Execute chain with classification context
        chain_response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: qa({
                "question": request.query,
                "chat_history": memory.chat_memory.messages
            })
        )

        # Process response
        response_content = chain_response.get("answer", "No response generated")
        source_documents = chain_response.get("source_documents", [])

# Transform content based on classification
        transformer = NLPTransformer()
        
        formatted_response, intent_analysis = await transformer.transform_content(
            query=request.query,
            raw_response=response_content,
            context=request.context.dict(),
            source_documents=source_documents,
            classification=classification.dict()
        )

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Prepare response
        response = {
            "status": "success",
            "answer": formatted_response,
            "classification": classification.dict(),
            "sources": [
                {
                    "id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title"),
                    "page": doc.metadata.get("page")
                } for doc in source_documents
            ],
            "metadata": ResponseMetadata(
                category=classification.category,
                confidence=classification.confidence,
                source_count=len(source_documents),
                processing_time=processing_time,
                conversation_context_used=bool(conversation_context),
                intent_analysis=intent_analysis.dict()
            ).dict()
        }

        return response

    except Exception as e:
        logger.error(f"Error in document query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def is_email_query(request: QueryRequest) -> bool:
    """Determine if the query should be routed to email system"""
    return adapter_is_email_query({
        "query": request.query,
        "info_sources": request.info_sources,
        "context": request.context.dict(),
        "target_system": "email" if request.email_filters else None
    })

# Main query endpoint that routes to appropriate handler
@app.post("/query")
async def query_documents(request: QueryRequest):
    """Enhanced main query endpoint with conversation history and classification"""
    try:
        # Load conversation history
        conversation_context = await conversation_handler.load_conversation_history(
            conversation_id=request.conversation_id
        )
        
        # Process query with context
        if is_email_query(request):
            response = await process_email_query(
                request,
                conversation_context=conversation_context
            )
        else:
            response = await process_document_query(
                request,
                conversation_context=conversation_context
            )

        # Save the interaction
        await conversation_handler.save_conversation_turn(
            conversation_id=request.conversation_id,
            role='user',
            content=request.query,
            metadata={
                "context": request.context.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await conversation_handler.save_conversation_turn(
            conversation_id=request.conversation_id,
            role='assistant',
            content=response["answer"],
            metadata={
                "sources": response.get("sources", []),
                "classification": response.get("classification", {}),
                "metadata": response.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return response
            
    except Exception as e:
        logger.error(f"Error in query routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("bubble_backend:app", host="0.0.0.0", port=port, reload=True)
