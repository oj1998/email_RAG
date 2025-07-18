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
from drilling_workflow import process_drilling_workflow

# Import email-related components
from langchain_community.embeddings import HuggingFaceEmbeddings
from client.gmail_client import GmailClient, EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

# Import our custom components
from conversation_handler import ConversationHandler, ConversationContext
from construction_classifier import ConstructionClassifier, QuestionType
from weighted_memory import WeightedConversationMemory
from format_mapper import FormatMapper, CategoryFormat, FormatStyle
from query_intent import SmartQueryIntentAnalyzer, QueryIntent, IntentAnalysis
from smart_response_generator import SmartResponseGenerator
from enhanced_similarity import EnhancedSmartResponseGenerator, EnhancedSourceAttribution

import aiohttp
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import APIRouter

from variance_analysis.document_variance_analyzer import DocumentVarianceAnalyzer
from variance_analysis.variance_formatter import VarianceFormatter, VarianceFormatStyle
from variance_analysis.variance_retriever import VarianceDocumentRetriever
from variance_integration import is_variance_analysis_query, process_variance_query

from knowledge_gaps import detect_knowledge_gap, create_knowledge_gap_router, KnowledgeGap

from rate_limiter import gpt4_limiter, gpt35_limiter, embeddings_limiter, rate_limited_call

import time
from hardcoded_responses import get_hardcoded_response
from langchain.chains import ConversationChain  # This is missing and causing the error

# Add this to bubble_backend.py near the top with your other logging setup
perf_logger = logging.getLogger("timing")
perf_logger.setLevel(logging.INFO)
perf_handler = logging.FileHandler("timing.log")
perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
perf_handler.setFormatter(perf_formatter)
perf_logger.addHandler(perf_handler)

# Add this function for inline timing
def log_time(section_name, start_time):
    elapsed_ms = (time.time() - start_time) * 1000
    perf_logger.info(f"{section_name}: {elapsed_ms:.2f}ms")

concurrent_query_semaphore = asyncio.Semaphore(3)  # or whatever limit is appropriate

query_router = APIRouter()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add this line to enable detailed logging for query_intent
logging.getLogger('query_intent').setLevel(logging.DEBUG)

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
    knowledge_gap: Optional[Dict] = None  


class ProcessRequest(BaseModel):
    document_id: str
    file_url: str
    metadata: Optional[Dict[str, Any]] = None

    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()



# Part 2: FastAPI Setup and NLP Transformer

# Startup and shutdown events manager
# In your bubble_backend.py, fix the lifespan function:

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, vector_store, conversation_handler, classifier
    try:
        # Initialize database pool with statement cache disabled
        logger.info("Initializing database pool...")
        pool = await asyncpg.create_pool(
            CONNECTION_STRING,
            min_size=5,
            max_size=20,
            statement_cache_size=0
        )
        logger.info("Database pool created successfully")
        
        # Verify/create database tables - DO ALL TABLES HERE TOGETHER
        async with pool.acquire() as conn:
            logger.info("Verifying database tables...")
            
            # Create messages table
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
            
            # Create drilling sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS drilling_sessions (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL UNIQUE,
                    workflow_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add verification for pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("Database tables verified/created successfully")
        
        # Initialize vector store for documents with better error handling
        logger.info("Initializing vector store...")
        try:
            vector_store = PGVector(
                collection_name="document_embeddings",
                connection_string=CONNECTION_STRING,
                embedding_function=OpenAIEmbeddings(),
                collection_metadata={"metadata_field_for_custom_id": "document_id"}
            )
            logger.info("Document vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Vector store initialization failed: {str(e)}"
            )
        
        # Initialize conversation handler
        conversation_handler = ConversationHandler(pool)
        logger.info("Conversation handler initialized successfully")
        
        # Initialize classifier
        classifier = ConstructionClassifier()
        logger.info("Construction classifier initialized successfully")
        
        # Initialize knowledge gap detector
        from knowledge_gaps.knowledge_gaps import get_detector
        get_detector(pool)
        logger.info("Knowledge gap detector initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
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
        self.intent_analyzer = SmartQueryIntentAnalyzer()
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
        try:
            # Analyze query intent
            intent_analysis = await self.intent_analyzer.analyze(query, context)
            
            # Get base category format
            category_format = self.format_mapper.get_format_for_category(
                classification['category']
            )
    
            # If it's a casual/discussion intent, use simplified format
            if intent_analysis and hasattr(intent_analysis, 'primary_intent') and intent_analysis.primary_intent in [QueryIntent.DISCUSSION, QueryIntent.CLARIFICATION]:
                formatted_content = await self._format_conversational(
                    raw_response,
                    classification['category'],
                    intent_analysis
                )
                return formatted_content, intent_analysis
    
            # For emergency intent, always use structured safety format
            if intent_analysis and hasattr(intent_analysis, 'primary_intent') and intent_analysis.primary_intent == QueryIntent.EMERGENCY:
                safety_format = self.format_mapper.get_format_for_category("SAFETY")
                formatted_content = await self._format_structured(
                    raw_response,
                    safety_format,
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
        except Exception as e:
            # Fallback to a simple formatting if analysis fails
            logger.warning(f"Content transformation failed: {str(e)}. Using fallback formatting.")
            return raw_response, IntentAnalysis(
                primary_intent=QueryIntent.INFORMATION,
                secondary_intents=[],
                confidence=0.5,
                reasoning="Fallback due to transformation error"
            )

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

        limiter = gpt4_limiter if "gpt-4" in self.llm.model_name else gpt35_limiter
        response = await rate_limited_call(
            self.llm.agenerate,
            limiter,
            [prompt.format(
                content=content,
                category=category,
                intent=intent_analysis.primary_intent.value
            )]
        )
        
        # Check if response is a list or has generations attribute
        if hasattr(response, 'generations') and response.generations:
            # Handle LangChain >= 0.0.267 response format
            if hasattr(response.generations[0][0], 'text'):
                return response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'content'):
                return response.generations[0][0].content
            elif isinstance(response.generations[0][0], str):
                return response.generations[0][0]
        
        # Handle direct string response or older versions
        if isinstance(response, str):
            return response
        
        # Handle list response directly
        if isinstance(response, list) and response:
            if isinstance(response[0], str):
                return response[0]
            elif hasattr(response[0], 'text'):
                return response[0].text
            elif hasattr(response[0], 'content'):
                return response[0].content
        
        # Fallback
        return content

   
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
        
        # Add rate limiting here
        limiter = gpt4_limiter if "gpt-4" in self.llm.model_name else gpt35_limiter
        response = await rate_limited_call(
            self.llm.agenerate,
            limiter,
            [prompt.format(
                content=content,
                missing_sections=", ".join(validation_errors)
            )]
        )
        
        # Check if response is a list or has generations attribute
        if hasattr(response, 'generations') and response.generations:
            # Handle LangChain >= 0.0.267 response format
            if hasattr(response.generations[0][0], 'text'):
                return response.generations[0][0].text
            elif hasattr(response.generations[0][0], 'content'):
                return response.generations[0][0].content
            elif isinstance(response.generations[0][0], str):
                return response.generations[0][0]
        
        # Handle direct string response or older versions
        if isinstance(response, str):
            return response
        
        # Handle list response directly
        if isinstance(response, list) and response:
            if isinstance(response[0], str):
                return response[0]
            elif hasattr(response[0], 'text'):
                return response[0].text
            elif hasattr(response[0], 'content'):
                return response[0].content
        
        # Fallback
        return content

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
            "conversation_history": conversation_context.dict().get("chat_history", []) if conversation_context else []
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
                    "emails": response.get("emails", []),  # <-- Updated to use emails
                    "query_type": "email",
                    "conversation_used": bool(conversation_context)
                }))

                # Replace your current logging with this more detailed version
        logger.info(f"Email query response structure: Keys = {list(response.keys())}")
        logger.info(f"Answer content: {response.get('answer', 'No answer')[:100]}...")
        
        # Log detailed information about all emails
        if "emails" in response and response["emails"]:
            logger.info(f"Number of emails returned: {len(response['emails'])}")
            logger.info("========== EMAIL DETAILS START ==========")
            
            for i, email in enumerate(response["emails"]):
                logger.info(f"========== EMAIL {i+1} OF {len(response['emails'])} ==========")
                logger.info(f"ID:         {email.get('id', 'No ID')}")
                logger.info(f"SUBJECT:    {email.get('subject', 'No subject')}")
                logger.info(f"SENDER:     {email.get('sender', 'No sender')}")
                logger.info(f"RECIPIENT:  {email.get('recipient', 'No recipient')}")
                logger.info(f"DATE:       {email.get('date', 'No date')}")
                logger.info(f"CONFIDENCE: {email.get('confidence', 'No confidence')}")
                
                # Log summary with clear formatting
                summary = email.get('summary', 'No summary')
                logger.info(f"SUMMARY:    {summary}")
                
                # Log a preview of content with clear formatting
                if 'content' in email and email['content']:
                    # Format the preview to be cleaner with line breaks replaced
                    content = email['content']
                    # Remove extra whitespace and replace newlines with spaces
                    content = ' '.join([line.strip() for line in content.split('\n')])
                    # Truncate if too long
                    if len(content) > 300:
                        content = content[:297] + "..."
                    logger.info(f"CONTENT PREVIEW: {content}")
                
                logger.info("-----------------------------------------")
            
            logger.info("========== EMAIL DETAILS END ==========")
        else:
            logger.info("No emails returned in the response")

        if "targeted_answer" in response:
            logger.info("========== TARGETED ANSWER DETAILS FROM ADAPTER ==========")
            logger.info(f"Format: {response['targeted_answer'].get('formatting_rules', {}).get('format_style', 'not specified')}")
            logger.info(f"Section Title: {response['targeted_answer'].get('formatting_rules', {}).get('section_title', 'none')}")
            
            # Log other important formatting rules
            for rule_name, rule_value in response['targeted_answer'].get('formatting_rules', {}).items():
                if rule_name not in ['format_style', 'section_title']:
                    logger.info(f"Rule - {rule_name}: {rule_value}")
            
            # Log a preview of the content
            content_preview = response['targeted_answer'].get('content', '')[:300]
            if len(response['targeted_answer'].get('content', '')) > 300:
                content_preview += "..."
            logger.info(f"Content Preview: {content_preview}")
            logger.info("=======================================================")
        
        # Log metadata
        if "metadata" in response:
            logger.info(f"Response metadata: {response['metadata']}")
            if response.get("metadata", {}).get("query_type") == "email_timeline":
                logger.info("========== COMPLETE TIMELINE FROM BUBBLE BACKEND ==========")
                logger.info(f"Timeline answer length: {len(response['answer'])} characters")
                # Log in chunks to handle large timelines
                answer = response['answer']
                for i in range(0, len(answer), 1000):
                    chunk = answer[i:i+1000]
                    logger.info(chunk)
                logger.info("========== END COMPLETE TIMELINE ==========")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in email query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add this function to bubble_backend.py
async def generate_knowledge_gap_response(
    query: str,
    knowledge_gap: KnowledgeGap,
    classification: QuestionType
) -> str:
    """Generate a specialized response for knowledge gap scenarios"""
    # Use LLM to generate a helpful 'I don't know' response
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    prompt = f"""
    Create a response for a construction question that I don't have enough information to answer confidently.
    
    Query: {query}
    Knowledge gap type: {knowledge_gap.gap_type}
    Domain: {knowledge_gap.domain or 'general construction'}
    
    The response should:
    1. Acknowledge the knowledge gap without using technical terms like 'knowledge gap'
    2. Explain what information would be needed to answer properly
    3. Suggest alternative approaches the user might take
    4. Use a conversational, helpful tone
    
    Response:
    """
    
    # Add rate limiting here
    response = await rate_limited_call(
        llm.ainvoke,
        gpt35_limiter,  # Using GPT-3.5 limiter since the model is gpt-3.5-turbo
        prompt
    )
    
    return response.content


async def process_document_query(
    request: QueryRequest,
    conversation_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Enhanced document query processing with improved source attribution"""
    async with concurrent_query_semaphore:
        total_start = time.time()
        start_time = datetime.utcnow()
        
        if not pool or not vector_store:
            raise HTTPException(status_code=503, detail="Service not fully initialized")
    
        try:
            # REMOVED: hardcoded response check - now handled at main endpoint level
            
            # Question classification
            classification_start = time.time()
            classification = await classifier.classify_question(
                question=request.query,
                conversation_context=conversation_context,
                current_context=request.context.dict()
            )
            classification_end = time.time()
            perf_logger.info(f"Classification: {(classification_end - classification_start) * 1000:.2f}ms")
    
            # Intent analysis
            intent_start = time.time()
            intent_analyzer = SmartQueryIntentAnalyzer()
            intent_analysis = await rate_limited_call(
                intent_analyzer.analyze,
                gpt35_limiter,  # Assume it uses GPT-3.5 for intent analysis
                request.query, 
                request.context.dict()
            )
            intent_end = time.time()
            perf_logger.info(f"Intent analysis: {(intent_end - intent_start) * 1000:.2f}ms")
    
            # Continue with rest of your existing logic...
            # Check for comparison queries (which we'll now route to variance analysis)
            routing_start = time.time()
            is_comparison = False
            
            # Log the final routing decision with reasoning
            variance_match = is_variance_analysis_query(request.query)
            logger.info(f"Routing decision components - is_comparison: {is_comparison}, category: {classification.category}, variance_match: {variance_match}")
            
            if is_comparison or classification.category == "COMPARISON" or variance_match:
                logger.info(f"Routing to variance analysis because: is_comparison={is_comparison}, category={classification.category}, variance_match={variance_match}")
                variance_start = time.time()
                result = await process_variance_query(
                    request=request,
                    classification=classification,
                    vector_store=vector_store,
                    conversation_context=conversation_context,
                    intent_analysis=intent_analysis
                )
                variance_end = time.time()
                perf_logger.info(f"Variance analysis: {(variance_end - variance_start) * 1000:.2f}ms")
                perf_logger.info(f"Total document processing (variance): {(variance_end - total_start) * 1000:.2f}ms")
                return result
            routing_end = time.time()
            perf_logger.info(f"Variance routing check: {(routing_end - routing_start) * 1000:.2f}ms")
            
            # Initialize enhanced response generator
            generator_start = time.time()
            enhanced_source_handler = EnhancedSmartResponseGenerator(
                llm=ChatOpenAI(
                    model_name="gpt-4" if classification.category == "SAFETY" else "gpt-3.5-turbo",
                    temperature=0.7
                ),
                vector_store=vector_store,
                classifier=classifier,
                min_confidence=0.6  # Adjustable threshold
            )
            generator_end = time.time()
            perf_logger.info(f"Initialize response generator: {(generator_end - generator_start) * 1000:.2f}ms")
            
            # Check if we need sources for this query
            sources_check_start = time.time()
            needs_sources = await enhanced_source_handler.should_use_sources(
                query=request.query,
                classification=classification,
                intent_analysis=intent_analysis
            )
            sources_check_end = time.time()
            perf_logger.info(f"Sources check: {(sources_check_end - sources_check_start) * 1000:.2f}ms")
            
            # Configure search based on classification and request
            filter_start = time.time()
            metadata_filter = {}
            if request.document_ids:
                metadata_filter["document_id"] = {"$in": request.document_ids}
            if request.metadata_filter:
                metadata_filter.update(request.metadata_filter)
            filter_end = time.time()
            perf_logger.info(f"Metadata filter setup: {(filter_end - filter_start) * 1000:.2f}ms")
    
            # Initialize memory with conversation history
            memory_start = time.time()
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
            memory_end = time.time()
            perf_logger.info(f"Memory initialization: {(memory_end - memory_start) * 1000:.2f}ms")
    
            # Get relevant documents directly using the enhanced method
            docs_start = time.time()
            if needs_sources:
                documents = await enhanced_source_handler._get_relevant_documents(
                    query=request.query,
                    classification=classification,
                    metadata_filter=metadata_filter,
                    k=2 if classification.category == "SAFETY" else 2
                )
            else:
                documents = []
            docs_end = time.time()
            perf_logger.info(f"Document retrieval: {(docs_end - docs_start) * 1000:.2f}ms")
    
            # Check document confidence and knowledge gaps
            knowledge_start = time.time()
            retrieved_content_confidence = 0.0
            if documents:
                # Average the semantic similarities if available
                similarities = [doc.metadata.get('similarity', 0) for doc in documents if hasattr(doc, 'metadata') and 'similarity' in doc.metadata]
                if similarities:
                    retrieved_content_confidence = sum(similarities) / len(similarities)
                else:
                    # Fallback to moderate confidence if similarity scores aren't available
                    retrieved_content_confidence = 0.6 if len(documents) >= 3 else 0.4
            
            # Detect knowledge gaps
            KNOWLEDGE_GAP_THRESHOLD = 0.65  # Match the threshold from logs
            if documents and retrieved_content_confidence >= KNOWLEDGE_GAP_THRESHOLD:
                # Create a simple "no gap" object without running the detection logic
                from knowledge_gaps import KnowledgeGap
                knowledge_gap = KnowledgeGap(
                    is_gap=False,
                    gap_type=None,
                    domain=None,
                    subcategory=None,
                    recommended_action=None
                )
                logger.info(f"Skipped knowledge gap detection - document confidence {retrieved_content_confidence} exceeds threshold {KNOWLEDGE_GAP_THRESHOLD}")
            else:
                # Only run knowledge gap detection if confidence is below threshold
                from knowledge_gaps.knowledge_gaps import detect_knowledge_gap
                knowledge_gap = await detect_knowledge_gap(
                    query=request.query,
                    classification=classification,
                    source_documents=documents,
                    context=request.context.dict() if hasattr(request.context, 'dict') else {},
                    pool=pool
                )
            knowledge_end = time.time()
            perf_logger.info(f"Knowledge gap check: {(knowledge_end - knowledge_start) * 1000:.2f}ms")
            
            # Include knowledge gap information in the response metadata
            knowledge_gap_metadata = None
            if knowledge_gap.is_gap:
                knowledge_gap_metadata = {
                    "gap_detected": True,
                    "gap_type": knowledge_gap.gap_type,
                    "domain": knowledge_gap.domain,
                    "subcategory": knowledge_gap.subcategory,
                    "recommended_action": knowledge_gap.recommended_action
                }
                logger.info(f"Knowledge gap detected: {knowledge_gap.domain}/{knowledge_gap.subcategory} - {knowledge_gap.gap_type}")
    
            if knowledge_gap.is_gap:
                # Generate a specialized "I don't know" response
                gap_response_start = time.time()
                not_knowing_response = await generate_knowledge_gap_response(
                    query=request.query,
                    knowledge_gap=knowledge_gap,
                    classification=classification
                )
                gap_response_end = time.time()
                processing_time = (gap_response_end - gap_response_start)
                perf_logger.info(f"Knowledge gap response: {processing_time * 1000:.2f}ms")
                perf_logger.info(f"Total document processing (knowledge gap): {(gap_response_end - total_start) * 1000:.2f}ms")
                
                # Return early with the specialized response
                return {
                    "status": "success",
                    "answer": not_knowing_response,
                    "classification": classification.dict(),
                    "sources": [],  # No sources since we don't know enough
                    "metadata": ResponseMetadata(
                        category="KNOWLEDGE_GAP",  # Override the category 
                        confidence=classification.confidence,
                        source_count=0,
                        processing_time=processing_time,
                        conversation_context_used=bool(conversation_context),
                        intent_analysis=intent_analysis.dict() if intent_analysis else None,
                        knowledge_gap=knowledge_gap_metadata,
                        format_used="knowledge_gap_format"  # Custom format identifier
                    ).dict()
                }
    
            # Initialize QA chain with appropriate model
            chain_start = time.time()
            limiter = gpt4_limiter if classification.category == "SAFETY" else gpt35_limiter
    
            if documents:
                # Create retriever with documents
                from langchain.schema import Document
                from langchain.retrievers.document_compressors import DocumentCompressorPipeline
                from langchain.retrievers import ContextualCompressionRetriever
                # Create a simple retriever from the documents
                from langchain.retrievers import TimeWeightedVectorStoreRetriever
                
                retriever_start = time.time()
                retriever = vector_store.as_retriever(
                    search_kwargs={
                        "k": len(documents),
                        "filter": metadata_filter if metadata_filter else None
                    }
                )
                retriever_end = time.time()
                perf_logger.info(f"Retriever setup: {(retriever_end - retriever_start) * 1000:.2f}ms")
                
                qa_setup_start = time.time()
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
                qa_setup_end = time.time()
                perf_logger.info(f"QA chain setup: {(qa_setup_end - qa_setup_start) * 1000:.2f}ms")
                
                # Define an async function that calls qa with rate limiting
                async def execute_qa_with_rate_limit():
                    qa_exec_start = time.time()
                    await limiter.acquire()
                    try:
                        # We still need to run in executor because qa is not async
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: qa({
                                "question": request.query,
                                "chat_history": memory.chat_memory.messages
                            })
                        )
                        qa_exec_end = time.time()
                        perf_logger.info(f"QA execution: {(qa_exec_end - qa_exec_start) * 1000:.2f}ms")
                        return result
                    finally:
                        limiter.release()
            
                # Execute with rate limiting
                chain_response = await execute_qa_with_rate_limit()
                
                response_content = chain_response.get("answer", "No response generated")
                source_documents = chain_response.get("source_documents", [])
            else:
                # For queries that don't need sources
                simple_qa_start = time.time()
                simple_qa = ConversationalChain.from_llm(
                    llm=ChatOpenAI(
                        model_name="gpt-4" if classification.category == "SAFETY" else "gpt-3.5-turbo",
                        temperature=0.7
                    ),
                    memory=memory,
                    verbose=True
                )
                simple_qa_end = time.time()
                perf_logger.info(f"Simple QA setup: {(simple_qa_end - simple_qa_start) * 1000:.2f}ms")
                
                # Define an async function for the simple_qa with rate limiting
                async def execute_simple_qa_with_rate_limit():
                    simple_exec_start = time.time()
                    await limiter.acquire()
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: simple_qa({
                                "question": request.query,
                                "chat_history": memory.chat_memory.messages
                            })
                        )
                        simple_exec_end = time.time()
                        perf_logger.info(f"Simple QA execution: {(simple_exec_end - simple_exec_start) * 1000:.2f}ms")
                        return result
                    finally:
                        limiter.release()
                
                # Execute with rate limiting
                chain_response = await execute_simple_qa_with_rate_limit()
                
                response_content = chain_response.get("answer", "No response generated")
                source_documents = []
            chain_end = time.time()
            perf_logger.info(f"Chain processing: {(chain_end - chain_start) * 1000:.2f}ms")
    
            # Transform content based on classification and intent
            transform_start = time.time()
            transformer = NLPTransformer()
            
            formatted_response, intent_analysis = await transformer.transform_content(
                query=request.query,
                raw_response=response_content,
                context=request.context.dict(),
                source_documents=source_documents,
                classification=classification.dict()
            )
            transform_end = time.time()
            perf_logger.info(f"Content transformation: {(transform_end - transform_start) * 1000:.2f}ms")
    
            # If we're using sources, extract attributions with the enhanced method
            attribution_start = time.time()
            source_attributions = []
            if needs_sources and source_documents:
                source_attributions = await enhanced_source_handler._extract_attributions(
                    response_content,
                    source_documents,
                    classification,
                    pool
                )
            attribution_end = time.time()
            perf_logger.info(f"Attribution extraction: {(attribution_end - attribution_start) * 1000:.2f}ms")
    
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Prepare sources for response
            source_format_start = time.time()
            sources = []
            if needs_sources and source_attributions:
                for attr in source_attributions:
                    # Check if this is an EnhancedSourceAttribution
                    if hasattr(attr, 'semantic_similarity'):
                        # Include enhanced attributes
                        sources.append({
                            "id": attr.source_id,
                            "title": attr.title or next(
                                (doc.metadata.get("title") for doc in source_documents 
                                 if doc.metadata.get("document_id") == attr.source_id),
                                "Unknown Document"
                            ),
                            "page": attr.page_number,
                            "confidence": attr.confidence,
                            "semantic_similarity": attr.semantic_similarity,
                            "content_overlap": attr.content_overlap,
                            "excerpt": attr.content,
                            "document_type": attr.document_type
                        })
                    else:
                        # Standard attribution format
                        sources.append({
                            "id": attr.source_id,
                            "title": next(
                                (doc.metadata.get("title") for doc in source_documents 
                                 if doc.metadata.get("document_id") == attr.source_id),
                                "Unknown Document"
                            ),
                            "page": attr.page_number,
                            "confidence": attr.confidence,
                            "excerpt": attr.content
                        })
            
            format_mapper = FormatMapper()
            category_format = format_mapper.get_format_for_category(classification.category)
            validation_errors = []
            source_format_end = time.time()
            perf_logger.info(f"Source formatting: {(source_format_end - source_format_start) * 1000:.2f}ms")
    
            # Prepare response
            response_prep_start = time.time()
            response = {
                "status": "success",
                "answer": formatted_response,
                "classification": classification.dict(),
                "sources": sources,
                "metadata": ResponseMetadata(
                    category=classification.category,
                    confidence=classification.confidence,
                    source_count=len(source_documents),
                    processing_time=processing_time,
                    conversation_context_used=bool(conversation_context),
                    intent_analysis=intent_analysis.dict(),
                    knowledge_gap=knowledge_gap_metadata,
                    format_used=getattr(category_format, 'style', FormatStyle.NARRATIVE).value 
                               if hasattr(category_format, 'style') else "unknown",
                    format_validation_errors=validation_errors if 'validation_errors' in locals() else []
                ).dict()
            }
            response_prep_end = time.time()
            perf_logger.info(f"Response preparation: {(response_prep_end - response_prep_start) * 1000:.2f}ms")
    
            total_end = time.time()
            perf_logger.info(f"Total document processing: {(total_end - total_start) * 1000:.2f}ms")
            return response
    
        except Exception as e:
            total_end = time.time()
            perf_logger.info(f"Failed document processing: {(total_end - total_start) * 1000:.2f}ms (error: {str(e)})")
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


async def download_file(file_url: str) -> bytes:
    """Download file from URL with improved error handling"""
    logger.info(f"Attempting to download file from URL: {file_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download file. Status: {response.status}")
                    raise HTTPException(status_code=response.status)
                return await response.read()
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main query endpoint that routes to appropriate handler
# Main query endpoint that routes to appropriate handler
@query_router.post("/query")
async def query_documents(request: QueryRequest):
    """Enhanced main query endpoint with conversation history and classification"""
    total_start = time.time()
    try:
        # CHECK FOR HARDCODED RESPONSES FIRST - BEFORE ANY OTHER PROCESSING
        hardcoded_response = get_hardcoded_response(request.query)
        if hardcoded_response:
            logger.info(f"Using hardcoded response for query: '{request.query}'")
            perf_logger.info(f"Hardcoded response - bypassing normal processing: {time.time() - total_start:.4f}s")
            
            # Save the interaction
            await conversation_handler.save_conversation_turn(
                conversation_id=request.conversation_id,
                role='user',
                content=request.query,
                metadata={
                    "context": request.context.dict(),
                    "query_type": "hardcoded",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await conversation_handler.save_conversation_turn(
                conversation_id=request.conversation_id,
                role='assistant',
                content=hardcoded_response["answer"],
                metadata={
                    "classification": hardcoded_response.get("classification", {}),
                    "metadata": hardcoded_response.get("metadata", {}),
                    "query_type": "hardcoded",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return hardcoded_response

        # Load conversation history
        history_start = time.time()
        conversation_context = await conversation_handler.load_conversation_history(
            conversation_id=request.conversation_id
        )
        history_end = time.time()
        perf_logger.info(f"Load conversation history: {(history_end - history_start) * 1000:.2f}ms")

       
        # Log the request details and routing decision factors
        logger.info(f"Processing query for conversation {request.conversation_id}: '{request.query}'")
        logger.info(f"Info sources: {request.info_sources}")
        
        # Make and log the routing decision
        routing_start = time.time()
        use_email_processing = is_email_query(request)
        logger.info(f"Routing decision: {'EMAIL' if use_email_processing else 'DOCUMENT'} processing")
        routing_end = time.time()
        perf_logger.info(f"Routing decision: {(routing_end - routing_start) * 1000:.2f}ms")
        
        # Process query with context
        processing_start = time.time()
        if use_email_processing:
            logger.info("Starting email query processing")
            response = await process_email_query(
                request,
                conversation_context=conversation_context
            )
            # Ensure query_type is set in the response
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["query_type"] = "email"
            logger.info("Email query processing completed successfully")
        else:
            logger.info("Starting document query processing")
            response = await process_document_query(
                request,
                conversation_context=conversation_context
            )
            # Ensure query_type is set in the response
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["query_type"] = "document"
            logger.info("Document query processing completed successfully")
        processing_end = time.time()
        perf_logger.info(f"Query processing: {(processing_end - processing_start) * 1000:.2f}ms")

        # Save the interaction
        save_start = time.time()
        await conversation_handler.save_conversation_turn(
            conversation_id=request.conversation_id,
            role='user',
            content=request.query,
            metadata={
                "context": request.context.dict(),
                "query_type": "email" if use_email_processing else "document",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await conversation_handler.save_conversation_turn(
            conversation_id=request.conversation_id,
            role='assistant',
            content=response["answer"],
            metadata={
                "emails": response.get("emails", []) if use_email_processing else [],
                "sources": response.get("sources", []) if not use_email_processing else [],
                "classification": response.get("classification", {}),
                "metadata": response.get("metadata", {}),
                "query_type": "email" if use_email_processing else "document",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        save_end = time.time()
        perf_logger.info(f"Save interaction: {(save_end - save_start) * 1000:.2f}ms")
        
        total_end = time.time()
        perf_logger.info(f"Total request: {(total_end - total_start) * 1000:.2f}ms")
        
        return response
            
    except Exception as e:
        total_end = time.time()
        perf_logger.info(f"Failed request: {(total_end - total_start) * 1000:.2f}ms (error: {str(e)})")
        logger.error(f"Error in query routing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def initialize_components():
    """Initialize all components needed for the query endpoint"""
    global pool, vector_store, conversation_handler, classifier
    
    # Initialize database pool
    pool = await asyncpg.create_pool(
        CONNECTION_STRING,
        min_size=5,
        max_size=20,
        statement_cache_size=0
    )
    
    # Create rate-limited embeddings
    class RateLimitedEmbeddings(OpenAIEmbeddings):
        async def aembed_documents(self, texts):
            return await rate_limited_call(
                super().aembed_documents,
                embeddings_limiter,
                texts
            )
        
        async def aembed_query(self, text):
            return await rate_limited_call(
                super().aembed_query,
                embeddings_limiter,
                text
            )
    
    # Initialize vector store with rate-limited embeddings
    vector_store = PGVector(
        collection_name="document_embeddings",
        connection_string=CONNECTION_STRING,
        embedding_function=RateLimitedEmbeddings(),
        collection_metadata={"metadata_field_for_custom_id": "document_id"}
    )
    
    # Initialize conversation handler
    conversation_handler = ConversationHandler(pool)
    
    # Initialize classifier
    classifier = ConstructionClassifier()
    
    return {
        "pool": pool,
        "vector_store": vector_store,
        "conversation_handler": conversation_handler,
        "classifier": classifier
    }


# Keep the standalone app definition for direct usage
app = FastAPI(lifespan=lifespan)
app.include_router(query_router)

app.include_router(create_knowledge_gap_router(pool))

# Export the router for use in other files
__all__ = ['query_router', 'initialize_components', 'process_document_query', 'process_email_query']


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("bubble_backend:app", host="0.0.0.0", port=port, reload=True)
