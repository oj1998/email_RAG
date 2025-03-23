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
from format_mapper import FormatMapper, CategoryFormat, FormatStyle
from query_intent import SmartQueryIntentAnalyzer, QueryIntent, IntentAnalysis
from smart_response_generator import SmartResponseGenerator
from enhanced_similarity import EnhancedSmartResponseGenerator, EnhancedSourceAttribution

import aiohttp
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import APIRouter

from comparisons import (
    ComparisonRetriever,
    ComparisonAnalyzer,
    ComparisonFormatter,
    ComparisonFormatStyle
)

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
        
        # Verify/create database tables
        async with pool.acquire() as conn:
            logger.info("Verifying database tables...")
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
            # Add verification for pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("Database tables verified/created successfully")
        
        # Initialize vector store for documents with better error handling
        logger.info("Initializing vector store...")
        try:
            vector_store = PGVector(
                collection_name="document_embeddings",
                connection_string=CONNECTION_STRING,
                embedding_function=OpenAIEmbeddings()
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
        
        response = await self.llm.agenerate([prompt.format(
            content=content,
            category=category,
            intent=intent_analysis.primary_intent.value
        )])
        
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
        
        response = await self.llm.agenerate(prompt.format(
            content=content,
            missing_sections=", ".join(validation_errors)
        ))
        
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


async def process_document_query(
    request: QueryRequest,
    conversation_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Enhanced document query processing with improved source attribution"""
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

        

        # Perform intent analysis
        intent_analyzer = SmartQueryIntentAnalyzer()
        intent_analysis = await intent_analyzer.analyze(
            request.query, 
            request.context.dict()
        )

        is_comparison = False
        if hasattr(classification, 'suggested_format') and classification.suggested_format:
            is_comparison = classification.suggested_format.get("is_comparison", False)
        
        # Route to comparison processing if needed
        if is_comparison or classification.category == "COMPARISON":
            logger.info(f"Routing to comparison query processing for query: {request.query}")
            return await process_comparison_query(
                request=request,
                classification=classification,
                conversation_context=conversation_context,
                intent_analysis=intent_analysis
            )
        
        # Initialize enhanced response generator
        enhanced_source_handler = EnhancedSmartResponseGenerator(
            llm=ChatOpenAI(
                model_name="gpt-4" if classification.category == "SAFETY" else "gpt-3.5-turbo",
                temperature=0.7
            ),
            vector_store=vector_store,
            classifier=classifier,
            min_confidence=0.6  # Adjustable threshold
        )
        
        # Check if we need sources for this query
        needs_sources = await enhanced_source_handler.should_use_sources(
            query=request.query,
            classification=classification,
            intent_analysis=intent_analysis
        )
        
        # Configure search based on classification and request
        metadata_filter = {}
        if request.document_ids:
            metadata_filter["document_id"] = {"$in": request.document_ids}
        if request.metadata_filter:
            metadata_filter.update(request.metadata_filter)

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

        # Get relevant documents directly using the enhanced method
        if needs_sources:
            documents = await enhanced_source_handler._get_relevant_documents(
                query=request.query,
                classification=classification,
                metadata_filter=metadata_filter,
                k=8 if classification.category == "SAFETY" else 5
            )
        else:
            documents = []

        # Initialize QA chain with appropriate model
        if documents:
            # Create retriever with documents
            from langchain.schema import Document
            from langchain.retrievers.document_compressors import DocumentCompressorPipeline
            from langchain.retrievers import ContextualCompressionRetriever

            # Create a simple retriever from the documents
            from langchain.retrievers import TimeWeightedVectorStoreRetriever
            
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": len(documents),
                    "filter": metadata_filter if metadata_filter else None
                }
            )
            
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
            
            response_content = chain_response.get("answer", "No response generated")
            source_documents = chain_response.get("source_documents", [])
        else:
            # For queries that don't need sources
            simple_qa = ConversationalChain.from_llm(
                llm=ChatOpenAI(
                    model_name="gpt-4" if classification.category == "SAFETY" else "gpt-3.5-turbo",
                    temperature=0.7
                ),
                memory=memory,
                verbose=True
            )
            
            chain_response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: simple_qa({
                    "question": request.query,
                    "chat_history": memory.chat_memory.messages
                })
            )
            
            response_content = chain_response.get("answer", "No response generated")
            source_documents = []

        # Transform content based on classification and intent
        transformer = NLPTransformer()
        
        formatted_response, intent_analysis = await transformer.transform_content(
            query=request.query,
            raw_response=response_content,
            context=request.context.dict(),
            source_documents=source_documents,
            classification=classification.dict()
        )

        # If we're using sources, extract attributions with the enhanced method
        source_attributions = []
        if needs_sources and source_documents:
            source_attributions = await enhanced_source_handler._extract_attributions(
                response_content,
                source_documents,
                classification
            )

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare sources for response
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

        # Prepare response
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
                format_used=getattr(category_format, 'style', FormatStyle.NARRATIVE).value 
                           if hasattr(category_format, 'style') else "unknown",
                format_validation_errors=validation_errors if 'validation_errors' in locals() else []
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
@query_router.post("/query")
async def query_documents(request: QueryRequest):
    """Enhanced main query endpoint with conversation history and classification"""
    try:
        # Load conversation history
        conversation_context = await conversation_handler.load_conversation_history(
            conversation_id=request.conversation_id
        )
        
        # Log the request details and routing decision factors
        logger.info(f"Processing query for conversation {request.conversation_id}: '{request.query}'")
        logger.info(f"Info sources: {request.info_sources}")
        
        # Make and log the routing decision
        use_email_processing = is_email_query(request)
        logger.info(f"Routing decision: {'EMAIL' if use_email_processing else 'DOCUMENT'} processing")
        
        # Process query with context
        # This is likely in your main query endpoint or in the query_documents function
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

        # Save the interaction
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
        
        return response
            
    except Exception as e:
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
    
    # Initialize vector store
    vector_store = PGVector(
        collection_name="document_embeddings",
        connection_string=CONNECTION_STRING,
        embedding_function=OpenAIEmbeddings()
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

async def process_comparison_query(
    request: QueryRequest,
    classification: QuestionType,
    conversation_context: Optional[Dict] = None,
    intent_analysis=None
) -> Dict[str, Any]:
    """Process a query asking for comparison between options"""
    start_time = datetime.utcnow()
    logger.info(f"Processing comparison query: {request.query}")
    
    try:
        # Import comparison modules
        from comparisons import (
            ComparisonRetriever,
            ComparisonAnalyzer,
            ComparisonFormatter,
            ComparisonFormatStyle
        )
        from langchain_openai import ChatOpenAI
        
        # Initialize comparison components
        comparison_retriever = ComparisonRetriever(
            vector_store=vector_store,
            llm=ChatOpenAI(model_name="gpt-4"),
            min_confidence=0.6
        )
        
        comparison_analyzer = ComparisonAnalyzer(
            llm=ChatOpenAI(model_name="gpt-4")
        )
        
        comparison_formatter = ComparisonFormatter()
        
        # 1. Extract topics and retrieve documents
        topics, topic_documents = await comparison_retriever.retrieve_for_comparison(
            query=request.query,
            metadata_filter=request.metadata_filter,
            k_per_topic=3
        )
        
        # Log the topics detected
        logger.info(f"Comparison query detected. Topics: {', '.join(topics)}")
        
        # 2. Analyze the comparison
        comparison_analysis = await comparison_analyzer.generate_comparison(
            query=request.query,
            topics=topics,
            topic_documents=topic_documents,
            domain_context="construction industry"
        )
        
        # 3. Format the comparison
        # Determine format style based on classification
        format_style = ComparisonFormatStyle.STRUCTURED
        if classification.suggested_format:
            style = classification.suggested_format.get("style", "").lower()
            if style == "step_by_step":
                format_style = ComparisonFormatStyle.STRUCTURED
            elif style == "narrative":
                format_style = ComparisonFormatStyle.NARRATIVE
            elif style == "tabular" or style == "table":
                format_style = ComparisonFormatStyle.TABULAR
        
        formatted_comparison = comparison_formatter.format_comparison(
            analysis=comparison_analysis,
            format_style=format_style
        )
        
        # Prepare sources for response
        sources = []
        for topic, analysis in comparison_analysis.topic_analyses.items():
            for doc_id in analysis.document_sources:
                # Find matching document in topic_documents
                for topic_docs in topic_documents.values():
                    for doc in topic_docs:
                        if doc.metadata.get("document_id") == doc_id or doc.metadata.get("source_id") == doc_id:
                            sources.append({
                                "id": doc_id,
                                "title": doc.metadata.get("title", "Unknown Document"),
                                "page": doc.metadata.get("page"),
                                "confidence": 0.8,  # Default confidence for comparison sources
                                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                "topic": topic  # Add topic to show which option this source belongs to
                            })
                            break
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare response
        response = {
            "status": "success",
            "answer": formatted_comparison,
            "classification": classification.dict(),
            "sources": sources,
            "metadata": {
                "category": classification.category,
                "confidence": classification.confidence,
                "is_comparison": True,
                "compared_topics": topics,
                "source_count": len(sources),
                "processing_time": processing_time,
                "conversation_context_used": bool(conversation_context)
            }
        }
        
        logger.info(f"Comparison query processing completed in {processing_time:.2f} seconds")
        return response
        
    except Exception as e:
        logger.error(f"Error in comparison query processing: {str(e)}", exc_info=True)
        
        # Fallback to standard processing
        logger.info("Falling back to standard query processing")
        return await process_document_query(
            request=request,
            classification=classification,
            conversation_context=conversation_context,
            intent_analysis=intent_analysis
        )

# Keep the standalone app definition for direct usage
app = FastAPI(lifespan=lifespan)
app.include_router(query_router)

# Export the router for use in other files
__all__ = ['query_router', 'initialize_components', 'process_document_query', 'process_email_query']


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("bubble_backend:app", host="0.0.0.0", port=port, reload=True)
