from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from loaders.vector_store_loader import VectorStoreFactory
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

# Import our new intent-based formatting system
# Comment this out if you want to disable the new formatting system
from Email_Output.email_output_adapter import EmailOutputManager

# Set up logging
logger = logging.getLogger(__name__)

# Global email QA system instance
EMAIL_QA_SYSTEM = None

# Initialize the email output manager for intent-based formatting
# You can set this to None to disable the new formatting
try:
    EMAIL_OUTPUT_MANAGER = EmailOutputManager()
    logger.info("Intent-based email formatting system initialized successfully")
except ImportError:
    logger.info("Intent-based email formatting system not available")
    EMAIL_OUTPUT_MANAGER = None
except Exception as e:
    logger.warning(f"Failed to initialize intent-based email formatting: {e}")
    EMAIL_OUTPUT_MANAGER = None

# Enable/disable intent-based formatting with this flag
USE_INTENT_FORMATTING = True

# Type definitions without importing from models
# This prevents circular imports
QueryRequest = Dict[str, Any]
EmailFilter = Dict[str, Any]

async def get_email_qa_system():
    """Initialize or return existing email QA system"""
    global EMAIL_QA_SYSTEM
    if EMAIL_QA_SYSTEM is None:
        try:
            # Initialize embedding model
            from langchain_openai import OpenAIEmbeddings
            embeddings_model = OpenAIEmbeddings()
            
            # Initialize language model
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-3.5-turbo",
                temperature=0.2
            )
            
            # Initialize vector store
            vector_store = VectorStoreFactory.create(
                embeddings_model,
                config={
                    "type": "supabase",
                    "supabase_url": os.getenv("SUPABASE_URL"),
                    "supabase_key": os.getenv("SUPABASE_SERVICE_KEY"),
                    "collection_name": "email_embeddings"
                }
            )
            
            # Create QA system
            EMAIL_QA_SYSTEM = EmailQASystem(
                vector_store=vector_store,
                embeddings_model=embeddings_model,
                llm=llm,
                k=5,
                use_reranker=True
            )
            logger.info("Email QA system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize email QA system: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Email system initialization failed: {str(e)}"
            )
    
    return EMAIL_QA_SYSTEM

def extract_time_filters(context: Dict[str, Any]) -> Dict[str, str]:
    """Extract time-based filters from context"""
    filters = {}
    if "timeframe" in context:
        timeframe = context.get("timeframe", "").lower()
        now = datetime.now()
        
        if "yesterday" in timeframe:
            yesterday = (now - timedelta(days=1))
            filters["after_date"] = yesterday.strftime("%Y/%m/%d")
            filters["before_date"] = now.strftime("%Y/%m/%d")
        elif "last week" in timeframe:
            last_week = (now - timedelta(days=7))
            filters["after_date"] = last_week.strftime("%Y/%m/%d")
            filters["before_date"] = now.strftime("%Y/%m/%d")
        elif "last month" in timeframe:
            last_month = (now - timedelta(days=30))
            filters["after_date"] = last_month.strftime("%Y/%m/%d")
            filters["before_date"] = now.strftime("%Y/%m/%d")
            
    return filters

async def process_email_query(query: str, conversation_id: str, context: Dict[str, Any] = None, email_filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process an email-specific query with optional intent-based formatting"""
    try:
        # Record start time for performance tracking
        start_time = datetime.now()
        
        # Get QA system
        qa_system = await get_email_qa_system()
        
        # Handle filters
        filter_options = {}
        if email_filters:
            filter_options = {k: v for k, v in email_filters.items() if v is not None}
        elif context:
            filter_options.update(extract_time_filters(context))
            
        # Get answer
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_system.query(
                question=query,
                filters=filter_options,
                k=5
            )
        )
        
        # Get relevant emails
        relevant_emails = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_system.get_relevant_emails(
                query=query,
                filters=filter_options,
                k=3
            )
        )
        
        # Format sources with more details
        sources = [
            {
                "id": email.metadata.get("email_id", "unknown"),
                "title": email.metadata.get("subject", "Email"),
                "sender": email.metadata.get("sender", "Unknown"),
                "recipient": email.metadata.get("recipient", "Unknown Recipient"),
                "date": email.metadata.get("date", ""),
                "confidence": email.metadata.get("relevance_score", 0),
                "excerpt": email.page_content[:200] + "..." if len(email.page_content) > 200 else email.page_content
            }
            for email in relevant_emails
        ]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare metadata
        metadata = {
            "query_type": "email",
            "processing_time": processing_time,
            "source_count": len(sources),
            "conversation_context_used": bool(context and context.get("conversation_history")),
            "filters_applied": filter_options
        }
        
        # Apply intent-based formatting if enabled
        if USE_INTENT_FORMATTING and EMAIL_OUTPUT_MANAGER is not None:
            try:
                logger.info(f"Applying intent-based formatting to email response for query: '{query}'")
                formatted_response = await EMAIL_OUTPUT_MANAGER.process_response(
                    query=query,
                    raw_response=answer,
                    sources=sources,
                    context=context,
                    metadata=metadata
                )
                
                # Add formatting info to metadata
                formatted_response["metadata"]["formatting_applied"] = True
                
                return formatted_response
            except Exception as e:
                logger.warning(f"Intent-based formatting failed: {str(e)}", exc_info=True)
                logger.info("Falling back to standard response format")
                # Continue with standard formatting if intent-based formatting fails
        
        # Return standard response if intent formatting is disabled or failed
        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error in email query processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Email query processing failed: {str(e)}"
        )

def create_email_router():
    """Create and configure the email-specific router"""
    router = APIRouter(prefix="/email", tags=["email"])
    
    @router.post("/query")
    async def query_emails(request: Dict[str, Any]):
        """Query emails endpoint"""
        if not request.get("query"):
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Check if intent formatting should be forced on or off for this request
        force_intent_formatting = request.get("force_intent_formatting")
        
        # Store the original flag
        original_flag = globals().get("USE_INTENT_FORMATTING")
        
        try:
            # Override flag if specified in request
            if force_intent_formatting is not None:
                globals()["USE_INTENT_FORMATTING"] = force_intent_formatting
                
            return await process_email_query(
                query=request.get("query"),
                conversation_id=request.get("conversation_id"),
                context=request.get("context"),
                email_filters=request.get("email_filters")
            )
        finally:
            # Restore original flag
            if force_intent_formatting is not None:
                globals()["USE_INTENT_FORMATTING"] = original_flag
    
    @router.get("/status")
    async def email_system_status():
        """Check email system status"""
        try:
            qa_system = await get_email_qa_system()
            return {
                "status": "healthy",
                "initialized": qa_system is not None,
                "intent_formatting_available": EMAIL_OUTPUT_MANAGER is not None,
                "intent_formatting_enabled": USE_INTENT_FORMATTING,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @router.post("/toggle_formatting")
    async def toggle_intent_formatting(enabled: bool = True):
        """Toggle intent-based formatting on or off"""
        global USE_INTENT_FORMATTING
        previous_value = USE_INTENT_FORMATTING
        USE_INTENT_FORMATTING = enabled
        return {
            "status": "success",
            "intent_formatting_enabled": USE_INTENT_FORMATTING,
            "previous_value": previous_value
        }
    
    return router

def is_email_query(request_data: Dict[str, Any]) -> bool:
    """Determine if a query should be routed to the email system"""
    # First, check if email was explicitly selected as an info source
    info_sources = request_data.get("info_sources", [])
    if info_sources and any(
        source.lower() in ["emails", "email", "mail", "gmail"]
        for source in info_sources
    ):
        logger.info(f"Routing to EMAIL: Source selection specified email ({info_sources})")
        return True
        
    # Then check for email filters
    if request_data.get("email_filters"):
        logger.info("Routing to EMAIL: Email filters specified")
        return True
        
    # Finally check for email keywords in the query
    query = request_data.get("query", "").lower()
    email_keywords = [
        "email", "mail", "gmail", "inbox", "message",
        "received", "sent", "folder", "label"
    ]
    
    found_keywords = [keyword for keyword in email_keywords if keyword in query]
    if found_keywords:
        logger.info(f"Routing to EMAIL: Query contains email keywords {found_keywords}")
        return True
    
    logger.info("Routing to DOCUMENT: No email indicators found")
    return False
