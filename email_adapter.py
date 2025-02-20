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

# Set up logging
logger = logging.getLogger(__name__)

# Global email QA system instance
EMAIL_QA_SYSTEM = None

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
            embeddings_model = HuggingFaceEmbeddings()
            
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
    """Process an email-specific query"""
    try:
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
        
        # Format sources
        sources = [
            {
                "id": email.metadata.get("email_id", "unknown"),
                "title": email.metadata.get("subject", "Email"),
                "sender": email.metadata.get("sender", "Unknown"),
                "date": email.metadata.get("date", ""),
                "confidence": email.metadata.get("relevance_score", 0)
            }
            for email in relevant_emails
        ]
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources
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
        
        return await process_email_query(
            query=request.get("query"),
            conversation_id=request.get("conversation_id"),
            context=request.get("context"),
            email_filters=request.get("email_filters")
        )
    
    @router.get("/status")
    async def email_system_status():
        """Check email system status"""
        try:
            qa_system = await get_email_qa_system()
            return {
                "status": "healthy",
                "initialized": qa_system is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    return router

def is_email_query(request_data: Dict[str, Any]) -> bool:
    """Determine if a query should be routed to the email system"""
    if request_data.get("email_filters"):
        return True
        
    info_sources = request_data.get("info_sources", [])
    if info_sources and any(
        source.lower() in ["emails", "email", "mail", "gmail"]
        for source in info_sources
    ):
        return True
        
    query = request_data.get("query", "").lower()
    email_keywords = [
        "email", "mail", "gmail", "inbox", "message",
        "received", "sent", "folder", "label"
    ]
    
    return any(keyword in query for keyword in email_keywords)
