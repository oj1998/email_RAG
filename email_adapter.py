from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import os
import logging
from dotenv import load_dotenv
import asyncio

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from client.gmail_client import EmailSearchOptions
from loaders.vector_store_loader import VectorStoreFactory
from retrievers.email_retriever import EmailQASystem, EmailFilterOptions

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
EMAIL_QA_SYSTEM = None

# Pydantic models for email filtering
class EmailFilter(BaseModel):
    after_date: Optional[str] = Field(None, description="Emails after this date (YYYY/MM/DD)")
    before_date: Optional[str] = Field(None, description="Emails before this date (YYYY/MM/DD)")
    from_email: Optional[str] = Field(None, description="Sender email address")
    to_email: Optional[str] = Field(None, description="Recipient email address")
    subject_contains: Optional[str] = Field(None, description="Subject contains these words")
    has_attachment: Optional[bool] = Field(None, description="Email has attachments")
    label: Optional[str] = Field(None, description="Email has this Gmail label")

class EmailQueryResponse(BaseModel):
    status: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

async def get_email_qa_system():
    """Get or initialize the email QA system"""
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
            raise HTTPException(status_code=500, detail=f"Email system initialization failed: {str(e)}")
    
    return EMAIL_QA_SYSTEM

async def process_email_query(
    query: str,
    conversation_id: str,
    context: Dict[str, Any],
    email_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a query specifically for email data
    
    Args:
        query: User's question
        conversation_id: ID of the conversation
        context: Context information
        email_filters: Optional filters for email retrieval
        
    Returns:
        Response with answer and metadata
    """
    try:
        # Get email QA system
        qa_system = await get_email_qa_system()
        
        # Extract context-based filters if appropriate
        if not email_filters and context:
            # Try to extract time-based filters from context
            email_filters = {}
            if "timeframe" in context:
                timeframe = context.get("timeframe", "").lower()
                if "yesterday" in timeframe:
                    from datetime import datetime, timedelta
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y/%m/%d")
                    email_filters["after_date"] = yesterday
                    email_filters["before_date"] = datetime.now().strftime("%Y/%m/%d")
                elif "last week" in timeframe:
                    from datetime import datetime, timedelta
                    last_week = (datetime.now() - timedelta(days=7)).strftime("%Y/%m/%d")
                    email_filters["after_date"] = last_week
                # Add more timeframe logic as needed
        
        # Apply any provided filters
        filters = EmailFilterOptions(**email_filters) if email_filters else None
        
        # Get answer from email QA system
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_system.query(question=query, filters=email_filters)
        )
        
        # Get source emails for reference
        relevant_emails = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_system.get_relevant_emails(query=query, filters=email_filters, k=3)
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
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error in email query processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Email query processing failed: {str(e)}")

def create_email_router():
    """Create and configure the email router"""
    router = APIRouter()
    
    @router.post("/query-emails", response_model=EmailQueryResponse)
    async def query_emails(request: dict):
        """Query emails endpoint"""
        try:
            query = request.get("query")
            conversation_id = request.get("conversation_id")
            context = request.get("context", {})
            email_filters = request.get("email_filters")
            
            if not query or not conversation_id:
                raise HTTPException(status_code=400, detail="Missing required fields: query and conversation_id")
            
            response = await process_email_query(
                query=query,
                conversation_id=conversation_id,
                context=context,
                email_filters=email_filters
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in query_emails endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return router

def is_email_query(request: Dict[str, Any]) -> bool:
    """
    Determine if the query should be routed to email system
    based on info_sources or explicit flags
    """
    # Check if request explicitly targets emails
    if request.get("target_system") == "email":
        return True
        
    # Check if emails are in the conversation's info_sources
    if "info_sources" in request:
        sources = request.get("info_sources", [])
        email_sources = ["emails", "email", "mail", "gmail"]
        if any(source.lower() in email_sources for source in sources):
            return True
            
    # Check context for email-specific indicators
    context = request.get("context", {})
    query = request.get("query", "").lower()
    
    email_keywords = ["email", "mail", "gmail", "inbox", "message", "received", "sent"]
    if any(keyword in query for keyword in email_keywords):
        return True
        
    return False

async def route_query(request: Dict[str, Any]):
    """
    Route a query to the appropriate system based on the request content
    
    This function acts as a facade that routes requests to either the
    email system or the default document system.
    """
    if is_email_query(request):
        # Route to email system
        email_filters = request.get("email_filters")
        return await process_email_query(
            query=request.get("query"),
            conversation_id=request.get("conversation_id"),
            context=request.get("context", {}),
            email_filters=email_filters
        )
    else:
        # This would call your existing query_documents function
        # Assuming it's exposed as a callable function
        from main import query_documents_internal
        return await query_documents_internal(request)
