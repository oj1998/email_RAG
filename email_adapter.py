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

# Import our intent detection system
from email_output.email_intent import EmailIntentDetector, EmailIntent
from email_output.targeted_email_answer import TargetedEmailAnswerGenerator

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

async def generate_intent_based_answer(query: str, intent: EmailIntent, emails: List[Dict], llm) -> str:
    """Generate an overall answer based on intent and retrieved emails"""
    
    if not emails:
        return "I couldn't find any emails matching your query."
    
    # Create a prompt based on the intent
    email_count = len(emails)
    
    intent_prompts = {
        EmailIntent.SEARCH: f"I searched for emails matching '{query}' and found {email_count} relevant results. Here's what I found:",
        EmailIntent.SUMMARIZE: f"Based on your request, I found {email_count} relevant emails. Here's a summary of the key information:",
        EmailIntent.LIST: f"Here's a list of {email_count} emails that match your query:",
        EmailIntent.EXTRACT: f"I've extracted information from {email_count} relevant emails based on your query:",
        EmailIntent.ANALYZE: f"After analyzing {email_count} relevant emails, here's what I found:",
        EmailIntent.COUNT: f"I found {email_count} emails matching your criteria.",
        EmailIntent.FORWARD: f"I found {email_count} emails that you might want to forward:",
        EmailIntent.ORGANIZE: f"I found {email_count} emails that could be organized:",
        EmailIntent.COMPOSE: f"Based on {email_count} related emails, here's some information to help compose your message:",
        EmailIntent.CONVERSATIONAL: f"I found {email_count} emails related to your question."
    }
    
    # Get the appropriate intro based on intent, or use a default
    intro = intent_prompts.get(intent, f"I found {email_count} emails related to your query. Here they are:")
    
    return intro

async def process_email_query(query: str, conversation_id: str, context: Dict[str, Any] = None, email_filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process an email-specific query with intent detection and relevant email retrieval"""
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
        
        # Detect intent using existing intent detector
        intent_detector = EmailIntentDetector(use_embeddings=False, use_llm=True)
        intent_analysis = await intent_detector.detect_intent(query, context)
        logger.info(f"Query: '{query}'")
        logger.info(f"Detected intent: {intent_analysis.primary_intent.value} (confidence: {intent_analysis.metadata.confidence:.2f})")
        if intent_analysis.secondary_intent:
            logger.info(f"Secondary intent: {intent_analysis.secondary_intent.value}")
        logger.info(f"Intent reasoning: {intent_analysis.metadata.reasoning}")
        
        # Retrieve relevant emails (regardless of intent)
        relevant_emails = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_system.get_relevant_emails(
                query=query,
                filters=filter_options,
                k=5  # Get top 5 relevant emails
            )
        )
        
        # Generate a brief summary for each email using the LLM
        summarized_emails = []
        for email in relevant_emails:
            # Limit email content length to avoid token limits
            email_content = email.page_content[:1500] if len(email.page_content) > 1500 else email.page_content
            
            # Generate summary using existing LLM
            try:
                summary_prompt = f"Summarize this email in 1-2 concise sentences:\n\n{email_content}"
                summary_response = qa_system.llm.invoke(summary_prompt)
                summary = summary_response.content.strip() if hasattr(summary_response, 'content') else str(summary_response).strip()
            except Exception as e:
                logger.warning(f"Error generating email summary: {e}")
                summary = f"Email appears to be from {email.metadata.get('sender', 'unknown')} regarding {email.metadata.get('subject', 'unknown topic')}."
            
            # Add to summarized emails list
            summarized_emails.append({
                "id": email.metadata.get("email_id", "unknown"),
                "sender": email.metadata.get("sender", "Unknown"),
                "recipient": email.metadata.get("recipient", "Unknown Recipient"),
                "subject": email.metadata.get("subject", "Email"),
                "date": email.metadata.get("date", ""),
                "confidence": email.metadata.get("relevance_score", 0),
                "content": email.page_content,
                "summary": summary
            })
        
        # NEW: Generate a targeted answer based on the retrieved emails
        answer_generator = TargetedEmailAnswerGenerator(llm=qa_system.llm)
        targeted_answer = await answer_generator.generate_targeted_answer(
            query=query,
            email_sources=summarized_emails,
            intent=intent_analysis.primary_intent
        )
        
        # Generate a brief overall answer based on intent and retrieved emails
        overall_answer = await generate_intent_based_answer(
            query=query,
            intent=intent_analysis.primary_intent,
            emails=summarized_emails,
            llm=qa_system.llm
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare metadata
        metadata = {
            "query_type": "email",
            "processing_time": processing_time,
            "source_count": len(summarized_emails),
            "conversation_context_used": bool(context and context.get("conversation_history")),
            "filters_applied": filter_options,
            "intent": {
                "primary": intent_analysis.primary_intent.value,
                "secondary": intent_analysis.secondary_intent.value if intent_analysis.secondary_intent else None,
                "confidence": intent_analysis.metadata.confidence,
                "reasoning": intent_analysis.metadata.reasoning
            }
        }
        
        # Return comprehensive response with both the original answer and targeted answer
        return {
            "status": "success",
            "answer": overall_answer,
            "targeted_answer": targeted_answer,
            "emails": summarized_emails,
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
                "intent_detection_available": True,
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
