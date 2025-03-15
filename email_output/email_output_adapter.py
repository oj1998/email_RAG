"""
Integration module to connect email intent detection and formatting to the existing system.
"""
from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime

# Import our components
from .email_intent import EmailIntentDetector, EmailIntent
from .email_formatter import EmailFormatter, EmailSource, FormatStyle

logger = logging.getLogger(__name__)

class EmailOutputManager:
    """
    Manages the intent-based formatting of email query responses.
    
    This class serves as an adapter between the existing email processing
    system and our new intent-based formatting system.
    """
    
    def __init__(self):
        """Initialize the manager with intent detector and formatter"""
        self.intent_detector = EmailIntentDetector(use_embeddings=False, use_llm=True)
        self.formatter = EmailFormatter()
        
    async def process_response(
        self,
        query: str,
        raw_response: str,
        sources: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an email query response with intent-based formatting
        
        Args:
            query: The original user query
            raw_response: The raw response from the RAG system
            sources: List of source emails used for the response
            context: Additional context about the conversation
            metadata: Additional metadata about the query/response
            
        Returns:
            Formatted response with intent metadata
        """
        try:
            # Default context if not provided
            if context is None:
                context = {}
                
            # Default metadata if not provided
            if metadata is None:
                metadata = {}
                
            # 1. Detect the intent of the query
            start_time = datetime.now()
            intent_analysis = await self.intent_detector.detect_intent(query, context)
            intent_detection_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Detected intent for query '{query}': {intent_analysis.primary_intent.value} " +
                       f"(confidence: {intent_analysis.metadata.confidence:.2f})")
            
            # 2. Format the response based on the detected intent
            start_time = datetime.now()
            formatted_response = await self.formatter.format_response(
                content=raw_response,
                intent=intent_analysis.primary_intent,
                sources=sources,
                metadata={
                    **metadata,
                    "intent": intent_analysis.primary_intent.value,
                    "confidence": intent_analysis.metadata.confidence
                }
            )
            formatting_time = (datetime.now() - start_time).total_seconds()
            
            # 3. Prepare the output in the format expected by the existing system
            response = {
                "status": "success",
                "answer": formatted_response.content,  # The formatted content becomes the answer
                "sources": sources,  # Keep original sources
                "metadata": {
                    "query_type": "email",
                    "intent": {
                        "primary": intent_analysis.primary_intent.value,
                        "secondary": intent_analysis.secondary_intent.value if intent_analysis.secondary_intent else None,
                        "confidence": intent_analysis.metadata.confidence,
                        "reasoning": intent_analysis.metadata.reasoning
                    },
                    "formatting": {
                        "style": self.formatter.mappings[intent_analysis.primary_intent].primary_style.value,
                        "processing_time": formatting_time
                    },
                    "processing_times": {
                        "intent_detection": intent_detection_time,
                        "formatting": formatting_time,
                        "total": intent_detection_time + formatting_time
                    },
                    **metadata  # Include original metadata
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing email response: {str(e)}", exc_info=True)
            # Return fallback response
            return {
                "status": "partial",
                "answer": raw_response,  # Use original response
                "sources": sources,
                "metadata": {
                    "query_type": "email",
                    "error": str(e),
                    **metadata
                }
            }


# Function to use in email_adapter.py to replace or enhance the current process_email_query
async def enhanced_process_email_query(
    query: str,
    conversation_id: str,
    context: Dict[str, Any] = None,
    email_filters: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced version of process_email_query that uses intent-based formatting
    
    This function can be used as a drop-in replacement for the existing
    process_email_query function in email_adapter.py
    """
    try:
        # Import here to avoid circular imports
        from email_adapter import get_email_qa_system
        
        # Get email QA system
        qa_system = await get_email_qa_system()
        
        # Process filters
        filter_options = {}
        if email_filters:
            filter_options = {k: v for k, v in email_filters.items() if v is not None}
            
        # Get raw answer from the QA system
        raw_answer = await asyncio.get_event_loop().run_in_executor(
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
        
        # Format sources for processing
        sources = [
            {
                "id": email.metadata.get("email_id", "unknown"),
                "sender": email.metadata.get("sender", "Unknown"),
                "recipient": email.metadata.get("recipient", "Unknown"),
                "subject": email.metadata.get("subject", "Email"),
                "date": email.metadata.get("date", ""),
                "confidence": email.metadata.get("relevance_score", 0),
                "excerpt": email.page_content[:200] + "..." if len(email.page_content) > 200 else email.page_content
            }
            for email in relevant_emails
        ]
        
        # Metadata for the query
        metadata = {
            "query_type": "email",
            "filters_applied": filter_options,
            "source_count": len(sources),
            "conversation_id": conversation_id,
            "conversation_context_used": bool(context and context.get("conversation_history"))
        }
        
        # Process with intent-based formatting
        output_manager = EmailOutputManager()
        formatted_response = await output_manager.process_response(
            query=query,
            raw_response=raw_answer,
            sources=sources,
            context=context,
            metadata=metadata
        )
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in enhanced email query processing: {str(e)}", exc_info=True)
        # Return a basic error response
        return {
            "status": "error",
            "answer": f"I encountered an error processing your email query: {str(e)}",
            "sources": [],
            "metadata": {
                "error": str(e),
                "query_type": "email"
            }
        }


# Simple test for the integration
async def test_integration():
    """Test the integrated email processing with a sample query"""
    # This is a mock test since we can't import the actual get_email_qa_system
    
    # Mock query and context
    query = "Find emails from John about the project deadline"
    context = {
        "timeframe": "last week",
        "conversation_history": [
            {"role": "user", "content": "Show me my recent emails"},
            {"role": "assistant", "content": "I found 10 recent emails in your inbox."}
        ]
    }
    
    # Mock raw answer
    raw_answer = """
    I found 3 emails from John Smith about the project deadline.
    The most recent email was sent yesterday at 5:30 PM with the subject "Final Project Deadline".
    John mentioned that the deadline has been extended to May 15th due to additional requirements.
    There was also an email from John on Monday discussing the timeline adjustments.
    """
    
    # Mock sources
    sources = [
        {
            "id": "email123",
            "sender": "john.smith@example.com",
            "subject": "Final Project Deadline",
            "date": "2023-05-01",
            "confidence": 0.95,
            "excerpt": "The deadline has been extended to May 15th due to additional requirements."
        },
        {
            "id": "email124",
            "sender": "john.smith@example.com",
            "subject": "Timeline Adjustments",
            "date": "2023-04-28",
            "confidence": 0.82,
            "excerpt": "We need to adjust the timeline for the following deliverables..."
        }
    ]
    
    # Process with our output manager
    output_manager = EmailOutputManager()
    result = await output_manager.process_response(
        query=query,
        raw_response=raw_answer,
        sources=sources,
        context=context
    )
    
    print("*** PROCESSED RESPONSE ***")
    print(f"Intent: {result['metadata']['intent']['primary']}")
    print(f"Confidence: {result['metadata']['intent']['confidence']}")
    print("\nFormatted Answer:")
    print(result['answer'])
    print("\nMetadata:")
    import json
    print(json.dumps(result['metadata'], indent=2))

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_integration())
