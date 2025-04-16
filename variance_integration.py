from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
import re
import json

from fastapi import HTTPException
from pydantic import BaseModel

# These imports would come from your existing codebase
from construction_classifier import QuestionType
from query_intent import QueryIntent, IntentAnalysis

# Import variance analysis components
from variance_analysis.document_variance_analyzer import DocumentVarianceAnalyzer, TopicVarianceAnalysis
from variance_analysis.variance_formatter import VarianceFormatter, VarianceFormatStyle
from variance_analysis.variance_retriever import VarianceDocumentRetriever
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# This class assumes it will be available from bubble_backend.py
class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    context: Optional[Dict[str, Any]] = None
    document_ids: Optional[List[str]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    email_filters: Optional[Dict[str, Any]] = None
    info_sources: Optional[List[str]] = None

# Function to check if a query is about variance analysis
def is_variance_analysis_query(question: str) -> bool:
    """Detect if a query is explicitly asking about differences between sources."""
    strict_pattern = r"(differences|discrepancies|variations|conflicts) between (sources|documents|references)"
    return bool(re.search(strict_pattern, question.lower()))

def get_hardcoded_custom_response(query: str) -> Optional[Dict[str, Any]]:
    """Check if query matches a hardcoded response that needs custom rendering."""
    # Normalize query for comparison
    normalized_query = query.strip().lower()
    
    # Define your hardcoded queries and responses
    hardcoded_responses = {
        "show me climate data variance": {
            "content": "# Climate Impact Analysis\n\n## Temperature Variations\n\nGlobal measurements show significant variations...",
            "render_type": "climate_data_visual",
            "custom_data": {
                "chart_type": "temperature_variance",
                "regions": ["North America", "Europe", "Asia"],
                "time_period": "2020-2024"
            },
            "sources": [
                {
                    "id": "doc-123",
                    "title": "Climate Report 2024",
                    "page": 42,
                    "confidence": 0.95,
                    "excerpt": "Temperature variations across regions indicate..."
                }
            ]
        }
        # Add more hardcoded responses as needed
    }
    
    # Return the matching response if found
    return hardcoded_responses.get(normalized_query)

# Main processing function for variance analysis queries
async def process_variance_query(
    request: QueryRequest,
    classification: QuestionType,
    vector_store,  # Pass this from bubble_backend.py
    conversation_context: Optional[Dict] = None,
    intent_analysis = None
) -> Dict[str, Any]:
    """Process a query asking for variance analysis between documents on the same topic"""
    start_time = datetime.utcnow()
    logger.info(f"Processing variance analysis query: {request.query}")

    # Check for custom hardcoded responses first
    hardcoded_response = get_hardcoded_custom_response(request.query)
    if hardcoded_response:
        logger.info(f"Using hardcoded custom response for query: {request.query}")
        
        return {
            "status": "success",
            "answer": hardcoded_response["content"],
            "classification": classification.dict() if hasattr(classification, 'dict') else classification,
            "sources": hardcoded_response.get("sources", []),
            "metadata": {
                "category": "CUSTOM_RENDERER_REQUIRED",
                "query_type": "special_visualization",
                "render_type": hardcoded_response["render_type"],
                "custom_renderer_data": hardcoded_response["custom_data"],
                "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                "conversation_context_used": bool(conversation_context),
                "intent_analysis": intent_analysis.dict() if hasattr(intent_analysis, 'dict') else None,
            }
        }
    
    try:
        # Initialize variance analysis components
        variance_retriever = VarianceDocumentRetriever(
            vector_store=vector_store,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0),
            min_relevance=0.65,
            max_documents=12
        )
        
        variance_analyzer = DocumentVarianceAnalyzer(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
            min_confidence=0.6
        )
        
        variance_formatter = VarianceFormatter()
        
        # Prepare metadata filter
        metadata_filter = {}
        if request.document_ids:
            metadata_filter["document_id"] = {"$in": request.document_ids}
        if request.metadata_filter:
            metadata_filter.update(request.metadata_filter)
            
        # 1. Retrieve documents for variance analysis
        logger.info("Retrieving documents for variance analysis...")
        documents = await variance_retriever.get_variance_documents(
            query=request.query,
            metadata_filter=metadata_filter
        )
        
        if len(documents) < 2:
            # Not enough documents for meaningful variance analysis
            logger.warning(f"Insufficient documents for variance analysis ({len(documents)})")
            return {
                "status": "success",
                "answer": "I couldn't find enough different sources about this topic to perform a variance analysis. " +
                          "Please try a different topic or query, or ensure there are multiple documents available " +
                          "that cover this subject.",
                "metadata": {
                    "query_type": "variance_analysis",
                    "document_count": len(documents),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "error": "insufficient_documents"
                }
            }
            
        # Optional: Merge document chunks to improve analysis quality
        logger.info("Merging document chunks...")
        merged_documents = await variance_retriever.merge_document_chunks(documents)
        
        # 2. Perform variance analysis
        logger.info("Performing variance analysis...")
        analysis_result = await variance_analyzer.analyze_topic_variances(
            query=request.query,
            documents=merged_documents if merged_documents else documents
        )
        
        # 3. Determine the best format based on classification and context
        format_style = VarianceFormatStyle.TABULAR  # Default format
        
        # Use classification to pick a better format if available
        if hasattr(classification, 'suggested_format') and classification.suggested_format:
            style = classification.suggested_format.get("style", "").lower()
            if style == "narrative":
                format_style = VarianceFormatStyle.NARRATIVE
            elif style == "tabular" or style == "table":
                format_style = VarianceFormatStyle.TABULAR
            elif style == "visual":
                format_style = VarianceFormatStyle.VISUAL
                
        # If intent analysis suggests this is a more in-depth analysis, use narrative
        if intent_analysis and hasattr(intent_analysis, 'primary_intent'):
            if intent_analysis.primary_intent == QueryIntent.INFORMATION:
                format_style = VarianceFormatStyle.NARRATIVE
                
        # 4. Format the variance analysis
        logger.info(f"Formatting variance analysis with style: {format_style}")
        formatted_analysis = variance_formatter.format_variance_analysis(
            analysis=analysis_result,
            format_style=format_style
        )
        
        # 5. Prepare sources for response
        sources = []
        for variance in analysis_result.key_variances:
            for doc_id, positions in variance.source_positions.items():
                # Find relevant document
                doc_metadata = next(
                    (doc.metadata for doc in merged_documents or documents 
                     if doc.metadata.get('document_id') == doc_id),
                    {}
                )
                
                # Add source with variance-specific information
                sources.append({
                    "id": doc_id,
                    "title": doc_metadata.get("title", "Unknown Document"),
                    "page": doc_metadata.get("page"),
                    "confidence": variance.confidence,
                    "aspect": variance.aspect,
                    "position": ", ".join(positions),
                    "excerpt": variance.source_excerpts.get(doc_id, "No excerpt available"),
                    "reliability": analysis_result.reliability_ranking.get(doc_id, 0.5)
                })
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"formatted_analysis: {formatted_analysis}")

        # Add this right before the return statement in process_variance_query
        logger.info(f"Returning variance analysis response: {json.dumps({
            'status': 'success',
            'answer_preview': formatted_analysis,
            'sources_count': len(sources),
            'metadata': {
                'category': 'VARIANCE_ANALYSIS',
                'format_used': format_style,
                'topic': analysis_result.topic,
                'variance_count': len(analysis_result.key_variances),
                'agreement_points': len(analysis_result.agreement_points)
            }
        }, default=str)}")
        
        # 6. Prepare and return complete response
        return {
            "status": "success",
            "answer": formatted_analysis,
            "classification": classification.dict() if hasattr(classification, 'dict') else classification,
            "sources": sources,
            "metadata": {
                "category": "VARIANCE_ANALYSIS",
                "confidence": getattr(classification, 'confidence', 0.8),
                "source_count": len(set(s["id"] for s in sources)),
                "processing_time": processing_time,
                "conversation_context_used": bool(conversation_context),
                "intent_analysis": intent_analysis.dict() if hasattr(intent_analysis, 'dict') else None,
                "format_used": format_style,
                "topic": analysis_result.topic,
                "variance_count": len(analysis_result.key_variances),
                "agreement_points": len(analysis_result.agreement_points)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in variance analysis: {str(e)}", exc_info=True)
        
        # Return a graceful error response
        return {
            "status": "error",
            "answer": "I encountered an issue while analyzing source variances. " + 
                      "This could be due to insufficient or incompatible information sources. " +
                      "Please try rephrasing your query or specifying a clearer topic.",
            "metadata": {
                "error": str(e),
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
        }

# Function to integrate with your bubble_backend.py routing logic
def integrate_variance_analysis():
    """
    Integrate variance analysis into the main bubble_backend.py
    
    This function shows what code to add to your bubble_backend.py
    to support variance analysis queries
    """
    # Add this to your classification logic (in construction_classifier.py)
    """
    async def classify_question(
        self, 
        question: str,
        conversation_context: Optional[Dict] = None,
        current_context: Optional[Dict] = None
    ) -> QuestionType:
        # Your existing code...
        
        # Add this check for variance analysis queries
        is_variance = await is_variance_analysis_query(question)
        
        # If variance analysis query is detected, modify classification
        if is_variance:
            if 'suggested_format' not in classification:
                classification['suggested_format'] = {}
            classification['suggested_format']['is_variance_analysis'] = True
            
            # If not already categorized as INFORMATION or COMPARISON, consider overriding
            if classification['category'] not in ['INFORMATION', 'COMPARISON'] and classification['confidence'] < 0.8:
                classification['category'] = 'INFORMATION'
                if 'reasoning' in classification:
                    classification['reasoning'] += " Detected variance analysis query patterns."
    """
    
    # Add this to your query processing logic (in bubble_backend.py)
    """
    async def process_document_query(
        request: QueryRequest,
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        # Your existing code...
        
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
        
        # Check if this is a variance analysis query
        is_variance = False
        if hasattr(classification, 'suggested_format') and classification.suggested_format:
            is_variance = classification.suggested_format.get("is_variance_analysis", False)
        
        # Route to variance analysis processing if needed
        if is_variance or is_variance_analysis_query(request.query):
            logger.info(f"Routing to variance analysis processing for query: {request.query}")
            return await process_variance_query(
                request=request,
                classification=classification,
                vector_store=vector_store,  # Pass your vector store
                conversation_context=conversation_context,
                intent_analysis=intent_analysis
            )
            
        # Rest of your existing code...
    """
