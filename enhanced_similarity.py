from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from construction_classifier import ConstructionClassifier, QuestionType
from query_intent import QueryIntent
import logging
import numpy as np
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class SourceAttribution(BaseModel):
    """Tracks which parts of the response came from which sources"""
    content: str
    source_id: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float

class EnhancedSourceAttribution(SourceAttribution):
    """Enhanced version with additional metrics"""
    semantic_similarity: float
    content_overlap: float
    title: Optional[str] = None
    document_type: Optional[str] = None
    
class ResponseWithSources(BaseModel):
    """Complete response with source attributions"""
    text: str
    needs_sources: bool
    attributions: List[SourceAttribution] = []
    classification: Optional[QuestionType] = None
    
class EnhancedSmartResponseGenerator:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStore,
        classifier: ConstructionClassifier,
        min_confidence: float = 0.6,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.classifier = classifier
        self.min_confidence = min_confidence
        
        # Initialize embeddings model if not provided
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        
        # Rest of initialization (prompts, etc.) remains the same
        self.source_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a construction knowledge expert. Analyze if this construction-related 
            question requires specific sourced information rather than general knowledge.
            
            Consider:
            1. Is this asking about specific procedures, codes, or specifications?
            2. Could incorrect information pose safety risks?
            3. Does this require project-specific details?
            4. Is this general knowledge any construction professional would know?
            
            Output only a single float number between 0 and 1 representing the confidence 
            that this question requires specific sourced information."""),
            ("user", "Question: {query}\nCategory: {category}")
        ])
        
        self.attribution_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a response for a construction-related question using the provided documents.
            For safety-related questions, be especially thorough in citing sources.
            
            Requirements:
            1. Use information from provided documents where relevant
            2. Note which specific documents inform each key point
            3. Maintain natural, professional tone
            4. For safety information, explicitly note source reliability
            
            Format as natural text without visible citations."""),
            ("user", "Question: {query}\nCategory: {category}\nContext Documents: {context}")
        ])

    async def should_use_sources(
        self,
        query: str,
        classification: QuestionType,
        intent_analysis=None
    ) -> bool:
        """Determine if the query requires source retrieval
        
        Args:
            query: The user's question
            classification: The category classification
            intent_analysis: Optional intent analysis result (IntentAnalysis object)
            
        Returns:
            bool: Whether sources should be used for this query
        """
        # No changes to this method - original logic works well
        # Very short queries likely don't need sources
        if len(query.strip()) < 15:
            return False
        
        # Intent-based determination (if intent_analysis is provided)
        if intent_analysis and hasattr(intent_analysis, 'primary_intent'):
            # These intents don't need sources
            no_source_intents = [
                QueryIntent.GREETING,
                QueryIntent.SMALL_TALK, 
                QueryIntent.ACKNOWLEDGMENT,
                QueryIntent.CLARIFICATION
            ]
            
            if intent_analysis.primary_intent in no_source_intents:
                return False
                
            # Emergency intent always needs sources for safety reasons
            if intent_analysis.primary_intent == QueryIntent.EMERGENCY:
                return True
        # Fallback to category-based determination (original approach)
        else:
            # Categories that don't need sources
            no_source_categories = ["GREETING", "CLARIFICATION", "SMALL_TALK", "ACKNOWLEDGMENT"]
            if classification.category in no_source_categories:
                return False
        
        # For everything else, use a simple check for very basic responses
        if len(query.split()) <= 5:  # Queries with 5 or fewer words
            response = await self.llm.ainvoke(
                ChatPromptTemplate.from_messages([
                    ("system", """Determine if this is a simple question requiring only a yes/no or very basic response.
                    Output ONLY 'BASIC' or 'COMPLEX'."""),
                    ("user", query)
                ]).format_messages()
            )
            
            if "BASIC" in response.content.upper():
                return False
        
        # Default to using sources for everything else
        return True

    async def _get_relevant_documents(
        self,
        query: str,
        classification: QuestionType,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Document]:
        """Get relevant documents with enhanced retrieval
        
        Args:
            query: The user's question
            classification: The question category classification
            metadata_filter: Optional dict of metadata filters
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: Relevant documents
        """
        # Adjust search parameters based on classification
        search_k = k * 2 if classification.category == "SAFETY" else k
        min_similarity = 0.7 if classification.category == "SAFETY" else self.min_confidence
        
        # Set up search kwargs
        search_kwargs = {
            "k": search_k,  # Get more results than needed for filtering
            "fetch_k": search_k * 2,  # Consider more candidates
            "lambda_mult": 0.5  # Balance between relevance and diversity
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        
        # Create a retriever with the specified parameters
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # Add a relevance filter to ensure minimum similarity
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings_model,
            similarity_threshold=min_similarity
        )
        
        filtered_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=embeddings_filter
        )
        
        # Using run_in_executor because LangChain's retriever is not async
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: filtered_retriever.get_relevant_documents(query)
            )
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            # Fall back to basic retriever if compression fails
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retriever.get_relevant_documents(query)
            )
            logger.info(f"Fall back retrieval: {len(docs)} documents")
            return docs

    async def _extract_attributions(
        self,
        response: str,
        documents: List[Document],
        classification: QuestionType
    ) -> List[SourceAttribution]:
        """Extract source attributions using enhanced vector-based similarity
        
        Args:
            response: The generated response text
            documents: List of source documents
            classification: The question classification
            
        Returns:
            List[SourceAttribution]: Source attributions with confidence scores
        """
        # Start timing for performance monitoring
        start_time = datetime.utcnow()
        
        # No documents means no attributions
        if not documents:
            return []
            
        attributions = []
        
        try:
            # Get response embedding
            response_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embeddings_model.embed_query(response)
            )
            
            # Process each document for attribution
            for doc in documents:
                # Skip empty or invalid documents
                if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                    continue
                
                content = doc.page_content
                metadata = doc.metadata or {}
                
                # Get document embedding for semantic similarity
                doc_embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.embeddings_model.embed_query(content)
                )
                
                # Calculate semantic similarity using cosine similarity
                semantic_similarity = np.dot(response_embedding, doc_embedding)
                # Convert from [-1,1] to [0,1] range
                semantic_similarity = (semantic_similarity + 1) / 2
                
                # Calculate content overlap (simpler word-based similarity as a cross-check)
                content_overlap = self._calculate_content_overlap(response, content)
                
                # Compute weighted final confidence
                # More weight to semantic similarity, but consider content overlap
                semantic_weight = 0.7
                overlap_weight = 0.3
                
                # Adjust weights for safety content
                if classification.category == "SAFETY":
                    semantic_weight = 0.6
                    overlap_weight = 0.4  # Exact matches more important for safety
                
                # Calculate combined confidence score
                confidence = (
                    semantic_weight * semantic_similarity +
                    overlap_weight * content_overlap
                )
                
                # Apply category-based adjustments
                if classification.category == "SAFETY":
                    # Boost safety documents
                    confidence = min(0.98, confidence * 1.15)
                
                # Skip if below threshold
                if confidence < self.min_confidence:
                    continue
                    
                # Find the most relevant excerpt from the document
                excerpt = self._extract_best_excerpt(content, response)
                
                # Create attribution with enhanced details
                attributions.append(
                    EnhancedSourceAttribution(
                        content=excerpt,
                        source_id=metadata.get('document_id', metadata.get('source_id')),
                        page_number=metadata.get('page'),
                        confidence=round(float(confidence), 4),
                        semantic_similarity=round(float(semantic_similarity), 4),
                        content_overlap=round(float(content_overlap), 4),
                        title=metadata.get('title'),
                        document_type=metadata.get('document_type')
                    )
                )
                
            # Sort by confidence and limit results
            attributions = sorted(attributions, key=lambda x: x.confidence, reverse=True)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(f"Attribution extraction took {processing_time:.2f}ms")
            
            return attributions
            
        except Exception as e:
            logger.error(f"Error in attribution extraction: {str(e)}")
            
            # Fall back to simple attribution if vector-based fails
            fallback_attributions = []
            for doc in documents:
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    metadata = doc.metadata or {}
                    fallback_attributions.append(
                        SourceAttribution(
                            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            source_id=metadata.get('document_id', metadata.get('source_id')),
                            page_number=metadata.get('page'),
                            confidence=0.8 if classification.category == "SAFETY" else 0.7
                        )
                    )
            return fallback_attributions

    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """Calculate content overlap using improved word-based similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Clean and tokenize texts
        words1 = set(w.lower() for w in text1.split() if len(w) > 3)
        words2 = set(w.lower() for w in text2.split() if len(w) > 3)
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity with length weighting
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Basic Jaccard similarity
        basic_similarity = len(intersection) / len(union)
        
        # Weight by key term overlap (longer words often more important)
        key_terms1 = set(w for w in words1 if len(w) > 6)
        key_terms2 = set(w for w in words2 if len(w) > 6)
        
        if key_terms1 and key_terms2:
            key_intersection = key_terms1.intersection(key_terms2)
            key_union = key_terms1.union(key_terms2)
            key_similarity = len(key_intersection) / len(key_union)
            
            # Weight key term similarity higher
            return (0.4 * basic_similarity) + (0.6 * key_similarity)
        
        return basic_similarity
        
    def _extract_best_excerpt(self, document_content: str, response: str, max_length: int = 220) -> str:
        """Extract the most relevant excerpt from a document
        
        Args:
            document_content: Full document content
            response: Generated response text
            max_length: Maximum length of excerpt
            
        Returns:
            str: Most relevant excerpt
        """
        # Split into paragraphs (more meaningful than sentences)
        paragraphs = [p.strip() for p in document_content.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in document_content.split('\n') if p.strip()]
        if not paragraphs:
            paragraphs = [document_content]
            
        # Fall back to simple excerpt if only one paragraph
        if len(paragraphs) == 1:
            content = paragraphs[0]
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content
            
        # Calculate similarity for each paragraph
        response_words = set(w.lower() for w in response.split() if len(w) > 3)
        
        para_scores = []
        for i, para in enumerate(paragraphs):
            para_words = set(w.lower() for w in para.split() if len(w) > 3)
            if not para_words:
                continue
                
            # Calculate overlap
            intersection = para_words.intersection(response_words)
            similarity = len(intersection) / len(para_words)
            
            # Add position bias (earlier paragraphs often contain key info)
            position_weight = 1.0 - (i / (len(paragraphs) * 2))  # Range from 1.0 to 0.5
            
            # Calculate final score
            score = (similarity * 0.8) + (position_weight * 0.2)
            
            para_scores.append((score, para))
            
        # Sort by score and combine best paragraphs up to max length
        if not para_scores:
            # Fall back to first part of document
            return document_content[:max_length] + "..."
            
        para_scores.sort(reverse=True)
        
        # Take best paragraph
        best_para = para_scores[0][1]
        if len(best_para) > max_length:
            return best_para[:max_length] + "..."
        return best_para
