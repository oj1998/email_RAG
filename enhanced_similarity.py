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

class RateLimitedEmbeddings:
    """Wrapper class for embeddings operations"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
    
    async def embed_query(self, text):
        """Async version of embed_query"""
        import asyncio
        return await asyncio.to_thread(self.embeddings_model.embed_query, text)
    
    async def embed_documents(self, texts):
        """Async version of embed_documents"""
        import asyncio
        return await asyncio.to_thread(self.embeddings_model.embed_documents, texts)

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
        
        # Initialize embeddings model
        if embeddings_model:
            self.embeddings_model = embeddings_model
        else:
            # Create default embeddings model
            self.embeddings_model = OpenAIEmbeddings()
        
        # Set up prompts
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
        k: int = 3  # Reduced from 5 to 3 by default
    ) -> List[Document]:
        """Get relevant documents with reduced count to limit API calls
        
        Args:
            query: The user's question
            classification: The question category classification
            metadata_filter: Optional dict of metadata filters
            k: Number of documents to retrieve (reduced default)
            
        Returns:
            List[Document]: Relevant documents
        """
        # Adjust search parameters based on classification - use more conservative values
        search_k = min(k + 2, 5) if classification.category == "SAFETY" else k
        
        # Set up search kwargs
        search_kwargs = {
            "k": search_k,
            "fetch_k": min(search_k * 2, 10)  # More conservative fetch_k
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        
        # Get documents with scores
        try:
            logger.info(f"Retrieving documents for query (k={search_k})")
            
            # Use asyncio.to_thread for better async handling
            try:
                docs_with_scores = await asyncio.to_thread(
                    self.vector_store.similarity_search_with_score,
                    query,
                    **search_kwargs
                )
            except Exception as e:
                logger.warning(f"similarity_search_with_score failed: {str(e)}, falling back to basic retrieval")
                docs = await asyncio.to_thread(
                    self.vector_store.similarity_search,
                    query,
                    k=search_k
                )
                # Create dummy scores
                docs_with_scores = [(doc, 0.75) for doc in docs]
            
            # Add scores to document metadata
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata['similarity'] = float(score)
                docs.append(doc)
                
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            # Return empty list on failure rather than crashing
            return []

    async def _batch_embed_documents(self, texts, batch_size=100):
        """Batch process document embeddings with optimal batch size
        
        Args:
            texts: List of text strings to embed
            batch_size: Size of batches to process (default 100)
            
        Returns:
            List of embeddings for all texts
        """
        if not texts:
            return []
        
        # Group texts into sensible batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        all_embeddings = []
        
        for i, batch in enumerate(batches):
            try:
                logger.info(f"Embedding batch {i+1}/{len(batches)} with {len(batch)} texts")
                # Use asyncio.to_thread for better async handling
                batch_embeddings = await asyncio.to_thread(
                    self.embeddings_model.embed_documents,
                    batch
                )
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Successfully embedded batch {i+1}")
            except Exception as e:
                logger.error(f"Error embedding batch {i+1}: {str(e)}")
                # Generate fallback embeddings for failed batch
                import numpy as np
                fallback_embeddings = [np.random.normal(0, 0.01, 1536).tolist() for _ in range(len(batch))]
                all_embeddings.extend(fallback_embeddings)
                logger.warning(f"Using fallback embeddings for batch {i+1}")
        
        return all_embeddings

    async def _extract_attributions(
        self,
        response: str,
        documents: List[Document],
        classification: QuestionType
    ) -> List[SourceAttribution]:
        """Extract source attributions with improved batching"""
        # Start timing for performance monitoring
        start_time = datetime.utcnow()
        
        # No documents means no attributions
        if not documents:
            return []
            
        attributions = []
        
        try:
            # Get response embedding only once
            response_embedding = await asyncio.to_thread(
                self.embeddings_model.embed_query, 
                response
            )
            
            # Extract all document contents first to enable batching
            doc_contents = []
            valid_docs = []
            
            for doc in documents:
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    doc_contents.append(doc.page_content)
                    valid_docs.append(doc)
            
            # Skip processing if no valid documents
            if not doc_contents:
                return []
                
            # Use batched embeddings
            logger.info(f"Batching embeddings for {len(doc_contents)} documents")
            doc_embeddings = await self._batch_embed_documents(doc_contents)
            
            # Process each document for attribution
            for i, doc in enumerate(valid_docs):
                if i >= len(doc_embeddings):
                    continue
                    
                content = doc.page_content
                metadata = doc.metadata or {}
                
                # Use the precomputed embedding
                doc_embedding = doc_embeddings[i]
                
                # Calculate semantic similarity using cosine similarity
                semantic_similarity = np.dot(response_embedding, doc_embedding)
                semantic_similarity = (semantic_similarity + 1) / 2  # Convert to 0-1 range
                
                # Calculate simple content overlap for a cross-check
                content_overlap = self._calculate_content_overlap(response, content)
                
                # Compute weighted final confidence
                semantic_weight = 0.7
                overlap_weight = 0.3
                
                confidence = (
                    semantic_weight * semantic_similarity +
                    overlap_weight * content_overlap
                )
                
                # Skip if below threshold
                if confidence < self.min_confidence:
                    continue
                    
                # Create attribution with enhanced details
                attributions.append(
                    EnhancedSourceAttribution(
                        content=content[:200] + "..." if len(content) > 200 else content,
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
            logger.info(f"Attribution extraction took {processing_time:.2f}ms")
            
            return attributions
            
        except Exception as e:
            logger.error(f"Error in attribution extraction: {str(e)}")
            
            # Simple fallback if extraction fails
            fallback_attributions = []
            for doc in documents[:3]:  # Limit to first 3 docs to reduce processing
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    metadata = doc.metadata or {}
                    fallback_attributions.append(
                        SourceAttribution(
                            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            source_id=metadata.get('document_id', metadata.get('source_id')),
                            page_number=metadata.get('page'),
                            confidence=0.7
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
