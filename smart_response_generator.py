from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from construction_classifier import ConstructionClassifier, QuestionType
from query_intent import QueryIntent
import logging

logger = logging.getLogger(__name__)

class SourceAttribution(BaseModel):
    """Tracks which parts of the response came from which sources"""
    content: str
    source_id: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float

class ResponseWithSources(BaseModel):
    """Complete response with source attributions"""
    text: str
    needs_sources: bool
    attributions: List[SourceAttribution] = []
    classification: Optional[QuestionType] = None
    
class SmartResponseGenerator:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStore,
        classifier: ConstructionClassifier
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.classifier = classifier
        
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

    async def generate_response(
        self,
        query: str,
        classification: QuestionType
    ) -> ResponseWithSources:
        """Generate a response, retrieving sources only if needed"""
        
        needs_sources = await self.should_use_sources(query, classification)
            
        if not needs_sources:
            # Generate simple response without sources
            response = await self._generate_basic_response(query, classification)
            return ResponseWithSources(
                text=response,
                needs_sources=False,
                classification=classification
            )
            
        # If we need sources, retrieve relevant documents
        documents = await self._get_relevant_documents(query, classification)
        
        if not documents:
            # Fallback to basic response if no relevant docs found
            logger.warning("No relevant documents found, falling back to basic response")
            response = await self._generate_basic_response(query, classification)
            return ResponseWithSources(
                text=response,
                needs_sources=False,
                classification=classification
            )
            
        # Generate response with source attribution
        response, attributions = await self._generate_attributed_response(
            query,
            classification,
            documents
        )
        
        return ResponseWithSources(
            text=response,
            needs_sources=True,
            attributions=attributions,
            classification=classification
        )

    async def _get_relevant_documents(
        self,
        query: str,
        classification: QuestionType
    ) -> List[Document]:
        """Get relevant documents with classification-specific settings"""
        search_kwargs = {
            "k": 2 if classification.category == "SAFETY" else 2,
            "score_threshold": 0.7 if classification.category == "SAFETY" else 0.5
        }
        
        return await self.vector_store.asimilar_search(
            query,
            **search_kwargs
        )

    async def _generate_basic_response(
        self,
        query: str,
        classification: QuestionType
    ) -> str:
        """Generate a response without sources"""
        messages = [
            ("system", f"""You are a knowledgeable construction assistant.
            Provide a natural, professional response for a {classification.category} question.
            Use general construction knowledge without referencing specific sources."""),
            ("user", query)
        ]
        
        response = await self.llm.ainvoke(
            ChatPromptTemplate.from_messages(messages).format_messages()
        )
        return response.content

    async def _generate_attributed_response(
        self,
        query: str,
        classification: QuestionType,
        documents: List[Document]
    ) -> Tuple[str, List[SourceAttribution]]:
        """Generate a response with source attribution"""
        
        # Format context for the prompt
        context = "\n\n".join(
            f"Document {i+1} ({doc.metadata.get('source_id', 'Unknown')} - "
            f"Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}"
            for i, doc in enumerate(documents)
        )
        
        # Generate response
        response = await self.llm.ainvoke(
            self.attribution_prompt.format_messages(
                query=query,
                category=classification.category,
                context=context
            )
        )
        
        # Extract attributions and clean response
        attributions = await self._extract_attributions(
            response.content,
            documents,
            classification
        )
        
        clean_response = self._clean_response_text(response.content)
        
        return clean_response, attributions

    async def _extract_attributions(
        self,
        response: str,
        documents: List[Document],
        classification: QuestionType
    ) -> List[SourceAttribution]:
        """Extract source attributions from response with smart matching"""
        attributions = []
        
        # For each document, check if its key information was used
        for doc in documents:
            # Split into sentences for more granular matching
            doc_sentences = doc.page_content.split('.')
            response_sentences = response.lower().split('.')
            
            matches = []
            for doc_sent in doc_sentences:
                doc_sent = doc_sent.lower().strip()
                if doc_sent and any(
                    self._calculate_similarity(doc_sent, resp_sent) > 0.7
                    for resp_sent in response_sentences
                ):
                    matches.append(doc_sent)
            
            if matches:
                attributions.append(
                    SourceAttribution(
                        content=". ".join(matches),
                        source_id=doc.metadata.get('source_id'),
                        page_number=doc.metadata.get('page'),
                        confidence=0.9 if classification.category == "SAFETY" else 0.8
                    )
                )
        
        return attributions

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity check"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _clean_response_text(self, text: str) -> str:
        """Clean response text of explicit source references"""
        import re
        
        # Remove explicit source references
        cleaned = re.sub(r"According to document \d+[,:]?\s*", "", text)
        cleaned = re.sub(r"From page \d+[,:]?\s*", "", cleaned)
        cleaned = re.sub(r"\(Source: [^)]+\)", "", cleaned)
        
        return cleaned.strip()
