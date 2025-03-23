from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
import re
import logging
import asyncio

logger = logging.getLogger(__name__)

class ComparisonRetriever:
    """Retriever specialized for comparison queries between multiple topics"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        min_confidence: float = 0.6
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.min_confidence = min_confidence
        
    async def extract_comparison_topics(self, query: str) -> List[str]:
        """Extract topics being compared from the query"""
        # Try to identify explicit comparison patterns
        explicit_patterns = [
            r"(?:compare|difference between|similarities between)\s+(.+?)\s+and\s+(.+?)(?:\s|$|\.|\?)",
            r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s|$|\.|\?)",
            r"(?:which is better|which should I use)(?::|,)?\s+(.+?)\s+or\s+(.+?)(?:\s|$|\.|\?)"
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        # If no explicit comparison, try to extract key topics
        # Use LLM to extract comparison topics
        try:
            extraction_prompt = f"""
            Extract the main topics being compared in this query. Return just a Python list of strings.
            Query: {query}
            
            For example:
            Query: "Compare steel beams vs wood beams for residential construction"
            Output: ["steel beams", "wood beams"]
            """
            
            response = await self.llm.ainvoke(extraction_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Try to extract a Python list
            list_match = re.search(r'\[.*?\]', content)
            if list_match:
                # Safely evaluate the list string
                topics_list = eval(list_match.group(0))
                if isinstance(topics_list, list) and len(topics_list) >= 2:
                    return topics_list[:2]  # Take up to first two topics
        except Exception as e:
            logger.warning(f"Error extracting comparison topics: {e}")
        
        # Fallback: simple keyword extraction
        # Just extract nouns and noun phrases
        words = query.lower().split()
        return [w for w in words if len(w) > 4 and not w.startswith(('compar', 'differ', 'better', 'worse'))][:2]
    
    async def retrieve_comparison_documents(
        self, 
        query: str,
        topics: List[str], 
        metadata_filter: Optional[Dict[str, Any]] = None,
        k_per_topic: int = 3
    ) -> Dict[str, List[Document]]:
        """Retrieve documents relevant to each comparison topic"""
        topic_documents = {}
        
        for topic in topics:
            # Create a topic-specific query
            topic_query = f"{topic} {query}"
            
            # Set up search kwargs
            search_kwargs = {"k": k_per_topic}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
            
            # Retrieve documents for this topic
            try:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.vector_store.similarity_search(topic_query, **search_kwargs)
                )
                topic_documents[topic] = docs
            except Exception as e:
                logger.error(f"Error retrieving documents for topic '{topic}': {e}")
                topic_documents[topic] = []
            
        return topic_documents
    
    async def retrieve_for_comparison(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k_per_topic: int = 3
    ) -> Tuple[List[str], Dict[str, List[Document]]]:
        """Main method to retrieve documents for comparison"""
        # Extract topics from query
        topics = await self.extract_comparison_topics(query)
        
        if len(topics) < 2:
            # If we couldn't extract at least 2 topics, try a different approach
            logger.warning(f"Could only extract {len(topics)} topics from query: {query}")
            # Try to get more topics by asking the LLM directly
            try:
                extraction_prompt = f"""
                I need to compare different options in this construction question.
                Extract exactly two main topics/options being compared.
                Return only a Python list of two strings.
                
                Question: {query}
                """
                
                response = await self.llm.ainvoke(extraction_prompt)
                content = response.content if hasattr(response, "content") else str(response)
                
                # Try to extract a Python list
                list_match = re.search(r'\[.*?\]', content)
                if list_match:
                    new_topics = eval(list_match.group(0))
                    if isinstance(new_topics, list) and len(new_topics) >= 2:
                        topics = new_topics[:2]
            except Exception as e:
                logger.warning(f"Error getting additional topics: {e}")
        
        # If we still don't have enough topics, get generic documents
        if len(topics) < 2:
            logger.warning(f"Falling back to generic document retrieval for query: {query}")
            # Retrieve documents for the whole query
            search_kwargs = {"k": k_per_topic * 2}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
                
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_store.similarity_search(query, **search_kwargs)
            )
            
            return ["Option 1", "Option 2"], {"Generic": docs}
        
        # Get documents for each topic
        topic_documents = await self.retrieve_comparison_documents(
            query=query,
            topics=topics,
            metadata_filter=metadata_filter,
            k_per_topic=k_per_topic
        )
        
        return topics, topic_documents
