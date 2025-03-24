from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import logging
import asyncio
import re

logger = logging.getLogger(__name__)

class TopicRetrievalResult(BaseModel):
    """Result of retrieving documents for a topic"""
    topic: str
    documents: List[str]  # Document IDs
    relevance_scores: Dict[str, float]  # Document ID to relevance score
    
class VarianceDocumentRetriever:
    """Retrieves and prepares documents for variance analysis"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: Optional[ChatOpenAI] = None,
        embeddings_model = None,
        min_relevance: float = 0.7,
        max_documents: int = 10
    ):
        self.vector_store = vector_store
        self.llm = llm or ChatOpenAI(temperature=0.0)
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        self.min_relevance = min_relevance
        self.max_documents = max_documents
        
        # Initialize prompts
        self.topic_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analysis expert. Extract the main topic that the user wants to analyze
            for document variance. Return ONLY the topic name as a single phrase, without any explanation or additional text."""),
            ("user", "Query: {query}")
        ])
        
        self.document_filter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analysis expert. Determine if this document excerpt is relevant to the
            specified topic for variance analysis. Focus only on relevance, not on quality or comprehensiveness.
            
            Return ONLY a number from 0.0 to 1.0 representing relevance, where:
            - 0.0 means completely irrelevant
            - 1.0 means highly relevant and focused on the topic
            
            Output ONLY the number, with no explanation or additional text."""),
            ("user", """Topic: {topic}
            
            Document Excerpt:
            {excerpt}""")
        ])
        
    async def extract_topic(self, query: str) -> str:
        """Extract the main topic from a variance analysis query"""
        response = await self.llm.ainvoke(
            self.topic_extraction_prompt.format_messages(query=query)
        )
        topic = response.content.strip()
        logger.info(f"Extracted topic for variance analysis: {topic}")
        return topic
    
    async def retrieve_documents_for_topic(
        self, 
        topic: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: int = 20  # Retrieve more initially for filtering
    ) -> List[Document]:
        """Retrieve documents relevant to a topic from the vector store"""
        try:
            # Use vector search to find potentially relevant documents
            raw_docs = await self.vector_store.asimilarity_search(
                topic,
                k=k,
                filter=metadata_filter
            )
            
            if not raw_docs:
                logger.warning(f"No documents found for topic: {topic}")
                return []
                
            logger.info(f"Retrieved {len(raw_docs)} initial documents for topic: {topic}")
            
            # Filter documents for relevance
            filtered_docs = await self._filter_documents_for_relevance(topic, raw_docs)
            
            # Sort by relevance
            filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)
            
            # Take top max_documents
            result_docs = [doc for doc, score in filtered_docs[:self.max_documents]]
            
            logger.info(f"Filtered to {len(result_docs)} relevant documents for topic: {topic}")
            return result_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents for topic {topic}: {str(e)}")
            # Fallback to basic retrieval
            try:
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": min(10, k), "filter": metadata_filter}
                )
                docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: retriever.get_relevant_documents(topic)
                )
                return docs
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []
    
    async def _filter_documents_for_relevance(
        self, 
        topic: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Filter documents for relevance to the topic using a combination of approaches"""
        filtered_docs = []
        
        # First, use embeddings for basic filtering
        if self.embeddings_model:
            try:
                # Compute topic embedding
                topic_embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.embeddings_model.embed_query(topic)
                )
                
                # Score documents based on semantic similarity
                for doc in documents:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        try:
                            doc_embedding = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.embeddings_model.embed_query(doc.page_content)
                            )
                            
                            # Calculate cosine similarity
                            similarity = np.dot(topic_embedding, doc_embedding)
                            # Normalize to 0-1 range
                            normalized_similarity = (similarity + 1) / 2
                            
                            if normalized_similarity >= self.min_relevance:
                                filtered_docs.append((doc, normalized_similarity))
                        except Exception as e:
                            logger.warning(f"Error computing embedding for document: {e}")
                            # Skip this document
            except Exception as e:
                logger.warning(f"Embedding-based filtering failed: {e}")
                # Fall back to using all documents without embedding filtering
                filtered_docs = [(doc, 0.7) for doc in documents]
        
        # If we have very few documents after embedding filtering, try to get more
        if len(filtered_docs) < 3 and documents:
            # Fall back to using all documents with a base score
            filtered_docs = [(doc, 0.7) for doc in documents]
        
        # Further refine relevance with LLM for more nuanced filtering
        # Only do this for a reasonable number of documents to avoid too many API calls
        if self.llm and len(filtered_docs) <= 15:
            refined_docs = []
            
            for doc, base_score in filtered_docs:
                try:
                    # Extract a representative excerpt
                    excerpt = self._extract_representative_excerpt(doc.page_content, topic)
                    
                    # Get LLM-based relevance score
                    response = await self.llm.ainvoke(
                        self.document_filter_prompt.format_messages(
                            topic=topic,
                            excerpt=excerpt
                        )
                    )
                    
                    # Parse the score
                    llm_score_str = response.content.strip()
                    llm_score_match = re.search(r'(\d+\.\d+|\d+)', llm_score_str)
                    if llm_score_match:
                        llm_score = float(llm_score_match.group(1))
                        
                        # Combine scores (weighted average)
                        combined_score = 0.7 * base_score + 0.3 * llm_score
                        
                        if combined_score >= self.min_relevance:
                            refined_docs.append((doc, combined_score))
                    else:
                        # If parsing fails, use the base score
                        refined_docs.append((doc, base_score))
                except Exception as e:
                    logger.warning(f"Error getting LLM relevance score: {e}")
                    # Keep the document with its base score
                    refined_docs.append((doc, base_score))
            
            return refined_docs
        
        return filtered_docs
    
    def _extract_representative_excerpt(self, text: str, topic: str, max_length: int = 300) -> str:
        """Extract a representative excerpt from text based on topic relevance"""
        # Simple implementation: Find sentences containing the topic or related terms
        topic_terms = topic.lower().split()
        
        # Try to find paragraphs containing topic terms
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if any(term in paragraph.lower() for term in topic_terms):
                if len(paragraph) <= max_length:
                    return paragraph
                else:
                    # Find a good breakpoint
                    end_idx = paragraph.rfind('.', 0, max_length)
                    if end_idx == -1:
                        end_idx = max_length
                    return paragraph[:end_idx+1]
        
        # If no paragraph contains topic terms, return the beginning of the text
        if len(text) <= max_length:
            return text
        else:
            end_idx = text.rfind('.', 0, max_length)
            if end_idx == -1:
                end_idx = max_length
            return text[:end_idx+1]
    
    async def get_variance_documents(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Main method to get documents for variance analysis based on a query
        
        Args:
            query: User's query about document variance
            metadata_filter: Optional filter for document metadata
            
        Returns:
            List of documents relevant for variance analysis
        """
        # Extract the topic from the query
        topic = await self.extract_topic(query)
        
        # Retrieve documents for the topic
        documents = await self.retrieve_documents_for_topic(
            topic=topic,
            metadata_filter=metadata_filter
        )
        
        # Make sure we have enough documents for a meaningful variance analysis
        if len(documents) < 2:
            logger.warning(f"Insufficient documents ({len(documents)}) for variance analysis on topic: {topic}")
        
        return documents
    
    def group_documents_by_source(
        self, 
        documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """Group documents by their source for more coherent analysis"""
        grouped = {}
        
        for doc in documents:
            # Determine source ID (could be document_id, source_id, or generated)
            source_id = doc.metadata.get('document_id')
            if not source_id:
                source_id = doc.metadata.get('source_id')
            if not source_id:
                # Generate a source ID from title or other metadata
                title = doc.metadata.get('title', 'Unknown')
                source_id = f"doc_{hash(title) % 10000}"
            
            if source_id not in grouped:
                grouped[source_id] = []
            
            grouped[source_id].append(doc)
        
        return grouped
    
    async def merge_document_chunks(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Merge chunks from the same document for better variance analysis
        
        This helps prevent artificial variances due to document chunking
        """
        # Group by document ID and sort by page/chunk index
        document_groups = {}
        
        for doc in documents:
            doc_id = doc.metadata.get('document_id')
            if not doc_id:
                # Skip documents without ID
                continue
                
            if doc_id not in document_groups:
                document_groups[doc_id] = []
                
            document_groups[doc_id].append(doc)
        
        # Sort each group by page number or chunk index
        for doc_id, group in document_groups.items():
            group.sort(key=lambda x: (
                x.metadata.get('page', 0),
                x.metadata.get('chunk_index', 0)
            ))
        
        # Merge content for each document
        merged_documents = []
        
        for doc_id, group in document_groups.items():
            if not group:
                continue
                
            # Take metadata from first chunk but merge content
            merged_content = "\n\n".join([doc.page_content for doc in group if hasattr(doc, 'page_content')])
            
            # Create new document with merged content
            merged_doc = Document(
                page_content=merged_content,
                metadata=group[0].metadata.copy()
            )
            
            # Update metadata to indicate merged status
            merged_doc.metadata['merged'] = True
            merged_doc.metadata['chunk_count'] = len(group)
            
            merged_documents.append(merged_doc)
        
        return merged_documents
