from typing import List, Dict, Optional, Union, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable, RunnableLambda

DEFAULT_EMAIL_PROMPT_TEMPLATE = """
You are an email assistant. Given a question, use the following email context to provide a helpful response.

Email Context:
{context}

Question:
{question}

Response (be concise and only reference information found in the emails):
"""

DEFAULT_SEARCH_QUERY_TEMPLATE = """
You are an expert at creating email search queries. Based on the user's question, create a search query that will find the most relevant emails.

User question: {question}

Think about what keywords would be most effective to search for. Consider dates, sender/recipient names, and key topic words.
Focus on being specific rather than broad to get the most relevant results.

Return only the search query string, nothing else.
"""

class EmailFilterOptions(BaseModel):
    """Filter parameters for email retrieval"""
    after_date: Optional[str] = Field(None, description="Emails after this date (YYYY/MM/DD)")
    before_date: Optional[str] = Field(None, description="Emails before this date (YYYY/MM/DD)")
    from_email: Optional[str] = Field(None, description="Sender email address")
    to_email: Optional[str] = Field(None, description="Recipient email address")
    subject_contains: Optional[str] = Field(None, description="Subject contains these words")
    has_attachment: Optional[bool] = Field(None, description="Email has attachments")
    label: Optional[str] = Field(None, description="Email has this Gmail label")

    def to_query_string(self) -> str:
        """Convert filters to Gmail query string format"""
        query_parts = []
        
        if self.after_date:
            query_parts.append(f"after:{self.after_date}")
        if self.before_date:
            query_parts.append(f"before:{self.before_date}")
        if self.from_email:
            query_parts.append(f"from:{self.from_email}")
        if self.to_email:
            query_parts.append(f"to:{self.to_email}")
        if self.subject_contains:
            query_parts.append(f"subject:({self.subject_contains})")
        if self.has_attachment is not None:
            if self.has_attachment:
                query_parts.append("has:attachment")
            else:
                query_parts.append("-has:attachment")
        if self.label:
            query_parts.append(f"label:{self.label}")
            
        return " ".join(query_parts)

    def to_metadata_filter(self) -> Dict[str, Any]:
        """Convert filters to metadata filter format for vector stores"""
        metadata_filter = {}
        
        if self.after_date:
            metadata_filter["date_gte"] = self.after_date
        if self.before_date:
            metadata_filter["date_lte"] = self.before_date
        if self.from_email:
            metadata_filter["sender_contains"] = self.from_email
        if self.to_email:
            metadata_filter["recipients_contains"] = self.to_email
        if self.subject_contains:
            metadata_filter["subject_contains"] = self.subject_contains
        if self.has_attachment is not None:
            if self.has_attachment:
                metadata_filter["metadata.attachment_count_gt"] = 0
            else:
                metadata_filter["metadata.attachment_count"] = 0
        if self.label:
            metadata_filter["label_ids_contains"] = self.label
            
        return metadata_filter

class EmailRetriever(BaseRetriever):
    """Retriever for querying embedded emails"""
    
    vector_store: VectorStore
    embeddings_model: Embeddings
    filters: Optional[EmailFilterOptions] = None
    k: int = 5
    search_type: str = "similarity"
    use_reranker: bool = False
    llm: Optional[BaseChatModel] = None
    
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings_model: Embeddings,
        filters: Optional[EmailFilterOptions] = None,
        k: int = 5,
        search_type: str = "similarity",
        use_reranker: bool = False,
        llm: Optional[BaseChatModel] = None
    ):
        """
        Initialize the email retriever
        
        Args:
            vector_store: Vector store containing email embeddings
            embeddings_model: Model to generate embeddings
            filters: Optional email filter options
            k: Number of documents to retrieve
            search_type: Search algorithm type (similarity, mmr)
            use_reranker: Whether to use reranking
            llm: Language model for reranking (if needed)
        """
        # Pass all parameters to parent constructor for validation
        super().__init__(
            vector_store=vector_store,
            embeddings_model=embeddings_model,
            filters=filters,
            k=k,
            search_type=search_type,
            use_reranker=use_reranker,
            llm=llm
        )
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query"""
        # If additional metadata filters are provided, apply them
        search_kwargs = {}
        
        if self.filters:
            # Convert filters to metadata filter format
            metadata_filter = self.filters.to_metadata_filter()
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
        
        # Retrieve documents
        if self.search_type == "mmr":
            docs = self.vector_store.max_marginal_relevance_search(
                query, k=self.k, **search_kwargs
            )
        else:
            # Default to similarity search
            docs = self.vector_store.similarity_search(
                query, k=self.k, **search_kwargs
            )
        
        # Apply reranking if enabled and LLM is provided
        if self.use_reranker and self.llm and len(docs) > 1:
            docs = self._rerank_documents(query, docs)
            
        return docs
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query"""
        try:
            # Use Cohere's reranker if available
            reranker = CohereRerank()
            reranked_docs = reranker.compress_documents(docs, query)
            return reranked_docs
        except Exception as e:
            # Fallback to simple relevance scoring if Cohere is not available
            # This is a simplified implementation
            relevance_scores = []
            for doc in docs:
                # Calculate simple relevance score based on term overlap
                query_terms = set(query.lower().split())
                content_terms = set(doc.page_content.lower().split())
                score = len(query_terms.intersection(content_terms)) / max(len(query_terms), 1)
                relevance_scores.append((doc, score))
            
            # Sort by score descending
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in relevance_scores]

    def as_runnable(self) -> RunnableSerializable:
        """Convert retriever to a runnable for use in LangChain pipelines"""
        def retrieve_func(query_dict):
            query = query_dict.get("query", "")
            if not query:
                query = query_dict.get("question", "")
            return self.get_relevant_documents(query)
            
        return RunnablePassthrough() | RunnableLambda(retrieve_func)

class EmailQASystem:
    """System for answering questions based on email content"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings_model: Embeddings,
        llm: BaseChatModel,
        k: int = 5,
        use_reranker: bool = False,
        prompt_template: str = DEFAULT_EMAIL_PROMPT_TEMPLATE,
        search_query_template: str = DEFAULT_SEARCH_QUERY_TEMPLATE
    ):
        """
        Initialize the email QA system
        
        Args:
            vector_store: Vector store containing email embeddings
            embeddings_model: Model to generate embeddings
            llm: Language model for generating answers
            k: Number of documents to retrieve
            use_reranker: Whether to use reranking
            prompt_template: Template for QA prompt
            search_query_template: Template for query generation
        """
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.k = k
        self.use_reranker = use_reranker
        self.prompt_template = prompt_template
        self.search_query_template = search_query_template
        
        # Initialize the retriever
        self.retriever = EmailRetriever(
            vector_store=vector_store,
            embeddings_model=embeddings_model,
            k=k,
            use_reranker=use_reranker,
            llm=llm
        )
        
        # Build the QA chain
        self.qa_chain = self._build_qa_chain()
        
    def _build_qa_chain(self):
        """Build QA chain with retriever and LLM"""
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Generate search query from natural language question
        search_query_prompt = ChatPromptTemplate.from_template(self.search_query_template)
        search_query_chain = (
            search_query_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # QA prompt
        qa_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        # Chain that generates answer from docs
        qa_chain = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["retrieved_docs"]))
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Full chain with retrieval
        def retrieve_with_query(inputs):
            if "query_str" in inputs and inputs["query_str"]:
                # If query_str is provided, use it directly
                query = inputs["query_str"]
            else:
                # Otherwise, generate search query from question
                query = search_query_chain.invoke({"question": inputs["question"]})
                
            # Update filters if provided
            if "filters" in inputs and inputs["filters"]:
                try:
                    self.retriever.filters = EmailFilterOptions(**inputs["filters"])
                except Exception as e:
                    print(f"Warning: Invalid filter options: {e}")
                    self.retriever.filters = None
            else:  # This line has incorrect indentation
                self.retriever.filters = None
            # Retrieve documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            return {
                "retrieved_docs": retrieved_docs,
                "question": inputs["question"]
            }
            
        full_chain = (
            RunnablePassthrough()
            | retrieve_with_query
            | qa_chain
        )
        
        return full_chain
    
    def query(
        self, 
        question: str, 
        query_str: Optional[str] = None,
        filters: Optional[Dict] = None,
        k: Optional[int] = None
    ) -> str:
        """
        Answer a question based on email content
        
        Args:
            question: Natural language question
            query_str: Optional explicit search query
            filters: Optional dictionary of email filters
            k: Optional override for number of documents to retrieve
            
        Returns:
            Answer string
        """
        # Temporarily update k if specified
        original_k = self.retriever.k
        if k is not None:
            self.retriever.k = k
            
        try:
            input_dict = {"question": question}
            if query_str:
                input_dict["query_str"] = query_str
            if filters:
                input_dict["filters"] = filters
                
            answer = self.qa_chain.invoke(input_dict)
            return answer
        finally:
            # Restore original k
            if k is not None:
                self.retriever.k = original_k
    
    def get_relevant_emails(
        self,
        query: str,
        filters: Optional[Dict] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant emails with relevance scores
        
        Args:
            query: Search query
            filters: Optional dictionary of email filters
            k: Optional override for number of documents to retrieve
            
        Returns:
            List of relevant email documents with relevance scores in metadata
        """
        # Temporarily update retriever settings
        original_k = self.retriever.k
        original_filters = self.retriever.filters
        
        try:
            if k is not None:
                self.retriever.k = k
            if filters:
                try:
                    self.retriever.filters = EmailFilterOptions(**filters)
                except Exception as e:
                    print(f"Warning: Invalid filter options: {e}")
                    self.retriever.filters = None
                    
            # Get search parameters
            search_kwargs = {}
            if self.retriever.filters:
                metadata_filter = self.retriever.filters.to_metadata_filter()
                if metadata_filter:
                    search_kwargs["filter"] = metadata_filter
            
            k_value = k if k is not None else self.retriever.k
            
            try:
                # Attempt to use similarity_search_with_score
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, 
                    k=k_value, 
                    **search_kwargs
                )
                
                # Add the score to each document's metadata
                for doc, score in docs_with_scores:
                    # Convert score to float and store in metadata
                    try:
                        # Handle different score formats (some might be cosine similarity, others distance)
                        score_float = float(score)
                        # Normalize if needed (some vector stores use distance metrics where lower is better)
                        doc.metadata["relevance_score"] = score_float
                    except (ValueError, TypeError):
                        # If conversion fails, use a default score
                        doc.metadata["relevance_score"] = 0.5
                
                documents = [doc for doc, _ in docs_with_scores]
                
            except (AttributeError, NotImplementedError) as e:
                # Fallback to original method if vector store doesn't support similarity_search_with_score
                print(f"Warning: similarity_search_with_score not supported, falling back: {e}")
                documents = self.retriever.get_relevant_documents(query)
                # Set a default relevance score so downstream code doesn't break
                for i, doc in enumerate(documents):
                    # Assign descending scores based on position
                    doc.metadata["relevance_score"] = 1.0 - (i * (0.5 / max(1, len(documents))))
            
            # Apply reranking if enabled
            if self.use_reranker and hasattr(self, 'llm') and self.llm and len(documents) > 1:
                try:
                    # Get original scores before reranking
                    original_scores = {id(doc): doc.metadata.get("relevance_score", 0.5) for doc in documents}
                    
                    # Apply reranking
                    reranker = CohereRerank()
                    reranked_docs = reranker.compress_documents(documents, query)
                    
                    # Transfer original scores to reranked docs, adjusted for new position
                    for i, doc in enumerate(reranked_docs):
                        if id(doc) in original_scores:
                            # Preserve original score but adjust slightly for new rank
                            rank_adjustment = 1.0 - (i * 0.05)
                            doc.metadata["relevance_score"] = original_scores[id(doc)] * rank_adjustment
                    
                    documents = reranked_docs
                except Exception as e:
                    print(f"Warning: Reranking failed: {e}")
                    # Continue with documents as-is
            
            return documents
            
        finally:
            # Restore original settings
            self.retriever.k = original_k
            self.retriever.filters = original_filters
