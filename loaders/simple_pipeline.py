from typing import List, Dict, Optional, Any, Callable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loaders.email_loader import EmailLoader
from client.gmail_client import GmailClient, EmailSearchOptions

class SimpleEmailPipeline:
    """
    Enhanced pipeline that handles email ingestion and embedding with progress tracking
    """
    def __init__(
        self, 
        embeddings_model: Embeddings,
        vector_store: VectorStore
    ):
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store
        self._progress_callback = None

    def register_progress_callback(self, callback_fn: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive progress updates
        
        Args:
            callback_fn: Function that accepts a progress data dictionary
        """
        self._progress_callback = callback_fn

    def _report_progress(self, stage: str, current: int, total: int, message: str = None) -> None:
        """
        Report progress to registered callback
        
        Args:
            stage: Current processing stage
            current: Current item number
            total: Total items to process
            message: Optional status message
        """
        if self._progress_callback:
            self._progress_callback({
                'stage': stage,
                'current': current,
                'total': total,
                'percentage': int((current / max(total, 1)) * 100),
                'message': message or ''
            })

    def ingest_emails(self, gmail_client: GmailClient, search_options: EmailSearchOptions) -> Dict[str, Any]:
        """
        Process emails and store their embeddings with progress tracking
        
        Args:
            gmail_client: Authenticated Gmail client
            search_options: Search and processing options
            
        Returns:
            Dict with processing statistics
        """
        # Report the start of loading
        self._report_progress('loading', 0, 1, "Starting email retrieval...")
        
        # Load and process emails
        loader = EmailLoader(gmail_client, search_options)
        
        # Load raw documents first
        self._report_progress('loading', 0, 1, "Retrieving emails from Gmail...")
        documents = loader.load()
        
        total_docs = len(documents)
        self._report_progress('loading', 1, 1, f"Retrieved {total_docs} emails")
        
        # Split documents using the chunk settings from search options
        self._report_progress('splitting', 0, 1, "Splitting documents...")
        split_documents = loader.load_and_split()
        
        total_chunks = len(split_documents)
        self._report_progress('splitting', 1, 1, f"Created {total_chunks} chunks from {total_docs} emails")
        
        # Create embeddings and store documents in batches
        self._report_progress('embedding', 0, total_chunks, "Creating embeddings...")
        
        # Process in smaller batches to report progress
        batch_size = min(20, max(1, total_chunks // 10))  # Adaptive batch size
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        for i in range(0, total_chunks, batch_size):
            # Get the current batch
            end_idx = min(i + batch_size, total_chunks)
            batch = split_documents[i:end_idx]
            batch_count = end_idx - i
            
            # Extract texts and metadata separately to avoid ID conflicts
            texts = [doc.page_content for doc in batch]
            metadatas = [
                {k: v for k, v in doc.metadata.items() if k != 'id'}
                for doc in batch
            ]
            
            # Store documents with embeddings
            self._report_progress(
                'embedding', 
                i // batch_size + 1, 
                total_batches, 
                f"Processing batch {i // batch_size + 1}/{total_batches} ({batch_count} chunks)"
            )

            print("=" * 80)
            print(f"PIPELINE: About to add {len(texts)} texts to vector store")
            print(f"PIPELINE: Vector store type: {type(self.vector_store).__name__}")
            print(f"PIPELINE: First few metadata keys: {list(metadatas[0].keys()) if metadatas and len(metadatas) > 0 else 'No metadata'}")
            
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
        
        # Final completion report
        self._report_progress('complete', 1, 1, "Processing complete")
        
        # Return processing statistics
        return {
            'total_emails': total_docs,
            'total_chunks': total_chunks,
            'avg_chunks_per_email': round(total_chunks / max(total_docs, 1), 2),
            'processing_parameters': {
                'chunk_size': search_options.chunk_size,
                'chunk_overlap': search_options.chunk_overlap,
                'max_results': search_options.max_results,
                'loading_strategy': search_options.loading_strategy
            },
            'status': 'complete'
        }

    def clear_vector_store(self) -> None:
        """
        Clear all documents from the vector store
        
        Note: This operation is irreversible and will delete all indexed emails.
        """
        # Check if the vector store has a delete_collection method (like Chroma)
        if hasattr(self.vector_store, 'delete_collection'):
            self.vector_store.delete_collection()
            if hasattr(self.vector_store, 'create_collection'):
                self.vector_store.create_collection()
        # Check for FAISS-style clear method
        elif hasattr(self.vector_store, 'reset'):
            self.vector_store.reset()
        # For other vector stores, we'll need to create a new instance
        else:
            raise NotImplementedError(
                "Clear operation not supported for this vector store type. "
                "You may need to manually create a new vector store instance."
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector store
        
        Returns:
            Dict with vector store statistics
        """
        # Different vector stores expose different APIs, try common patterns
        stats = {
            'vector_store_type': self.vector_store.__class__.__name__
        }
        
        # Try to get document count
        if hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'count'):
            stats['document_count'] = self.vector_store._collection.count()
        elif hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'ntotal'):
            stats['document_count'] = self.vector_store.index.ntotal
        else:
            # For vector stores without direct count methods, estimate if possible
            stats['document_count'] = 'Unknown'
            
        # Try to get embedding dimension
        if hasattr(self.vector_store, '_embedding_dimension'):
            stats['embedding_dimension'] = self.vector_store._embedding_dimension
        elif hasattr(self.embeddings_model, 'embedding_dimension'):
            stats['embedding_dimension'] = self.embeddings_model.embedding_dimension
        else:
            stats['embedding_dimension'] = 'Unknown'
            
        return stats
