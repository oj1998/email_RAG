from langchain_community.vectorstores import SupabaseVectorStore
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class EnhancedSupabaseVectorStore(SupabaseVectorStore):
    """Extended version of SupabaseVectorStore that guarantees similarity_search_with_score support."""
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return documents most similar to query and their similarity scores.
        
        This implementation ensures that scores are always returned with documents.
        
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            
        Returns:
            List of Documents most similar to the query and their similarity scores.
        """
        try:
            # Try the parent implementation first
            return super().similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        except (AttributeError, NotImplementedError) as e:
            logger.info(f"Original similarity_search_with_score not supported: {e}. Using custom implementation.")
            
            # Implement our own version using similarity_search_by_vector
            # First, get the embedding for the query
            embedding = self.embedding_function.embed_query(query)
            
            # Get kwargs for the search
            pgvector_kwargs = kwargs.get("pgvector_kwargs", {})
            
            # Use match_vectors for score
            if not hasattr(self.client, "rpc"):
                raise ValueError("Supabase client does not have rpc method")
            
            match_documents_params = {
                "query_embedding": embedding,
                "match_count": k,
            }
            
            if filter is not None:
                # Convert filter to a format understood by Supabase
                match_documents_params["filter"] = self._get_filter_string(filter)
                
            # Add any additional pgvector specific parameters
            match_documents_params.update(pgvector_kwargs)
            
            # Call the match_vectors function via RPC
            res = self.client.rpc(
                self.query_name, match_documents_params
            ).execute()
            
            # Process the results
            if len(res.data) == 0:
                return []
            
            documents_with_scores = []
            for result in res.data:
                metadata = {}
                # Remove embedding from metadata if it exists
                for key, value in result.items():
                    if key != "content" and key != "embedding" and key != "similarity":
                        metadata[key] = value
                
                # Create Document object
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=metadata
                )
                
                # Get similarity score - higher is better in this case
                score = result.get("similarity", 0.0)
                
                documents_with_scores.append((doc, score))
            
            return documents_with_scores
    
    def _get_filter_string(self, filter_dict: Dict[str, Any]) -> str:
        """Convert a filter dict to a SQL string for Supabase filtering."""
        conditions = []
        
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Handle operators like $eq, $gt, etc.
                for op_key, op_value in value.items():
                    if op_key == "$eq":
                        conditions.append(f"{key} = '{op_value}'")
                    elif op_key == "$ne":
                        conditions.append(f"{key} != '{op_value}'")
                    elif op_key == "$gt":
                        conditions.append(f"{key} > '{op_value}'")
                    elif op_key == "$gte":
                        conditions.append(f"{key} >= '{op_value}'")
                    elif op_key == "$lt":
                        conditions.append(f"{key} < '{op_value}'")
                    elif op_key == "$lte":
                        conditions.append(f"{key} <= '{op_value}'")
                    elif op_key == "$in":
                        values_str = "', '".join([str(v) for v in op_value])
                        conditions.append(f"{key} IN ('{values_str}')")
            else:
                # Simple equality
                conditions.append(f"{key} = '{value}'")
        
        return " AND ".join(conditions)
