import ast
import uuid
from enum import Enum
from typing import List, Optional, Any, Dict, Iterable

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, PGVector, SupabaseVectorStore
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, parse_obj_as
import pandas as pd
from supabase.client import create_client
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import DisconnectionError

_COL_EMBEDDINGS = "embedding"
_COL_METADATA = "metadata"
_COL_CONTENT = "content"

# Custom Supabase vector store that properly handles auto-generated IDs
class CustomSupabaseVectorStore(SupabaseVectorStore):
    """Custom Supabase vector store that properly handles auto-generated IDs"""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore with proper ID handling

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts. Will be ignored.

        Returns:
            List of IDs from the vectorstore
        """
        if not texts:
            return []

        # Process inputs
        embeddings = self._embedding.embed_documents(list(texts))
        
        # Create records without explicit ID values
        records = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            record = {
                "content": text,
                "embedding": embedding,
            }
            if metadatas is not None:
                record["metadata"] = metadatas[i]
            records.append(record)

        # Insert directly using Supabase client
        # This lets Supabase handle ID generation
        result = self._client.table(self.table_name).insert(records).execute()
        
        # Extract the generated IDs from the result
        if hasattr(result, 'data') and result.data:
            return [str(item['id']) for item in result.data]
        return []


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"
    SUPABASE = "supabase"

class VectorStoreConfig(BaseModel):
    type: VectorStoreType = VectorStoreType.SUPABASE
    collection_name: str = "email_embeddings"
    persist_directory: str = "./chroma_db"
    connection_string: str = ""
    supabase_url: str = ""
    supabase_key: str = ""

class VectorStoreFactory:
    @staticmethod
    def create(embeddings_model: Embeddings, config: dict = None):
        if config:
            settings = parse_obj_as(VectorStoreConfig, config)
        else:
            settings = VectorStoreConfig()
            
        if settings.type == VectorStoreType.CHROMA:
            return VectorStoreFactory._load_chromadb_store(embeddings_model, settings)
        elif settings.type == VectorStoreType.PGVECTOR:
            return VectorStoreFactory._load_pgvector_store(embeddings_model, settings)
        elif settings.type == VectorStoreType.SUPABASE:
            return VectorStoreFactory._load_supabase_store(embeddings_model, settings)
        else:
            raise ValueError(f"Invalid vector store type, must be one of {VectorStoreType.__members__.keys()}")

    @staticmethod
    def _load_chromadb_store(embeddings_model: Embeddings, settings) -> Chroma:
        return Chroma(
            persist_directory=settings.persist_directory,
            collection_name=settings.collection_name,
            embedding_function=embeddings_model,
        )

    @staticmethod
    def _load_supabase_store(embeddings_model: Embeddings, settings) -> VectorStore:
        """
        Creates a custom Supabase vector store instance that properly handles IDs
        """
        supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        
        return CustomSupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings_model,
            table_name=settings.collection_name,
            query_name="match_email_embeddings"
        )

    @staticmethod
    def _load_pgvector_store(embeddings_model: Embeddings, settings) -> PGVector:
        store = PGVector(
            connection_string=settings.connection_string,
            collection_name=settings.collection_name,
            embedding_function=embeddings_model
        )
        return VectorStoreFactory._load_data_into_langchain_pgvector(settings, store)

    @staticmethod
    def _load_data_into_langchain_pgvector(settings, vectorstore: PGVector) -> PGVector:
        """
        Fetches data from the existing pgvector table and loads it into the langchain pgvector vector store
        """
        df = VectorStoreFactory._fetch_data_from_db(settings)

        df[_COL_EMBEDDINGS] = df[_COL_EMBEDDINGS].apply(ast.literal_eval)
        df[_COL_METADATA] = df[_COL_METADATA].apply(ast.literal_eval)

        metadata = df[_COL_METADATA].tolist()
        embeddings = df[_COL_EMBEDDINGS].tolist()
        texts = df[_COL_CONTENT].tolist()
        
        # Use add_texts to avoid id conflicts
        vectorstore.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata
        )
        return vectorstore

    @staticmethod
    def _fetch_data_from_db(settings: VectorStoreConfig) -> pd.DataFrame:
        """
        Fetches data from database
        """
        engine = create_engine(settings.connection_string)
        try:
            with engine.connect() as connection:
                query = f"""
                    SELECT embedding, metadata, content 
                    FROM {settings.collection_name}
                """
                return pd.read_sql(query, connection)
        except DisconnectionError:
            raise ConnectionError("Failed to connect to database")
