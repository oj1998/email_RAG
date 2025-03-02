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
        """Add texts to the vectorstore with proper ID handling"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"CustomSupabaseVectorStore.add_texts called with {len(list(texts))} texts")
        print("=" * 80)
        print(f"SUPABASE: add_texts called with {len(list(texts))} texts")
        
        if not texts:
            logger.warning("No texts provided to add_texts, returning empty list")
            return []
    
        try:
            # Process inputs
            logger.info("Generating embeddings for texts")
            embeddings = self._embedding.embed_documents(list(texts))
            logger.info(f"Generated {len(embeddings)} embeddings")
            print(f"SUPABASE: Generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 'unknown'}")
            
            # Create records without explicit ID values
            records = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                record = {
                    "content": text,
                    "embedding": embedding,
                }
                if metadatas is not None:
                    meta = metadatas[i]
                    record["metadata"] = meta  # Keep the full metadata
                    
                    # Extract specific fields to match table schema
                    record["email_id"] = meta.get("email_id", "")
                    record["thread_id"] = meta.get("thread_id", "")
                    record["subject"] = meta.get("subject", "")
                    record["sender"] = meta.get("sender", "")
                    record["recipients"] = meta.get("recipients", "")
                    record["date"] = meta.get("date", "")
                    record["label_ids"] = meta.get("label_ids", [])
                    record["mime_type"] = meta.get("mime_type", "")
                records.append(record)
            
            logger.info(f"Prepared {len(records)} records for insertion")
            print(f"SUPABASE: About to insert {len(records)} records into {self.table_name}")
            print(f"SUPABASE: First record keys: {list(records[0].keys()) if records else 'No records'}")
            
            # Insert directly using Supabase client
            # This lets Supabase handle ID generation
            logger.info(f"Inserting records into table {self.table_name}")
            result = self._client.table(self.table_name).insert(records).execute()

            print(f"SUPABASE: Insert result: {result}")
            print(f"SUPABASE: Response data: {result.data if hasattr(result, 'data') else 'No data'}")
            print(f"SUPABASE: Response error: {result.error if hasattr(result, 'error') else 'No error'}")

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Supabase insert operation details:")
            logger.info(f"Table name: {self.table_name}")
            logger.info(f"Number of records: {len(records)}")
            logger.info(f"Response data: {result.data if hasattr(result, 'data') else 'No data'}")
            logger.info(f"Response error: {result.error if hasattr(result, 'error') else 'No error'}")
            
            logger.info(f"Insert result: {result}")
            
            # Extract the generated IDs from the result
            if hasattr(result, 'data') and result.data:
                ids = [str(item['id']) for item in result.data]
                logger.info(f"Successfully inserted {len(ids)} records")
                return ids
            else:
                logger.warning("No data returned from insert operation")
                return []
        except Exception as e:
            logger.error(f"Error in add_texts: {str(e)}", exc_info=True)
            raise


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
