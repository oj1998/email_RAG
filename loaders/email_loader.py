from typing import Iterator, List
import re
from bs4 import BeautifulSoup
import bs4.element
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import pandas as pd

from gmail_client import GmailClient, EmailSearchOptions

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

class EmailLoader(BaseLoader):
    '''Loads emails into document representation'''
    
    def __init__(self, email_client: GmailClient, search_options: EmailSearchOptions):
        self.email_client = email_client
        self.search_options = search_options

    def _is_tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, bs4.element.Comment):
            return False
        return True

    def _preprocess_raw_html(self, html: str) -> str:
        """Extract meaningful text from raw HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.find_all(text=True)
        visible_texts = filter(self._is_tag_visible, texts)
        return '\n'.join(t.strip() for t in visible_texts)

    def _ingest_email_row(self, row: pd.Series) -> Document:
    """Convert a single email row to a Document with proper metadata for Supabase storage."""
    # Existing HTML and encoding handling code stays the same
    if 'html' in row['mimeType'].lower():
        body_str = self._preprocess_raw_html(row['body'])
    else:
        body_str = row['body']

    # Handle encoding issues - keep existing code
    if isinstance(body_str, bytes):
        encoding = chardet.detect(body_str)['encoding']
        if 'windows' in encoding.lower():
            encoding = 'utf-8'
        try:
            body_str = str(body_str, encoding=encoding)
        except UnicodeDecodeError:
            body_str = row['body'].decode(encoding, errors='ignore')

    # Clean up newlines - keep existing code
    body_str = re.sub(r'[\r\n]\s*[\r\n]', '\n\n', body_str)

    # Updated metadata structure for Supabase
    metadata = {
        'email_id': str(row['id']),  # Ensure string type
        'thread_id': str(row['thread_id']),
        'label_ids': row['label_ids'] if isinstance(row['label_ids'], list) else [],
        'from': row['from'] or '',
        'to': row['to'] or '',
        'subject': row['subject'] or '',
        'date': row['date'] or '',
        'mime_type': row['mimeType'] or '',
        'metadata': {  # Nested metadata for additional info
            'attachment_count': len(row['attachments']) if row['attachments'] else 0,
            'attachment_details': [
                {
                    'filename': att.get('filename', ''),
                    'mime_type': att.get('mime_type', ''),
                    'size': att.get('size', 0)
                }
                for att in (row['attachments'] or [])
            ]
        }
    }

    # Create document with processed content and metadata
    email_doc = Document(
        page_content=body_str,
        metadata=metadata
    )

    # Ensure no None values
    for key, value in email_doc.metadata.items():
        if value is None:
            if isinstance(metadata[key], list):
                email_doc.metadata[key] = []
            else:
                email_doc.metadata[key] = ""

    return email_doc

    def load_and_split(self, text_splitter: TextSplitter = None) -> List[Document]:
        """Load and split documents using specified or default text splitter."""
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            
        all_documents = self.load()
        split_documents = []
        for doc in all_documents:
            split_documents.extend(text_splitter.split_documents([doc]))
        return split_documents

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents one at a time."""
        emails_df = self.email_client.search_emails(self.search_options)
        for _, row in emails_df.iterrows():
            yield self._ingest_email_row(row)
