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
        """Convert a single email row to a Document with proper metadata."""
        # Handle HTML content if present
        if 'html' in row['mimeType'].lower():
            body_str = self._preprocess_raw_html(row['body'])
        else:
            body_str = row['body']

        # Handle encoding issues
        if isinstance(body_str, bytes):
            encoding = chardet.detect(body_str)['encoding']
            if 'windows' in encoding.lower():
                encoding = 'utf-8'
            try:
                body_str = str(body_str, encoding=encoding)
            except UnicodeDecodeError:
                body_str = row['body'].decode(encoding, errors='ignore')

        # Clean up newlines
        body_str = re.sub(r'[\r\n]\s*[\r\n]', '\n\n', body_str)

        # Create document with comprehensive metadata
        email_doc = Document(
            page_content=body_str,
            metadata={
                'email_id': row['id'],
                'thread_id': row['thread_id'],
                'label_ids': row['label_ids'],
                'from': row['from'],
                'to': row['to'],
                'subject': row['subject'],
                'date': row['date'],
                'has_attachments': bool(row['attachments']),
                'attachment_count': len(row['attachments']) if row['attachments'] else 0,
                'attachment_details': row['attachments'] if row['attachments'] else [],
                'mime_type': row['mimeType']
            }
        )

        # Replace None values with empty strings
        for key in email_doc.metadata:
            if email_doc.metadata[key] is None:
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
