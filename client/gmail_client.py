import base64
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum  # Add missing import
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd

class LoadingStrategy(str, Enum):
    NEWER_FIRST = "newer-first"
    OLDER_FIRST = "older-first"
    IMPORTANT_FIRST = "important-first"

@dataclass
class EmailSearchOptions:
    """Enhanced search parameters for Gmail API"""
    # Basic search parameters
    query: str = None  # Gmail search query
    max_results: int = 100  # Number of emails to load
    include_attachments: bool = False
    
    # Filter parameters
    subject: str = None
    from_email: str = None
    to_email: str = None
    after_date: str = None  # Format: YYYY/MM/DD
    before_date: str = None  # Format: YYYY/MM/DD
    has_label: str = None
    
    # Processing parameters
    loading_strategy: LoadingStrategy = LoadingStrategy.NEWER_FIRST
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    def build_query(self) -> str:
        """Build a complete Gmail API query string from the parameters"""
        query_parts = []
        
        # Add main query if specified
        if self.query:
            query_parts.append(f"({self.query})")
            
        # Add filter parameters
        if self.subject:
            query_parts.append(f"subject:({self.subject})")
        if self.from_email:
            query_parts.append(f"from:{self.from_email}")
        if self.to_email:
            query_parts.append(f"to:{self.to_email}")
        if self.after_date:
            query_parts.append(f"after:{self.after_date}")
        if self.before_date:
            query_parts.append(f"before:{self.before_date}")
        if self.has_label:
            query_parts.append(f"label:{self.has_label}")
        
        # Return combined query or default to all emails
        return " ".join(query_parts) if query_parts else "in:anywhere"

class GmailClient:
    """Client for interacting with Gmail API"""
    
    def __init__(self, credentials: Dict = None, token: Dict = None):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.credentials = credentials
        self.token = token
        self.service = None
        if credentials and token:
            self.service = self._get_gmail_service()

    @staticmethod
    def get_auth_url(credentials: Dict) -> tuple[Flow, str]:
        """Generate authorization URL for Gmail OAuth"""
        flow = Flow.from_client_config(
            credentials,
            scopes=['https://www.googleapis.com/auth/gmail.readonly'],
            redirect_uri=os.getenv('REDIRECT_URI', 'http://localhost:8000/oauth2callback')
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return flow, auth_url

    def authorize_with_code(self, flow: Flow, code: str) -> Dict:
        """Exchange authorization code for tokens"""
        flow.fetch_token(code=code)
        credentials = flow.credentials
        self.token = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        self.service = self._get_gmail_service()
        return self.token

    def _get_gmail_service(self):
        """Get authenticated Gmail service"""
        if not self.token:
            raise ValueError("No token available. Must authenticate first.")
            
        creds = Credentials(
            token=self.token['token'],
            refresh_token=self.token['refresh_token'],
            token_uri=self.token['token_uri'],
            client_id=self.token['client_id'],
            client_secret=self.token['client_secret'],
            scopes=self.token['scopes']
        )
        
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Update stored token
                self.token['token'] = creds.token

        return build('gmail', 'v1', credentials=creds)

    def _get_message_data(self, msg_id: str, include_attachments: bool = False) -> Dict:
        """
        Get full message data for a specific email
        
        Args:
            msg_id: Gmail message ID
            include_attachments: Whether to include attachment data
            
        Returns:
            Dict containing email data
        """
        message = self.service.users().messages().get(
            userId='me', id=msg_id, format='full').execute()
        
        headers = message['payload']['headers']
        
        # Extract email metadata
        email_data = {
            'id': message['id'],
            'thread_id': message['threadId'],
            'label_ids': message['labelIds'],
            'subject': '',
            'from': '',
            'to': '',
            'date': '',
            'body': '',
            'attachments': [],
            'mimeType': message['payload'].get('mimeType', '')
        }
        
        # Get header fields
        for header in headers:
            name = header['name'].lower()
            if name == 'subject':
                email_data['subject'] = header['value']
            elif name == 'from':
                email_data['from'] = header['value']
            elif name == 'to':
                email_data['to'] = header['value']
            elif name == 'date':
                email_data['date'] = header['value']

        # Get message body
        if 'parts' in message['payload']:
            parts = message['payload']['parts']
            email_data['body'] = self._get_body_from_parts(parts)
            if include_attachments:
                email_data['attachments'] = self._get_attachments_from_parts(parts, message['id'])
        else:
            data = message['payload']['body'].get('data', '')
            email_data['body'] = base64.urlsafe_b64decode(data).decode('utf-8')

        return email_data

    def _get_body_from_parts(self, parts: List) -> str:
        """Extract email body from message parts"""
        body = ""
        for part in parts:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data', '')
                text = base64.urlsafe_b64decode(data).decode('utf-8')
                body += text
            elif 'parts' in part:
                body += self._get_body_from_parts(part['parts'])
        return body

    def _get_attachments_from_parts(self, parts: List, message_id: str) -> List[Dict]:
        """Extract attachment metadata from message parts"""
        attachments = []
        for part in parts:
            if 'filename' in part and part['filename']:
                attachment = {
                    'filename': part['filename'],
                    'mime_type': part['mimeType'],
                    'size': part['body'].get('size', 0),
                    'attachment_id': part['body'].get('attachmentId', ''),
                    'message_id': message_id
                }
                attachments.append(attachment)
            if 'parts' in part:
                attachments.extend(self._get_attachments_from_parts(part['parts'], message_id))
        return attachments

    def get_user_email(self) -> Optional[str]:
        """
        Get the email address of the authenticated user
        
        Returns:
            Email address of the authenticated user or None if unsuccessful
        """
        if not self.service:
            return None
            
        try:
            # Call the Gmail API to get user profile
            profile = self.service.users().getProfile(userId='me').execute()
            return profile.get('emailAddress')
        except Exception as e:
            # Log the error (assuming you have logging configured)
            import logging
            logging.error(f"Error getting user email: {str(e)}")
            return None

    def search_emails(self, options: EmailSearchOptions) -> pd.DataFrame:
        """
        Search emails and return as DataFrame with enhanced sorting and filtering
        
        Args:
            options: Search options including query string and parameters
            
        Returns:
            DataFrame containing email data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"search_emails called with loading_strategy: {options.loading_strategy}")
        logger.info(f"loading_strategy type: {type(options.loading_strategy)}")
        
        if not self.service:
            raise ValueError("Gmail service not initialized. Must authenticate first.")
            
        # Build complete query from options
        query = options.build_query()
        
        # Set up ordering based on loading strategy
        ordering = None
        if options.loading_strategy == LoadingStrategy.NEWER_FIRST:
            ordering = "NEWEST_FIRST"  # Default Gmail API behavior
        elif options.loading_strategy == LoadingStrategy.OLDER_FIRST:
            ordering = "OLDEST_FIRST"
        
        # Execute search with appropriate parameters
        
        params = {
            'userId': 'me',
            'q': query,
            'maxResults': options.max_results
        }
        #if ordering:
            #params['orderBy'] = ordering
        
        results = self.service.users().messages().list(**params).execute()
        messages = results.get('messages', [])
        
        # Handle IMPORTANT_FIRST strategy through post-processing
        if options.loading_strategy == LoadingStrategy.IMPORTANT_FIRST:
            # We'll need to get message data first to check importance
            # This is a simplified implementation
            message_ids = [msg['id'] for msg in messages]
            emails_data = []
            
            for msg_id in message_ids:
                email_data = self._get_message_data(
                    msg_id, 
                    include_attachments=options.include_attachments
                )
                emails_data.append(email_data)
            
            # Sort by whether the email has IMPORTANT label
            emails_df = pd.DataFrame(emails_data)
            if not emails_df.empty:
                emails_df['is_important'] = emails_df['label_ids'].apply(
                    lambda labels: 'IMPORTANT' in (labels or [])
                )
                emails_df = emails_df.sort_values(by='is_important', ascending=False)
            
            return emails_df
        
        # For normal ordering strategies, process as usual
        emails_data = []
        for message in messages:
            email_data = self._get_message_data(
                message['id'], 
                include_attachments=options.include_attachments
            )
            emails_data.append(email_data)
            
        return pd.DataFrame(emails_data)
