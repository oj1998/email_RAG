
import base64
import os.path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd

@dataclass
class EmailSearchOptions:
    """Search parameters for Gmail API"""
    query: str = None  # Gmail search query
    max_results: int = 100
    include_attachments: bool = False
    
class GmailClient:
    """Client for interacting with Gmail API"""
    
    def __init__(self, credentials_path: str = None):
    self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    self.credentials = os.getenv('GOOGLE_CREDENTIALS')
    if self.credentials:
        # Write environment variable content to temporary file
        with open('temp_credentials.json', 'w') as f:
            f.write(self.credentials)
        self.credentials_path = 'temp_credentials.json'
    else:
        self.credentials_path = credentials_path
    self.service = self._get_gmail_service()

    def _get_gmail_service(self):
        """Get authenticated Gmail service"""
        creds = None
        # Token.json stores user access/refresh tokens
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
            
        # If no valid credentials, let user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save credentials for future runs
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

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

    def search_emails(self, options: EmailSearchOptions) -> pd.DataFrame:
        """
        Search emails and return as DataFrame
        
        Args:
            options: Search options including query string and parameters
            
        Returns:
            DataFrame containing email data with columns:
            - id: Email ID
            - thread_id: Conversation thread ID
            - subject: Email subject
            - from: Sender
            - to: Recipients
            - date: Date sent
            - body: Email content
            - label_ids: Gmail labels
            - attachments: List of attachment metadata (if requested)
        """
        # Execute search
        query = options.query if options.query else "in:anywhere"
        results = self.service.users().messages().list(
            userId='me', 
            q=query,
            maxResults=options.max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        # Get full data for each message
        emails_data = []
        for message in messages:
            email_data = self._get_message_data(
                message['id'], 
                include_attachments=options.include_attachments
            )
            emails_data.append(email_data)
            
        return pd.DataFrame(emails_data)
