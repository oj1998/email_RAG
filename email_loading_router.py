from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os

# Import local components
from client.gmail_client import GmailClient, EmailSearchOptions, LoadingStrategy
from loaders.simple_pipeline import SimpleEmailPipeline

logger = logging.getLogger(__name__)

# Models for request/response
class EmailLoadingConfig(BaseModel):
    # General settings
    load_amount: int = Field(100, ge=1, le=1000, description="Number of emails to load")
    loading_strategy: str = "newer-first"
    include_attachments: bool = False
    
    # Filters
    subject: Optional[str] = None
    from_email: Optional[str] = None
    to_email: Optional[str] = None
    after_date: Optional[str] = None
    before_date: Optional[str] = None
    has_label: Optional[str] = None
    
    # Advanced
    chunk_size: int = Field(500, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Dict[str, Any]
    created_at: str
    updated_at: str

# Global job status tracker
email_loading_jobs = {}

# Dependencies
def get_gmail_client() -> GmailClient:
    """Get authenticated Gmail client from global state"""
    from main import GMAIL_CLIENTS  # Import here to avoid circular imports
    
    if 'user' not in GMAIL_CLIENTS:
        raise HTTPException(
            status_code=401, 
            detail="Gmail client not authenticated. Please authenticate with Gmail first."
        )
    return GMAIL_CLIENTS['user']

def get_embeddings_model():
    """Get embeddings model"""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()

def get_vector_store():
    """Get vector store"""
    from main import vector_store  # Import from main app state
    
    if vector_store is None:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. Please wait for system startup to complete."
        )
    return vector_store

# Progress handler function
def handle_progress_update(job_id: str, progress_data: Dict[str, Any]):
    """Update job status with progress information"""
    if job_id in email_loading_jobs:
        email_loading_jobs[job_id]['progress'] = progress_data
        email_loading_jobs[job_id]['status'] = progress_data.get('stage', 'running')
        email_loading_jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
        logger.info(f"Job {job_id} progress: {progress_data['stage']} - {progress_data['percentage']}%")

# Background task to run email processing
async def process_emails_task(
    job_id: str, 
    config: EmailLoadingConfig, 
    gmail_client: GmailClient,
    embeddings_model,
    vector_store
):
    """Background task to process emails"""
    try:
        # Create search options from config
        search_options = EmailSearchOptions(
            max_results=config.load_amount,
            include_attachments=config.include_attachments,
            subject=config.subject,
            from_email=config.from_email,
            to_email=config.to_email,
            after_date=config.after_date,
            before_date=config.before_date,
            has_label=config.has_label,
            loading_strategy=config.loading_strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Initialize pipeline
        pipeline = SimpleEmailPipeline(embeddings_model, vector_store)
        
        # Register progress callback
        pipeline.register_progress_callback(
            lambda progress: handle_progress_update(job_id, progress)
        )
        
        # Run ingestion
        result = pipeline.ingest_emails(gmail_client, search_options)
        
        # Update job status with completion info
        email_loading_jobs[job_id]['status'] = 'completed'
        email_loading_jobs[job_id]['result'] = result
        email_loading_jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Email loading job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in email loading job {job_id}: {str(e)}")
        # Update job status with error info
        email_loading_jobs[job_id]['status'] = 'failed'
        email_loading_jobs[job_id]['error'] = str(e)
        email_loading_jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()

def create_email_loading_router():
    """Create and return a router for email loading endpoints"""
    router = APIRouter(prefix="/api/emails", tags=["email-loading"])
    
    @router.post("/load", response_model=JobStatus)
    async def start_email_loading(
        config: EmailLoadingConfig, 
        background_tasks: BackgroundTasks,
        gmail_client: GmailClient = Depends(get_gmail_client),
    ):
        """Start the email loading process with the given configuration"""
        # Generate a unique job ID
        job_id = f"email_load_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize job status
        email_loading_jobs[job_id] = {
            'job_id': job_id,
            'status': 'initializing',
            'progress': {},
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Get dependencies
        embeddings_model = get_embeddings_model()
        vector_store = get_vector_store()
        
        # Start background task
        background_tasks.add_task(
            process_emails_task, 
            job_id=job_id, 
            config=config, 
            gmail_client=gmail_client,
            embeddings_model=embeddings_model,
            vector_store=vector_store
        )
        
        logger.info(f"Started email loading job {job_id}")
        return email_loading_jobs[job_id]

    @router.get("/load/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        """Get the status of an email loading job"""
        if job_id not in email_loading_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return email_loading_jobs[job_id]

    @router.get("/load", response_model=List[JobStatus])
    async def list_jobs():
        """List all email loading jobs"""
        return list(email_loading_jobs.values())
    
    @router.get("/stats")
    async def get_email_stats():
        """Get statistics about the email vector store"""
        try:
            # Create temporary pipeline to access stats
            embeddings_model = get_embeddings_model()
            vector_store = get_vector_store()
            
            pipeline = SimpleEmailPipeline(embeddings_model, vector_store)
            stats = pipeline.get_stats()
            
            return {
                "status": "success",
                "email_stats": stats,
                "jobs": {
                    "total": len(email_loading_jobs),
                    "completed": sum(1 for job in email_loading_jobs.values() if job.get('status') == 'completed'),
                    "failed": sum(1 for job in email_loading_jobs.values() if job.get('status') == 'failed'),
                    "in_progress": sum(1 for job in email_loading_jobs.values() if job.get('status') not in ['completed', 'failed'])
                }
            }
        except Exception as e:
            logger.error(f"Error retrieving email stats: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve email stats: {str(e)}")
    
    return router
