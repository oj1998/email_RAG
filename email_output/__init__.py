"""
Email Output package for intent-based response formatting.

This package provides components for detecting email query intents
and formatting responses appropriately for each intent type.
"""

from .email_intent import EmailIntent, EmailIntentDetector, EmailIntentAnalysis
from .email_formatter import FormatStyle, EmailFormatter, EmailResponse, EmailSource
from .enhanced_email_formatter import EmailFormatStyle, EnhancedEmailFormatter, EmailCategoryFormat
from .email_output_adapter import EmailOutputManager, enhanced_process_email_query

__all__ = [
    'EmailIntent',
    'EmailIntentDetector',
    'EmailIntentAnalysis',
    'FormatStyle',
    'EmailFormatter',
    'EmailFormatStyle',
    'EnhancedEmailFormatter',
    'EmailResponse',
    'EmailSource',
    'EmailCategoryFormat',
    'EmailOutputManager',
    'enhanced_process_email_query'
]
