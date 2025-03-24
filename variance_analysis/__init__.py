"""
Document Variance Analysis Module

This package provides tools for analyzing variances between multiple documents
on the same topic, helping users identify inconsistencies in source material.
"""

from .document_variance_analyzer import DocumentVarianceAnalyzer, DocumentVariance, TopicVarianceAnalysis
from .variance_formatter import VarianceFormatter, VarianceFormatStyle
from .variance_retriever import VarianceDocumentRetriever

__version__ = "0.1.0"

__all__ = [
    'DocumentVarianceAnalyzer',
    'DocumentVariance',
    'TopicVarianceAnalysis',
    'VarianceFormatter',
    'VarianceFormatStyle',
    'VarianceDocumentRetriever'
]
