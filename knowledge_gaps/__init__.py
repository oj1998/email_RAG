# Import and expose the main functionality from knowledge_gaps.py
from .knowledge_gaps import (
    detect_knowledge_gap,
    create_knowledge_gap_router,
    get_detector,
    KnowledgeGap,
    KnowledgeGapDetector
)

# This makes these functions and classes available directly from the package
__all__ = [
    'detect_knowledge_gap',
    'create_knowledge_gap_router',
    'get_detector',
    'KnowledgeGap',
    'KnowledgeGapDetector'
]
