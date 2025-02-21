# format_mapper.py
from typing import Dict, List, Optional
from pydantic import BaseModel
from enum import Enum

class FormatStyle(Enum):
    WARNING_FIRST = "warning_first"
    STEP_BY_STEP = "step_by_step"
    SPECIFICATION = "specification"
    NARRATIVE = "narrative"

# [Previous FormatMapper code here]

# query_intent.py
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel

class QueryIntent(Enum):
    INSTRUCTION = "instruction"
    INFORMATION = "information"
    CLARIFICATION = "clarification"
    DISCUSSION = "discussion"

class QueryIntentAnalyzer:
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.INSTRUCTION: [
                "how do i",
                "what should i",
                "proper way",
                "steps to",
                "guide me"
            ],
            QueryIntent.INFORMATION: [
                "what is",
                "explain",
                "tell me about",
                "describe"
            ],
            QueryIntent.CLARIFICATION: [
                "quick question",
                "just checking",
                "verify",
                "confirm"
            ],
            QueryIntent.DISCUSSION: [
                "curious about",
                "wonder why",
                "interested in",
                "thoughts on"
            ]
        }
        
    def analyze(self, query: str, context: Dict) -> QueryIntent:
        query_lower = query.lower()
        
        # Check context for conversation history
        is_followup = context.get('is_followup', False)
        prev_queries = context.get('previousQueries', [])
        
        # If it's a follow-up question, likely continuation of previous intent
        if is_followup and prev_queries:
            return self._analyze_followup(query_lower, prev_queries)
            
        # Match against patterns
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
                
        # Analyze query structure
        return self._analyze_structure(query_lower)
        
    def _analyze_followup(self, query: str, prev_queries: List[str]) -> QueryIntent:
        # Logic for determining intent of follow-up questions
        pass
        
    def _analyze_structure(self, query: str) -> QueryIntent:
        # Fallback analysis based on question structure
        pass

# Modified NLPTransformer (in main file)
class NLPTransformer:
    def __init__(self):
        self.format_mapper = FormatMapper()
        self.intent_analyzer = QueryIntentAnalyzer()
        
    async def transform_content(
        self, 
        query: str,
        raw_response: str,
        context: Dict,
        source_documents: List,
        classification: Dict
    ) -> str:
        # Get intent
        intent = self.intent_analyzer.analyze(query, context)
        
        # Get category format
        category_format = self.format_mapper.get_format_for_category(
            classification['category']
        )
        
        # If conversational intent, use simpler format
        if intent in [QueryIntent.DISCUSSION, QueryIntent.CLARIFICATION]:
            return self._format_conversational(
                raw_response,
                classification['category']
            )
            
        # Otherwise use structured format
        return self._format_structured(
            raw_response,
            category_format,
            classification
        )
        
    def _format_conversational(self, content: str, category: str) -> str:
        """Format content in a conversational style."""
        # Add minimal formatting while keeping it casual
        if category == "SAFETY":
            # Still include critical safety info but in a lighter way
            return f"Just so you know: {content}"
        return content
        
    def _format_structured(
        self,
        content: str,
        format_spec: CategoryFormat,
        classification: Dict
    ) -> str:
        """Apply full structured formatting."""
        return self.format_mapper.apply_formatting(
            content,
            format_spec,
            classification
        )
