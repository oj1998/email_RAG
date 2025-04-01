"""
Knowledge gap detection and tracking module for construction assistant.
This module provides functionality to detect, log, and analyze areas where
the system cannot provide reliable answers.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# Import these from your existing application
from construction_classifier import QuestionType

# Set up logging
logger = logging.getLogger(__name__)

# Models
class KnowledgeGap(BaseModel):
    """Information about detected knowledge gaps"""
    is_gap: bool
    gap_type: Optional[str] = None
    domain: Optional[str] = None
    subcategory: Optional[str] = None
    confidence: float = 1.0
    recommended_action: Optional[str] = None


class KnowledgeGapDetector:
    """Detects and analyzes knowledge gaps in construction queries"""
    
    def __init__(self, pool=None, min_source_count: int = 2, 
                 min_source_confidence: float = 0.65,
                 min_classification_confidence: float = 0.7):
        """
        Initialize the knowledge gap detector
        
        Args:
            pool: Database connection pool
            min_source_count: Minimum number of sources required
            min_source_confidence: Minimum confidence in sources
            min_classification_confidence: Minimum classification confidence
        """
        self.pool = pool
        self.MIN_SOURCE_COUNT = min_source_count
        self.MIN_SOURCE_CONFIDENCE = min_source_confidence
        self.MIN_CLASSIFICATION_CONFIDENCE = min_classification_confidence
        
        # Initialize domain keyword mappings
        self._init_domain_keywords()
        
    def _init_domain_keywords(self):
        """Initialize domain-specific keyword mappings"""
        self.domain_keywords = {
            "roofing": ["roof", "shingles", "flashing", "gutters", "eaves", "truss"],
            "electrical": ["wiring", "circuit", "breaker", "voltage", "conduit", "electrical"],
            "plumbing": ["pipe", "drain", "valve", "plumbing", "water", "sewer", "fixture"],
            "hvac": ["hvac", "heating", "cooling", "ventilation", "duct", "refrigerant"],
            "foundation": ["foundation", "footing", "slab", "basement", "concrete", "soil"],
            "framing": ["framing", "stud", "joist", "beam", "truss", "header"],
            "insulation": ["insulation", "r-value", "thermal", "vapor barrier", "moisture"],
            "drywall": ["drywall", "sheetrock", "gypsum", "joint compound", "mud", "taping"],
            "flooring": ["flooring", "tile", "hardwood", "laminate", "vinyl", "subfloor"],
            "painting": ["paint", "primer", "stain", "brush", "roller", "finish"],
            "masonry": ["brick", "stone", "mortar", "block", "masonry", "concrete"],
            "landscaping": ["landscaping", "grading", "drainage", "retaining wall", "soil"]
        }
        
        self.subcategory_mappings = {
            "roofing": {
                "shingles": ["shingle", "asphalt"],
                "metal": ["metal roof", "sheet metal"],
                "flat": ["flat roof", "membrane", "epdm", "tpo"],
                "tile": ["tile", "slate", "clay", "concrete tile"]
            },
            "electrical": {
                "wiring": ["wire", "cable", "romex"],
                "panels": ["panel", "breaker", "fuse"],
                "fixtures": ["fixture", "light", "fan"],
                "low_voltage": ["low voltage", "network", "smart", "automation"]
            }
            # Add mappings for other domains as needed
        }
        
    async def detect_gap(
        self,
        query: str,
        classification: QuestionType,
        source_documents: List,
        retrieved_content_confidence: float,
        context: Dict[str, Any] = None
    ) -> KnowledgeGap:
        """
        Detect knowledge gaps for queries that cannot be reliably answered.
        
        Args:
            query: The user's question
            classification: The question category classification
            source_documents: Retrieved documents for the query
            retrieved_content_confidence: Confidence in the retrieved content
            context: Additional context information
            
        Returns:
            KnowledgeGap object with detection results
        """
        # Initialize knowledge gap result
        knowledge_gap = KnowledgeGap(
            is_gap=False,
            confidence=1.0
        )
        
        # Check sufficient number of sources
        if len(source_documents) < self.MIN_SOURCE_COUNT:
            knowledge_gap.is_gap = True
            knowledge_gap.gap_type = "insufficient_sources"
            knowledge_gap.confidence = 0.9  # High confidence this is a gap
        
        # Check classification confidence
        if classification.confidence < self.MIN_CLASSIFICATION_CONFIDENCE:
            knowledge_gap.is_gap = True
            knowledge_gap.gap_type = "uncertain_classification"
            knowledge_gap.confidence = 0.8  # High confidence this is a gap
        
        # Check retrieval confidence
        if retrieved_content_confidence < self.MIN_SOURCE_CONFIDENCE:
            knowledge_gap.is_gap = True
            knowledge_gap.gap_type = "low_source_relevance"
            knowledge_gap.confidence = 0.85  # High confidence this is a gap
        
        # If a knowledge gap is detected, determine the domain
        if knowledge_gap.is_gap:
            # Determine specific domain and subcategory
            domain_info = await self.extract_domain_info(query, classification.category)
            knowledge_gap.domain = domain_info.get("domain")
            knowledge_gap.subcategory = domain_info.get("subcategory")
            
            # Log the knowledge gap
            if self.pool:
                await self.log_knowledge_gap(
                    query=query,
                    classification=classification.dict(),
                    knowledge_gap=knowledge_gap.dict(),
                    context=context
                )
            
            # Determine recommended action
            knowledge_gap.recommended_action = self.determine_action(knowledge_gap)
        
        return knowledge_gap

    async def extract_domain_info(self, query: str, category: str) -> Dict[str, str]:
        """
        Extract specific domain and subcategory from the query.
        
        Args:
            query: The user's question
            category: The high-level category
            
        Returns:
            Dict with domain and subcategory
        """
        # Look for domain keywords in the query
        query_lower = query.lower()
        matched_domain = None
        max_matches = 0
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                matched_domain = domain
        
        # Use LLM for more precise extraction if needed
        if not matched_domain or max_matches <= 1:
            try:
                llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
                response = await llm.ainvoke([
                    {"role": "system", "content": "You are a construction domain classifier. Extract the specific domain and subcategory from this construction query."},
                    {"role": "user", "content": f"Query: {query}\nBroad category: {category}\n\nRespond in JSON format with 'domain' and 'subcategory' fields."}
                ])
                
                # Extract JSON from response
                content = response.content
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    json_match = re.search(r'{.*}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(0)
                
                result = json.loads(content)
                return {
                    "domain": result.get("domain", "unspecified"),
                    "subcategory": result.get("subcategory", "general")
                }
            except Exception as e:
                logger.warning(f"Error in domain extraction: {e}")
                return {"domain": matched_domain or "unspecified", "subcategory": "general"}
        
        # Find subcategory if domain is matched
        subcategory = "general"
        if matched_domain and matched_domain in self.subcategory_mappings:
            for sub, sub_keywords in self.subcategory_mappings[matched_domain].items():
                if any(kw in query_lower for kw in sub_keywords):
                    subcategory = sub
                    break
        
        return {
            "domain": matched_domain or "unspecified",
            "subcategory": subcategory
        }

    async def log_knowledge_gap(
        self,
        query: str,
        classification: Dict,
        knowledge_gap: Dict,
        context: Optional[Dict] = None
    ) -> None:
        """
        Log knowledge gap information to database for analysis.
        
        Args:
            query: The user's question
            classification: The question classification information
            knowledge_gap: The knowledge gap information
            context: Additional context information
        """
        try:
            if not self.pool:
                logger.warning("Database pool not available, skipping knowledge gap logging")
                return
            
            # Insert the knowledge gap record
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO knowledge_gaps 
                    (query, category, gap_type, domain, subcategory, confidence, context, classification, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                """, 
                    query,
                    classification.get("category", "UNKNOWN"),
                    knowledge_gap.get("gap_type", "unknown"),
                    knowledge_gap.get("domain", "unspecified"),
                    knowledge_gap.get("subcategory", "general"),
                    knowledge_gap.get("confidence", 0.0),
                    json.dumps(context) if context else None,
                    json.dumps(classification)
                )
                
                logger.info(f"Logged knowledge gap: {knowledge_gap.get('domain')}/{knowledge_gap.get('subcategory')} - {knowledge_gap.get('gap_type')}")
                
        except Exception as e:
            logger.error(f"Error logging knowledge gap: {e}")

    def determine_action(self, knowledge_gap: KnowledgeGap) -> str:
        """
        Determine recommended action based on knowledge gap type.
        
        Args:
            knowledge_gap: The knowledge gap information
            
        Returns:
            String with recommended action
        """
        gap_type = knowledge_gap.gap_type
        domain = knowledge_gap.domain
        
        actions = {
            "insufficient_sources": f"Add more content about {domain}",
            "uncertain_classification": f"Improve classification for {domain} queries",
            "low_source_relevance": f"Enhance quality of {domain} content"
        }
        
        return actions.get(gap_type, "Review and analyze knowledge gap further")

    def calculate_retrieved_content_confidence(self, source_documents: List) -> float:
        """
        Calculate confidence score for retrieved content.
        
        Args:
            source_documents: Retrieved documents
            
        Returns:
            Float representing confidence in retrieved content
        """
        if not source_documents:
            return 0.0
            
        # Average the semantic similarities if available
        similarities = [doc.metadata.get('similarity', 0) for doc in source_documents if 'similarity' in doc.metadata]
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            # Fallback to moderate confidence if similarity scores aren't available
            return 0.6 if len(source_documents) >= 3 else 0.4


# API router for knowledge gap endpoints
def create_knowledge_gap_router(pool=None):
    """Create and configure FastAPI router for knowledge gap endpoints"""
    router = APIRouter(prefix="/knowledge-gaps", tags=["knowledge-gaps"])
    
    @router.get("/")
    async def get_knowledge_gaps(
        domain: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100
    ):
        """Get knowledge gaps for analysis"""
        try:
            if not pool:
                return {"status": "error", "message": "Database connection not available"}
                
            query_conditions = []
            query_params = []
            
            if domain:
                query_conditions.append("domain = $" + str(len(query_params) + 1))
                query_params.append(domain)
                
            if from_date:
                query_conditions.append("created_at >= $" + str(len(query_params) + 1))
                query_params.append(from_date)
                
            if to_date:
                query_conditions.append("created_at <= $" + str(len(query_params) + 1))
                query_params.append(to_date)
                
            where_clause = " WHERE " + " AND ".join(query_conditions) if query_conditions else ""
            
            async with pool.acquire() as conn:
                records = await conn.fetch(
                    f"""
                    SELECT * FROM knowledge_gaps
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT $""" + str(len(query_params) + 1),
                    *query_params, limit
                )
                
                results = []
                for record in records:
                    results.append({
                        "id": record["id"],
                        "query": record["query"],
                        "category": record["category"],
                        "gap_type": record["gap_type"],
                        "domain": record["domain"],
                        "subcategory": record["subcategory"],
                        "confidence": record["confidence"],
                        "created_at": record["created_at"].isoformat(),
                        "context": record["context"],
                        "classification": record["classification"]
                    })
                    
                return {
                    "status": "success",
                    "knowledge_gaps": results,
                    "count": len(results)
                }
                
        except Exception as e:
            logger.error(f"Error retrieving knowledge gaps: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/analytics")
    async def knowledge_gap_analytics(
        time_period: str = "week"  # Options: day, week, month, quarter, year
    ):
        """Get analytics about knowledge gaps"""
        try:
            if not pool:
                return {"status": "error", "message": "Database connection not available"}
                
            # Set time window based on parameter
            time_filters = {
                "day": "created_at > NOW() - INTERVAL '1 day'",
                "week": "created_at > NOW() - INTERVAL '7 days'",
                "month": "created_at > NOW() - INTERVAL '30 days'",
                "quarter": "created_at > NOW() - INTERVAL '90 days'",
                "year": "created_at > NOW() - INTERVAL '365 days'"
            }
            
            time_filter = time_filters.get(time_period, time_filters["month"])
            
            async with pool.acquire() as conn:
                # Get domain breakdown
                domain_breakdown = await conn.fetch(f"""
                    SELECT 
                        domain, 
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM knowledge_gaps
                    WHERE {time_filter}
                    GROUP BY domain
                    ORDER BY count DESC
                """)
                
                # Get gap type breakdown
                gap_type_breakdown = await conn.fetch(f"""
                    SELECT 
                        gap_type, 
                        COUNT(*) as count
                    FROM knowledge_gaps
                    WHERE {time_filter}
                    GROUP BY gap_type
                    ORDER BY count DESC
                """)
                
                # Get trend data
                trend_interval = {
                    "day": "hour",
                    "week": "day",
                    "month": "day",
                    "quarter": "week",
                    "year": "month"
                }.get(time_period, "day")
                
                trend_data = await conn.fetch(f"""
                    SELECT 
                        date_trunc('{trend_interval}', created_at) as time_period,
                        COUNT(*) as count
                    FROM knowledge_gaps
                    WHERE {time_filter}
                    GROUP BY time_period
                    ORDER BY time_period
                """)
                
                # Top queries
                top_queries = await conn.fetch(f"""
                    SELECT 
                        query,
                        domain,
                        subcategory,
                        COUNT(*) as frequency
                    FROM knowledge_gaps
                    WHERE {time_filter}
                    GROUP BY query, domain, subcategory
                    ORDER BY frequency DESC
                    LIMIT 10
                """)
                
                return {
                    "status": "success",
                    "time_period": time_period,
                    "domains": [
                        {"domain": record["domain"], "count": record["count"], "avg_confidence": record["avg_confidence"]}
                        for record in domain_breakdown
                    ],
                    "gap_types": [
                        {"type": record["gap_type"], "count": record["count"]}
                        for record in gap_type_breakdown
                    ],
                    "trends": [
                        {"period": record["time_period"].isoformat(), "count": record["count"]}
                        for record in trend_data
                    ],
                    "top_queries": [
                        {"query": record["query"], "domain": record["domain"], "subcategory": record["subcategory"], "frequency": record["frequency"]}
                        for record in top_queries
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error generating knowledge gap analytics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    return router


# Global detector instance for convenience
_detector = None

# Helper functions for simpler imports
def get_detector(pool=None):
    """Get the global knowledge gap detector instance"""
    global _detector
    if _detector is None:
        _detector = KnowledgeGapDetector(pool)
    return _detector

async def detect_knowledge_gap(
    query: str,
    classification,
    source_documents: List,
    context: Dict[str, Any] = None,
    pool=None
) -> KnowledgeGap:
    """
    Helper function to detect knowledge gaps.
    
    Args:
        query: The user's question
        classification: The question classification
        source_documents: Retrieved documents
        context: Additional context
        pool: Database pool (optional)
        
    Returns:
        KnowledgeGap object
    """
    detector = get_detector(pool)
    
    # Calculate confidence in retrieved content
    retrieved_content_confidence = detector.calculate_retrieved_content_confidence(source_documents)
    
    # Detect knowledge gap
    return await detector.detect_gap(
        query=query,
        classification=classification,
        source_documents=source_documents,
        retrieved_content_confidence=retrieved_content_confidence,
        context=context
    )
