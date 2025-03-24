from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
import logging
import asyncio
import re

logger = logging.getLogger(__name__)

class DocumentVariance(BaseModel):
    """Represents a variance between documents on the same topic"""
    topic: str
    aspect: str  # The specific aspect where variance was found (e.g., "installation method", "safety requirements")
    variance_description: str  # Description of how sources differ
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence in the identified variance
    source_positions: Dict[str, List[str]]  # Document IDs mapped to their positions on this variance
    source_excerpts: Dict[str, str]  # Relevant excerpts from each source
    
class TopicVarianceAnalysis(BaseModel):
    """Complete analysis of variances for a topic"""
    topic: str
    key_variances: List[DocumentVariance]
    agreement_points: List[str]  # Points where sources agree
    general_assessment: str  # Overall assessment of source agreement/disagreement
    source_count: int
    reliability_ranking: Dict[str, float]  # Source ID to reliability score
    
class DocumentVarianceAnalyzer:
    """Analyzes variances between documents on the same topic"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        min_confidence: float = 0.6,
        embeddings_model = None
    ):
        self.llm = llm
        self.min_confidence = min_confidence
        self.embeddings_model = embeddings_model
        
        # Initialize prompts for variance detection
        self.topic_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analysis expert. Extract the main topic that the user wants to analyze 
            variances for. Return ONLY the topic name without any explanation or additional text."""),
            ("user", "Query: {query}")
        ])
        
        self.aspect_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a construction document analyst. From the documents provided about a single topic,
            identify key aspects where sources might have differences. For example: 'installation methods',
            'material specifications', 'safety requirements', etc.
            
            List only the aspect names, one per line, without numbering or explanations."""),
            ("user", "Topic: {topic}\n\nDocument Excerpts:\n{excerpts}")
        ])
        
        self.variance_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a construction document analyst. Analyze how different sources describe the same aspect 
            of a topic. Identify specific variances, inconsistencies, or conflicts between sources.
            
            Respond with a JSON object containing:
            {
                "variance_exists": true/false,
                "variance_description": "detailed description of the variance",
                "confidence": 0.0-1.0 (how confident you are this is a real variance, not just different wording),
                "source_positions": {
                    "doc_id1": ["position_description"],
                    "doc_id2": ["position_description"]
                },
                "agreement_points": ["points where sources agree"]
            }"""),
            ("user", """Topic: {topic}
            Aspect: {aspect}
            
            Source Documents:
            {source_docs}""")
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a construction document analyst. Create a comprehensive summary of the 
            variances detected between different sources on a topic. Include an overall assessment of how
            significant the differences are and which sources seem most reliable based on specificity,
            comprehensiveness, and currency of information.
            
            Provide your analysis in a professional, balanced tone that helps users understand 
            the differences without indicating which source is definitively correct."""),
            ("user", """Topic: {topic}
            
            Detected Variances:
            {variances}
            
            Agreement Points:
            {agreements}
            
            Source Information:
            {source_info}""")
        ])
        
    async def extract_main_topic(self, query: str) -> str:
        """Extract the main topic from the user's query"""
        response = await self.llm.ainvoke(
            self.topic_extraction_prompt.format_messages(query=query)
        )
        topic = response.content.strip()
        logger.info(f"Extracted topic: {topic}")
        return topic
        
    async def identify_key_aspects(self, topic: str, documents: List[Document]) -> List[str]:
        """Identify key aspects of the topic for variance analysis"""
        # Create a condensed version of documents for the prompt
        excerpts = "\n\n".join([
            f"Document {i+1} ({doc.metadata.get('title', 'Untitled')}): {doc.page_content[:300]}..."
            for i, doc in enumerate(documents)
        ])
        
        response = await self.llm.ainvoke(
            self.aspect_extraction_prompt.format_messages(
                topic=topic,
                excerpts=excerpts
            )
        )
        
        # Parse aspects (one per line)
        aspects = [
            aspect.strip() 
            for aspect in response.content.strip().split("\n") 
            if aspect.strip()
        ]
        
        logger.info(f"Identified {len(aspects)} key aspects for topic '{topic}': {aspects}")
        return aspects
        
    async def analyze_aspect_variance(
        self, 
        topic: str,
        aspect: str, 
        documents: List[Document]
    ) -> Tuple[Optional[DocumentVariance], List[str]]:
        """Analyze variance for a specific aspect across documents"""
        # Format documents for analysis
        formatted_docs = "\n\n".join([
            f"Document ID: {doc.metadata.get('document_id', f'doc_{i}')}\n"
            f"Title: {doc.metadata.get('title', 'Untitled')}\n"
            f"Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Analyze variance
        response = await self.llm.ainvoke(
            self.variance_analysis_prompt.format_messages(
                topic=topic,
                aspect=aspect,
                source_docs=formatted_docs
            )
        )
        
        try:
            # Parse JSON response
            import json
            variance_data = json.loads(response.content)
            
            # If no variance exists, return agreement points only
            if not variance_data.get("variance_exists", False):
                return None, variance_data.get("agreement_points", [])
                
            # Create DocumentVariance object
            source_excerpts = {}
            for doc in documents:
                doc_id = doc.metadata.get('document_id', 'unknown')
                # Extract a relevant excerpt (simple implementation)
                excerpt = self._extract_relevant_excerpt(doc.page_content, aspect, 200)
                if excerpt:
                    source_excerpts[doc_id] = excerpt
            
            variance = DocumentVariance(
                topic=topic,
                aspect=aspect,
                variance_description=variance_data.get("variance_description", ""),
                confidence=variance_data.get("confidence", 0.7),
                source_positions=variance_data.get("source_positions", {}),
                source_excerpts=source_excerpts
            )
            
            return variance, variance_data.get("agreement_points", [])
            
        except Exception as e:
            logger.error(f"Error parsing variance analysis result: {e}")
            # Return a basic variance in case of parsing error
            return None, []
            
    def _extract_relevant_excerpt(self, text: str, keyword: str, max_length: int = 200) -> str:
        """Extract the most relevant excerpt from text based on a keyword"""
        lower_text = text.lower()
        lower_keyword = keyword.lower()
        
        # Try to find the keyword in the text
        if lower_keyword in lower_text:
            keyword_index = lower_text.index(lower_keyword)
            
            # Calculate start and end positions to get context around the keyword
            start = max(0, keyword_index - max_length // 2)
            end = min(len(text), keyword_index + max_length // 2)
            
            # Adjust to try to capture complete sentences
            while start > 0 and text[start] not in ".!?\n":
                start -= 1
            while end < len(text) - 1 and text[end] not in ".!?\n":
                end += 1
                
            # Extract excerpt and ensure it's not too long
            excerpt = text[start:end+1].strip()
            if len(excerpt) > max_length:
                excerpt = excerpt[:max_length] + "..."
                
            return excerpt
        
        # Fallback to first max_length characters if keyword not found
        return text[:max_length] + "..." if len(text) > max_length else text
        
    async def analyze_topic_variances(
        self, 
        query: str, 
        documents: List[Document]
    ) -> TopicVarianceAnalysis:
        """Complete analysis of variances for a topic across multiple documents"""
        if not documents or len(documents) < 2:
            raise ValueError("At least two documents are required for variance analysis")
            
        # Extract the main topic
        topic = await self.extract_main_topic(query)
        
        # Identify key aspects for analysis
        aspects = await self.identify_key_aspects(topic, documents)
        
        # Analyze each aspect for variances
        variances = []
        all_agreement_points = []
        
        for aspect in aspects:
            variance, agreement_points = await self.analyze_aspect_variance(topic, aspect, documents)
            if variance and variance.confidence >= self.min_confidence:
                variances.append(variance)
            all_agreement_points.extend(agreement_points)
        
        # Calculate source reliability based on metadata and variance analysis
        reliability_scores = self._calculate_reliability_scores(documents, variances)
        
        # Generate the overall assessment
        general_assessment = await self._generate_assessment_summary(
            topic, 
            variances, 
            all_agreement_points, 
            documents
        )
        
        return TopicVarianceAnalysis(
            topic=topic,
            key_variances=variances,
            agreement_points=all_agreement_points,
            general_assessment=general_assessment,
            source_count=len(documents),
            reliability_ranking=reliability_scores
        )
        
    def _calculate_reliability_scores(
        self, 
        documents: List[Document], 
        variances: List[DocumentVariance]
    ) -> Dict[str, float]:
        """Calculate reliability scores for each source"""
        scores = {}
        
        # Initialize base scores using metadata
        for doc in documents:
            doc_id = doc.metadata.get('document_id', 'unknown')
            
            # Base score starts at 0.7
            score = 0.7
            
            # Adjust based on metadata
            # More recent documents get higher scores
            if 'creation_date' in doc.metadata:
                try:
                    from datetime import datetime
                    creation_date = doc.metadata['creation_date']
                    if isinstance(creation_date, str):
                        # Simple heuristic - more recent documents score higher
                        if '2023' in creation_date or '2024' in creation_date:
                            score += 0.1
                        elif '2020' in creation_date or '2021' in creation_date or '2022' in creation_date:
                            score += 0.05
                except:
                    pass
                
            # Official documents score higher
            doc_type = doc.metadata.get('document_type', '').lower()
            if 'official' in doc_type or 'standard' in doc_type or 'code' in doc_type:
                score += 0.1
                
            # Document with more comprehensive content scores higher (simple length heuristic)
            content_length = len(doc.page_content)
            if content_length > 5000:
                score += 0.05
                
            scores[doc_id] = min(1.0, score)  # Cap at 1.0
            
        # Adjust scores based on variance analysis (if a source is frequently an outlier, reduce score)
        for variance in variances:
            source_positions = variance.source_positions
            if len(source_positions) > 1:
                # Find outlier sources (sources that differ from the majority)
                majority_view = None
                position_counts = {}
                
                # Count positions
                for _, positions in source_positions.items():
                    for pos in positions:
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                
                # Find majority view
                if position_counts:
                    majority_view = max(position_counts.items(), key=lambda x: x[1])[0]
                
                # Adjust scores for outliers
                if majority_view:
                    for doc_id, positions in source_positions.items():
                        if doc_id in scores and not any(majority_view in pos for pos in positions):
                            # Penalize sources that disagree with majority view
                            scores[doc_id] -= 0.05
        
        # Normalize scores to keep them between 0 and 1
        return {doc_id: max(0.1, min(1.0, score)) for doc_id, score in scores.items()}
    
    async def _generate_assessment_summary(
        self,
        topic: str,
        variances: List[DocumentVariance],
        agreement_points: List[str],
        documents: List[Document]
    ) -> str:
        """Generate an overall assessment summary"""
        # Format variances for the prompt
        variance_text = "\n\n".join([
            f"Aspect: {v.aspect}\n"
            f"Description: {v.variance_description}\n"
            f"Confidence: {v.confidence}\n"
            f"Source Positions: {v.source_positions}"
            for v in variances
        ]) if variances else "No significant variances detected."
        
        # Format agreement points
        agreement_text = "\n- ".join([f"{point}" for point in agreement_points])
        if agreement_text:
            agreement_text = "- " + agreement_text
        else:
            agreement_text = "No clear agreement points identified."
        
        # Format source information
        source_info = "\n\n".join([
            f"Document ID: {doc.metadata.get('document_id', f'doc_{i}')}\n"
            f"Title: {doc.metadata.get('title', 'Untitled')}\n"
            f"Type: {doc.metadata.get('document_type', 'Unknown')}\n"
            f"Date: {doc.metadata.get('creation_date', 'Unknown')}"
            for i, doc in enumerate(documents)
        ])
        
        # Generate summary
        response = await self.llm.ainvoke(
            self.summary_prompt.format_messages(
                topic=topic,
                variances=variance_text,
                agreements=agreement_text,
                source_info=source_info
            )
        )
        
        return response.content.strip()
