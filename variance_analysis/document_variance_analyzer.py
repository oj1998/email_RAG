from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
import logging
import asyncio
import re
from datetime import datetime

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
        
        logger.info(f"Initialized DocumentVarianceAnalyzer with min_confidence={min_confidence}")
        
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
            logger.info(f"Extracting main topic from query: '{query}'")
            extraction_start = datetime.utcnow()
            
            response = await self.llm.ainvoke(
                self.topic_extraction_prompt.format_messages(query=query)
            )
            topic = response.content.strip()
            
            extraction_time = (datetime.utcnow() - extraction_start).total_seconds()
            logger.info(f"Extracted topic: '{topic}' in {extraction_time:.2f}s")
            return topic
        
    async def identify_key_aspects(self, topic: str, documents: List[Document]) -> List[str]:
        """Identify key aspects of the topic for variance analysis"""
        logger.info(f"Identifying key aspects for topic: '{topic}' using {len(documents)} documents")
        aspect_start = datetime.utcnow()
        
        # Create a condensed version of documents for the prompt
        excerpts = "\n\n".join([
            f"Document {i+1} ({doc.metadata.get('title', 'Untitled')}): {doc.page_content[:300]}..."
            for i, doc in enumerate(documents[:5])  # Use first 5 docs to avoid token limits
        ])
        
        logger.info(f"Created excerpts from {min(5, len(documents))} documents for aspect identification")
        
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
        
        aspect_time = (datetime.utcnow() - aspect_start).total_seconds()
        logger.info(f"Identified {len(aspects)} key aspects for topic '{topic}' in {aspect_time:.2f}s: {aspects}")
        return aspects

    async def analyze_aspect_variance(
            self, 
            topic: str,
            aspect: str, 
            documents: List[Document]
        ) -> Tuple[Optional[DocumentVariance], List[str]]:
            """Analyze variance for a specific aspect across documents"""
            logger.info(f"Analyzing variance for aspect '{aspect}' of topic '{topic}'")
            analysis_start = datetime.utcnow()
            
            # Format documents for analysis
            formatted_docs = "\n\n".join([
                f"Document ID: {doc.metadata.get('document_id', f'doc_{i}')}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Page: {doc.metadata.get('page', 'N/A')}\n"
                f"Content: {doc.page_content[:500]}..."  # Truncate to avoid token limits
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
                    agreement_points = variance_data.get("agreement_points", [])
                    logger.info(f"No variance detected for aspect '{aspect}'. Found {len(agreement_points)} agreement points.")
                    return None, agreement_points
                    
                # Create DocumentVariance object
                source_excerpts = {}
                for doc in documents:
                    doc_id = doc.metadata.get('document_id', 'unknown')
                    # Extract a relevant excerpt (simple implementation)
                    excerpt = self._extract_relevant_excerpt(doc.page_content, aspect, 200)
                    if excerpt:
                        source_excerpts[doc_id] = excerpt
                
                # Extract source positions and document IDs
                source_positions = variance_data.get("source_positions", {})
                confidence = variance_data.get("confidence", 0.7)
                
                logger.info(f"Detected variance for aspect '{aspect}' with confidence {confidence:.2f}")
                logger.info(f"Source positions: {list(source_positions.keys())}")
                
                variance = DocumentVariance(
                    topic=topic,
                    aspect=aspect,
                    variance_description=variance_data.get("variance_description", ""),
                    confidence=confidence,
                    source_positions=source_positions,
                    source_excerpts=source_excerpts
                )
                
                analysis_time = (datetime.utcnow() - analysis_start).total_seconds()
                logger.info(f"Completed variance analysis for aspect '{aspect}' in {analysis_time:.2f}s")
                
                return variance, variance_data.get("agreement_points", [])
                
            except Exception as e:
                logger.error(f"Error parsing variance analysis result for aspect '{aspect}': {e}", exc_info=True)
                analysis_time = (datetime.utcnow() - analysis_start).total_seconds()
                logger.warning(f"Variance analysis failed for aspect '{aspect}' after {analysis_time:.2f}s")
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
            analysis_start = datetime.utcnow()
            logger.info(f"Starting variance analysis for query: '{query}' with {len(documents)} documents")
            
            if not documents or len(documents) < 2:
                logger.error(f"Insufficient documents ({len(documents) if documents else 0}) for variance analysis")
                raise ValueError("At least two documents are required for variance analysis")
                
            # Extract the main topic
            topic = await self.extract_main_topic(query)
            
            # Identify key aspects for analysis
            aspect_start = datetime.utcnow()
            aspects = await self.identify_key_aspects(topic, documents)
            aspect_time = (datetime.utcnow() - aspect_start).total_seconds()
            logger.info(f"Aspect identification completed in {aspect_time:.2f}s")
            
            # Analyze each aspect for variances
            variance_start = datetime.utcnow()
            variances = []
            all_agreement_points = []
            
            logger.info(f"Analyzing {len(aspects)} aspects for variances...")
            
            for i, aspect in enumerate(aspects):
                logger.info(f"Analyzing aspect {i+1}/{len(aspects)}: '{aspect}'")
                variance, agreement_points = await self.analyze_aspect_variance(topic, aspect, documents)
                
                if variance and variance.confidence >= self.min_confidence:
                    variances.append(variance)
                    logger.info(f"✓ Found variance for aspect '{aspect}' with confidence {variance.confidence:.2f}")
                else:
                    logger.info(f"✗ No significant variance detected for aspect '{aspect}'")
                    
                if agreement_points:
                    logger.info(f"Found {len(agreement_points)} agreement points for aspect '{aspect}'")
                    all_agreement_points.extend(agreement_points)
            
            variance_time = (datetime.utcnow() - variance_start).total_seconds()
            logger.info(f"Analyzed {len(aspects)} aspects in {variance_time:.2f}s, found {len(variances)} variances")
            
            # Calculate source reliability based on metadata and variance analysis
            reliability_start = datetime.utcnow()
            logger.info("Calculating source reliability scores...")
            reliability_scores = self._calculate_reliability_scores(documents, variances)
            
            # Log reliability results
            if reliability_scores:
                sorted_scores = sorted(reliability_scores.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Reliability scores: {sorted_scores}")
                if sorted_scores:
                    logger.info(f"Most reliable source: {sorted_scores[0][0]} ({sorted_scores[0][1]:.2f})")
                    if len(sorted_scores) > 1:
                        logger.info(f"Least reliable source: {sorted_scores[-1][0]} ({sorted_scores[-1][1]:.2f})")
            
            reliability_time = (datetime.utcnow() - reliability_start).total_seconds()
            logger.info(f"Reliability calculation completed in {reliability_time:.2f}s")
            
            # Generate the overall assessment
            summary_start = datetime.utcnow()
            logger.info("Generating assessment summary...")
            general_assessment = await self._generate_assessment_summary(
                topic, 
                variances, 
                all_agreement_points, 
                documents
            )
            summary_time = (datetime.utcnow() - summary_start).total_seconds()
            logger.info(f"Summary generation completed in {summary_time:.2f}s")
            
            # Create the final analysis object
            result = TopicVarianceAnalysis(
                topic=topic,
                key_variances=variances,
                agreement_points=all_agreement_points,
                general_assessment=general_assessment,
                source_count=len(documents),
                reliability_ranking=reliability_scores
            )
            
            analysis_time = (datetime.utcnow() - analysis_start).total_seconds()
            logger.info(f"Total variance analysis completed in {analysis_time:.2f}s")
            logger.info(f"Results: {len(variances)} variances, {len(all_agreement_points)} agreement points across {len(documents)} sources")
            
            return result
    
    def _calculate_reliability_scores(
            self, 
            documents: List[Document], 
            variances: List[DocumentVariance]
        ) -> Dict[str, float]:
            """Calculate reliability scores for each source"""
            logger.info(f"Calculating reliability scores for {len(documents)} documents")
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
                                logger.debug(f"Document {doc_id}: +0.1 for recent creation date ({creation_date})")
                            elif '2020' in creation_date or '2021' in creation_date or '2022' in creation_date:
                                score += 0.05
                                logger.debug(f"Document {doc_id}: +0.05 for moderately recent creation date ({creation_date})")
                    except Exception as e:
                        logger.debug(f"Error processing creation date for document {doc_id}: {e}")
                    
                # Official documents score higher
                doc_type = doc.metadata.get('document_type', '').lower()
                if 'official' in doc_type or 'standard' in doc_type or 'code' in doc_type:
                    score += 0.1
                    logger.debug(f"Document {doc_id}: +0.1 for official document type ({doc_type})")
                    
                # Document with more comprehensive content scores higher (simple length heuristic)
                content_length = len(doc.page_content)
                if content_length > 5000:
                    score += 0.05
                    logger.debug(f"Document {doc_id}: +0.05 for comprehensive content ({content_length} chars)")
                    
                scores[doc_id] = min(1.0, score)  # Cap at 1.0
                logger.debug(f"Document {doc_id}: Initial reliability score: {scores[doc_id]:.2f}")
                
            # Adjust scores based on variance analysis (if a source is frequently an outlier, reduce score)
            if variances:
                logger.info("Adjusting reliability scores based on variance analysis")
                
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
                        logger.debug(f"Aspect '{variance.aspect}': Majority view is '{majority_view}'")
                    
                    # Adjust scores for outliers
                    if majority_view:
                        for doc_id, positions in source_positions.items():
                            if doc_id in scores and not any(majority_view in pos for pos in positions):
                                # Penalize sources that disagree with majority view
                                original_score = scores[doc_id]
                                scores[doc_id] -= 0.05
                                logger.debug(f"Document {doc_id}: -0.05 for disagreeing with majority on '{variance.aspect}'")
                                logger.debug(f"Document {doc_id}: Score adjusted from {original_score:.2f} to {scores[doc_id]:.2f}")
            
            # Normalize scores to keep them between 0 and 1
            normalized_scores = {doc_id: max(0.1, min(1.0, score)) for doc_id, score in scores.items()}
            logger.info(f"Final reliability scores: {', '.join([f'{k}={v:.2f}' for k, v in normalized_scores.items()])}")
            
            return normalized_scores
        
        async def _generate_assessment_summary(
            self,
            topic: str,
            variances: List[DocumentVariance],
            agreement_points: List[str],
            documents: List[Document]
        ) -> str:
            """Generate an overall assessment summary"""
            logger.info(f"Generating assessment summary for topic '{topic}'")
            summary_start = datetime.utcnow()
            
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
            
            logger.info(f"Formatted input for summary generation: {len(variance_text)} chars of variance data, " +
                        f"{len(agreement_text)} chars of agreement data, {len(source_info)} chars of source info")
            
            # Generate summary
            response = await self.llm.ainvoke(
                self.summary_prompt.format_messages(
                    topic=topic,
                    variances=variance_text,
                    agreements=agreement_text,
                    source_info=source_info
                )
            )
            
            result = response.content.strip()
            summary_time = (datetime.utcnow() - summary_start).total_seconds()
            
            logger.info(f"Generated assessment summary ({len(result)} chars) in {summary_time:.2f}s")
            logger.debug(f"Summary preview: {result[:100]}...")
            
            return result
