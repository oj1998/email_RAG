from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
import logging
import json
import asyncio
import re

logger = logging.getLogger(__name__)

class ComparisonCriterion(BaseModel):
    """A specific criterion for comparison between topics"""
    name: str
    description: str
    importance: float = Field(default=0.5, ge=0, le=1)

class TopicAnalysis(BaseModel):
    """Analysis of a specific comparison topic"""
    topic: str
    strengths: List[str]
    weaknesses: List[str]
    key_attributes: Dict[str, Any]
    suitability_score: float = Field(default=0.5, ge=0, le=1)
    supporting_quotes: List[str] = []
    document_sources: List[str] = []

class ComparisonAnalysis(BaseModel):
    """Complete comparison analysis between multiple topics"""
    topics: List[str]
    criteria: List[ComparisonCriterion]
    topic_analyses: Dict[str, TopicAnalysis]
    overall_recommendation: str
    confidence: float = Field(default=0.5, ge=0, le=1)
    domain_context: str
    query_context: str

class ComparisonAnalyzer:
    """Analyzes documents to generate structured comparison analysis"""
    
    def __init__(
        self,
        llm: BaseChatModel
    ):
        self.llm = llm
        
    async def extract_comparison_criteria(
        self,
        query: str,
        domain_context: str,
        topics: List[str]
    ) -> List[ComparisonCriterion]:
        """Extract relevant comparison criteria from the query and domain"""
        prompt = f"""
        Identify the key criteria for comparing {" and ".join(topics)} in the construction domain.
        Consider these factors:
        1. The specific query: "{query}"
        2. The construction context: "{domain_context}"
        
        For each criterion, provide:
        - A short name (1-3 words)
        - A brief description
        - Importance (0.0-1.0 score)
        
        Return a JSON array of criteria objects with "name", "description", and "importance" keys.
        Provide 3-5 criteria that would be most relevant for this comparison.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from response
            json_str = self._extract_json_from_text(content)
            if not json_str:
                logger.warning("Couldn't extract JSON from criteria response")
                criteria_list = [
                    {"name": "Cost", "description": "Initial and long-term expenses", "importance": 0.8},
                    {"name": "Performance", "description": "How well it meets functional requirements", "importance": 0.9},
                    {"name": "Durability", "description": "Lifespan and maintenance needs", "importance": 0.7}
                ]
            else:
                criteria_list = json.loads(json_str)
            
            # Convert to ComparisonCriterion objects
            return [ComparisonCriterion(**criteria) for criteria in criteria_list]
        
        except Exception as e:
            logger.error(f"Error extracting comparison criteria: {e}")
            # Return default criteria as fallback
            return [
                ComparisonCriterion(name="Cost", description="Initial and long-term expenses", importance=0.8),
                ComparisonCriterion(name="Performance", description="How well it meets functional requirements", importance=0.9),
                ComparisonCriterion(name="Durability", description="Lifespan and maintenance needs", importance=0.7)
            ]
    
    async def analyze_topic(
        self,
        topic: str,
        documents: List[Document],
        criteria: List[ComparisonCriterion],
        query: str
    ) -> TopicAnalysis:
        """Analyze a topic based on relevant documents and criteria"""
        # Format document content
        doc_content = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:1000]}..." if len(doc.page_content) > 1000 else f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        # Format criteria
        criteria_str = "\n".join([
            f"- {c.name}: {c.description} (Importance: {c.importance})"
            for c in criteria
        ])
        
        # Create analysis prompt
        prompt = f"""
        Analyze "{topic}" based on the following documents and criteria.
        
        CONTEXT QUERY:
        {query}
        
        EVALUATION CRITERIA:
        {criteria_str}
        
        RELEVANT DOCUMENTS:
        {doc_content}
        
        Provide a structured analysis in JSON format with these keys:
        1. "strengths": List of strengths (3-5 points)
        2. "weaknesses": List of weaknesses (3-5 points)
        3. "key_attributes": Object with values for each criterion
        4. "suitability_score": Overall score from 0.0-1.0
        5. "supporting_quotes": List of 3-5 direct quotes from the documents
        
        Return ONLY the JSON object.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from response
            json_str = self._extract_json_from_text(content)
            if not json_str:
                logger.warning(f"Couldn't extract JSON from topic analysis for {topic}")
                return self._create_fallback_topic_analysis(topic, documents)
            
            analysis_data = json.loads(json_str)
            
            # Add document sources
            doc_sources = []
            for doc in documents:
                if "document_id" in doc.metadata:
                    doc_sources.append(doc.metadata["document_id"])
                elif "source_id" in doc.metadata:
                    doc_sources.append(doc.metadata["source_id"])
            
            # Create TopicAnalysis object
            return TopicAnalysis(
                topic=topic,
                strengths=analysis_data.get("strengths", []),
                weaknesses=analysis_data.get("weaknesses", []),
                key_attributes=analysis_data.get("key_attributes", {}),
                suitability_score=analysis_data.get("suitability_score", 0.5),
                supporting_quotes=analysis_data.get("supporting_quotes", []),
                document_sources=list(set(doc_sources))  # Remove duplicates
            )
        
        except Exception as e:
            logger.error(f"Error analyzing topic {topic}: {e}")
            return self._create_fallback_topic_analysis(topic, documents)
    
    async def generate_comparison(
        self,
        query: str,
        topics: List[str],
        topic_documents: Dict[str, List[Document]],
        domain_context: str = "construction industry"
    ) -> ComparisonAnalysis:
        """Generate a complete comparison analysis"""
        # Extract comparison criteria
        criteria = await self.extract_comparison_criteria(query, domain_context, topics)
        
        # Analyze each topic
        topic_analyses = {}
        for topic in topics:
            docs = topic_documents.get(topic, [])
            if not docs:
                logger.warning(f"No documents for topic: {topic}")
                continue
                
            analysis = await self.analyze_topic(topic, docs, criteria, query)
            topic_analyses[topic] = analysis
        
        # Generate overall recommendation
        overall_recommendation = await self._generate_recommendation(
            query, topics, topic_analyses, criteria
        )
        
        # Calculate confidence based on document quality and coverage
        confidence = self._calculate_confidence(topic_analyses, criteria)
        
        return ComparisonAnalysis(
            topics=topics,
            criteria=criteria,
            topic_analyses=topic_analyses,
            overall_recommendation=overall_recommendation,
            confidence=confidence,
            domain_context=domain_context,
            query_context=query
        )
    
    async def _generate_recommendation(
        self,
        query: str,
        topics: List[str],
        topic_analyses: Dict[str, TopicAnalysis],
        criteria: List[ComparisonCriterion]
    ) -> str:
        """Generate overall recommendation based on analyses"""
        # Format topic analyses for the LLM
        analyses_str = ""
        for topic, analysis in topic_analyses.items():
            analyses_str += f"\n\n--- {topic.upper()} ---\n"
            analyses_str += f"Strengths: {', '.join(analysis.strengths)}\n"
            analyses_str += f"Weaknesses: {', '.join(analysis.weaknesses)}\n"
            analyses_str += f"Suitability Score: {analysis.suitability_score}\n"
        
        prompt = f"""
        Based on the following comparison analysis between {" and ".join(topics)},
        provide a concise recommendation (2-3 sentences).
        
        Original query: "{query}"
        
        CRITERIA:
        {', '.join([c.name for c in criteria])}
        
        ANALYSES:
        {analyses_str}
        
        Consider both the strengths and weaknesses, as well as the context of the query.
        Provide a clear, justified recommendation.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            recommendation = response.content if hasattr(response, "content") else str(response)
            return recommendation.strip()
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            
            # Create a simple fallback recommendation
            best_topic = max(
                topic_analyses.items(),
                key=lambda x: x[1].suitability_score,
                default=(topics[0], None)
            )[0]
            
            return f"Based on the available information, {best_topic} appears to be more suitable for this application, but consider specific requirements and constraints for your particular situation."
    
    def _calculate_confidence(
        self,
        topic_analyses: Dict[str, TopicAnalysis],
        criteria: List[ComparisonCriterion]
    ) -> float:
        """Calculate confidence score for the comparison analysis"""
        if not topic_analyses:
            return 0.2  # Very low confidence if no analyses
        
        # Factors that affect confidence:
        # 1. Number of supporting quotes
        # 2. Document source coverage
        # 3. Number of criteria addressed
        
        avg_quotes = sum(len(a.supporting_quotes) for a in topic_analyses.values()) / len(topic_analyses)
        quote_factor = min(1.0, avg_quotes / 5)  # Normalize to 0-1
        
        avg_sources = sum(len(a.document_sources) for a in topic_analyses.values()) / len(topic_analyses)
        source_factor = min(1.0, avg_sources / 3)  # Normalize to 0-1
        
        # Check criteria coverage
        criteria_names = [c.name.lower() for c in criteria]
        covered_criteria = 0
        
        for analysis in topic_analyses.values():
            attrs = {k.lower(): v for k, v in analysis.key_attributes.items()}
            for criterion in criteria_names:
                if any(criterion in key for key in attrs.keys()):
                    covered_criteria += 1
        
        criteria_factor = covered_criteria / (len(criteria) * len(topic_analyses))
        
        # Weighted confidence calculation
        confidence = (
            0.4 * quote_factor +
            0.3 * source_factor +
            0.3 * criteria_factor
        )
        
        return min(0.95, max(0.3, confidence))  # Bound between 0.3 and 0.95
    
    def _create_fallback_topic_analysis(
        self,
        topic: str,
        documents: List[Document]
    ) -> TopicAnalysis:
        """Create a basic topic analysis when LLM analysis fails"""
        # Extract basic data from documents
        doc_text = " ".join([doc.page_content for doc in documents])
        
        # Simple word-based extraction for strengths/weaknesses
        strengths = []
        weaknesses = []
        
        # Look for positive/negative indicators
        positive_indicators = ["advantage", "benefit", "better", "good", "strong", "recommended"]
        negative_indicators = ["disadvantage", "drawback", "worse", "poor", "weak", "caution"]
        
        # Extract simple sentences containing indicators
        sentences = re.split(r'[.!?]', doc_text)
        
        for sentence in sentences:
            s = sentence.lower().strip()
            if any(indicator in s for indicator in positive_indicators):
                if len(s) > 10 and len(strengths) < 3:
                    strengths.append(sentence.strip())
            if any(indicator in s for indicator in negative_indicators):
                if len(s) > 10 and len(weaknesses) < 3:
                    weaknesses.append(sentence.strip())
        
        # Default strengths/weaknesses if none found
        if not strengths:
            strengths = [f"{topic} may offer suitable performance for certain applications"]
        if not weaknesses:
            weaknesses = [f"Limited information available about potential drawbacks of {topic}"]
        
        # Collect document sources
        doc_sources = []
        for doc in documents:
            if "document_id" in doc.metadata:
                doc_sources.append(doc.metadata["document_id"])
            elif "source_id" in doc.metadata:
                doc_sources.append(doc.metadata["source_id"])
        
        return TopicAnalysis(
            topic=topic,
            strengths=strengths,
            weaknesses=weaknesses,
            key_attributes={},
            suitability_score=0.5,  # Neutral score
            supporting_quotes=sentences[:min(3, len(sentences))],  # Take first few sentences
            document_sources=list(set(doc_sources))  # Remove duplicates
        )
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text"""
        # Try to find a JSON object
        json_pattern = r'({[\s\S]*})'
        array_pattern = r'(\[[\s\S]*\])'
        
        # Try to find a JSON object
        json_match = re.search(json_pattern, text)
        if json_match:
            try:
                # Validate it's valid JSON
                json_str = json_match.group(1)
                json.loads(json_str)  # This will raise an exception if invalid
                return json_str
            except:
                pass
        
        # Try to find a JSON array
        array_match = re.search(array_pattern, text)
        if array_match:
            try:
                json_str = array_match.group(1)
                json.loads(json_str)  # This will raise an exception if invalid
                return json_str
            except:
                pass
        
        return None
