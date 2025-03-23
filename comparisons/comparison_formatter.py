from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel
from comparisons.comparison_analyzer import ComparisonAnalysis, TopicAnalysis, ComparisonCriterion
import logging

logger = logging.getLogger(__name__)

class ComparisonFormatStyle(Enum):
    """Different formatting styles for comparison responses"""
    TABULAR = "tabular"           # Side-by-side table comparison
    NARRATIVE = "narrative"       # Prose-style comparison
    BULLET = "bullet"             # Bullet point lists
    STRUCTURED = "structured"     # Structured sections with headers
    MARKDOWN_TABLE = "markdown_table"  # Markdown table format

class ComparisonFormatter:
    """Format comparison analyses into readable text formats"""
    
    def format_comparison(
        self,
        analysis: ComparisonAnalysis,
        format_style: ComparisonFormatStyle = ComparisonFormatStyle.STRUCTURED,
        max_length: Optional[int] = None
    ) -> str:
        """Format a comparison analysis using the specified style"""
        if format_style == ComparisonFormatStyle.TABULAR:
            return self._format_tabular(analysis)
        elif format_style == ComparisonFormatStyle.NARRATIVE:
            return self._format_narrative(analysis)
        elif format_style == ComparisonFormatStyle.BULLET:
            return self._format_bullet(analysis)
        elif format_style == ComparisonFormatStyle.MARKDOWN_TABLE:
            return self._format_markdown_table(analysis)
        else:
            # Default to structured format
            return self._format_structured(analysis)
    
    def _format_structured(self, analysis: ComparisonAnalysis) -> str:
        """Format comparison with clear sections and headers"""
        formatted = f"# Comparison Analysis: {' vs '.join(analysis.topics)}\n\n"
        
        # Add the overall recommendation first
        formatted += f"## Recommendation\n\n{analysis.overall_recommendation}\n\n"
        
        # Add comparison criteria section
        formatted += "## Comparison Criteria\n\n"
        for criterion in analysis.criteria:
            formatted += f"- **{criterion.name}**: {criterion.description}\n"
        formatted += "\n"
        
        # Add detailed analysis for each topic
        for topic, topic_analysis in analysis.topic_analyses.items():
            formatted += f"## {topic}\n\n"
            
            # Strengths section
            formatted += "### Strengths\n\n"
            for strength in topic_analysis.strengths:
                formatted += f"- {strength}\n"
            formatted += "\n"
            
            # Weaknesses section
            formatted += "### Weaknesses\n\n"
            for weakness in topic_analysis.weaknesses:
                formatted += f"- {weakness}\n"
            formatted += "\n"
            
            # Key attributes section if available
            if topic_analysis.key_attributes:
                formatted += "### Key Attributes\n\n"
                for attr, value in topic_analysis.key_attributes.items():
                    formatted += f"- **{attr}**: {value}\n"
                formatted += "\n"
            
            # Supporting quotes if available
            if topic_analysis.supporting_quotes:
                formatted += "### Supporting Evidence\n\n"
                for quote in topic_analysis.supporting_quotes:
                    formatted += f"> {quote}\n\n"
        
        # Add source information
        formatted += "## Sources\n\n"
        sources = set()
        for topic_analysis in analysis.topic_analyses.values():
            sources.update(topic_analysis.document_sources)
        
        for source in sources:
            formatted += f"- Document ID: {source}\n"
        
        return formatted
    
    def _format_tabular(self, analysis: ComparisonAnalysis) -> str:
        """Format comparison as a side-by-side table"""
        formatted = f"# {' vs '.join(analysis.topics)} Comparison\n\n"
        
        # Include recommendation
        formatted += f"**Recommendation**: {analysis.overall_recommendation}\n\n"
        
        # Create the comparison table header
        table_header = "| Aspect | " + " | ".join(analysis.topics) + " |\n"
        table_divider = "|--------|" + "|".join(["-" * (len(topic) + 2) for topic in analysis.topics]) + "|\n"
        
        table_content = table_header + table_divider
        
        # Add criteria rows
        for criterion in analysis.criteria:
            row = f"| **{criterion.name}** |"
            
            for topic in analysis.topics:
                topic_analysis = analysis.topic_analyses.get(topic)
                if topic_analysis:
                    # Try to find the criterion in key_attributes
                    value = "N/A"
                    for attr_name, attr_value in topic_analysis.key_attributes.items():
                        if criterion.name.lower() in attr_name.lower():
                            value = str(attr_value)
                            break
                    row += f" {value} |"
                else:
                    row += " N/A |"
            
            table_content += row + "\n"
        
        # Add strengths row
        table_content += "| **Strengths** |"
        for topic in analysis.topics:
            topic_analysis = analysis.topic_analyses.get(topic)
            if topic_analysis and topic_analysis.strengths:
                strengths = "<br>".join([f"• {s}" for s in topic_analysis.strengths])
                table_content += f" {strengths} |"
            else:
                table_content += " N/A |"
        table_content += "\n"
        
        # Add weaknesses row
        table_content += "| **Weaknesses** |"
        for topic in analysis.topics:
            topic_analysis = analysis.topic_analyses.get(topic)
            if topic_analysis and topic_analysis.weaknesses:
                weaknesses = "<br>".join([f"• {s}" for s in topic_analysis.weaknesses])
                table_content += f" {weaknesses} |"
            else:
                table_content += " N/A |"
        table_content += "\n"
        
        # Add suitability score row
        table_content += "| **Overall Score** |"
        for topic in analysis.topics:
            topic_analysis = analysis.topic_analyses.get(topic)
            if topic_analysis:
                score = topic_analysis.suitability_score
                table_content += f" {score:.1f}/1.0 |"
            else:
                table_content += " N/A |"
        table_content += "\n"
        
        formatted += table_content + "\n"
        
        # Add sources section
        formatted += "**Sources**: "
        sources = set()
        for topic_analysis in analysis.topic_analyses.values():
            sources.update(topic_analysis.document_sources)
        formatted += ", ".join(sources)
        
        return formatted
    
    def _format_markdown_table(self, analysis: ComparisonAnalysis) -> str:
        """Format comparison as a clean markdown table"""
        # This uses the tabular format but with cleaner formatting for markdown
        return self._format_tabular(analysis)
    
    def _format_narrative(self, analysis: ComparisonAnalysis) -> str:
        """Format comparison as prose narrative"""
        formatted = f"# {' vs '.join(analysis.topics)} Comparison\n\n"
        
        # Include introduction and context
        formatted += f"This comparison analyzes {' and '.join(analysis.topics)} for {analysis.query_context}. "
        formatted += "The analysis is based on document sources from the construction industry.\n\n"
        
        # Add overall recommendation
        formatted += f"**Recommendation**: {analysis.overall_recommendation}\n\n"
        
        # Main comparison narrative
        formatted += "## Comparison Summary\n\n"
        
        # Generate comparison narrative
        topic_narratives = []
        for topic in analysis.topics:
            topic_analysis = analysis.topic_analyses.get(topic)
            if not topic_analysis:
                continue
                
            narrative = f"**{topic}** "
            
            # Strengths
            if topic_analysis.strengths:
                narrative += f"offers advantages in {', '.join(topic_analysis.strengths[:2])}. "
            
            # Weaknesses
            if topic_analysis.weaknesses:
                narrative += f"However, it has limitations regarding {', '.join(topic_analysis.weaknesses[:2])}. "
            
            # Suitability score
            narrative += f"Overall suitability score: {topic_analysis.suitability_score:.1f}/1.0."
            
            topic_narratives.append(narrative)
        
        formatted += "\n\n".join(topic_narratives) + "\n\n"
        
        # Add criteria-based comparison
        formatted += "## Criteria Comparison\n\n"
        for criterion in analysis.criteria:
            formatted += f"### {criterion.name}\n\n"
            
            criterion_comparison = []
            for topic in analysis.topics:
                topic_analysis = analysis.topic_analyses.get(topic)
                if not topic_analysis:
                    continue
                    
                # Find criterion in key_attributes
                value = "N/A"
                for attr_name, attr_value in topic_analysis.key_attributes.items():
                    if criterion.name.lower() in attr_name.lower():
                        value = str(attr_value)
                        break
                
                criterion_comparison.append(f"**{topic}**: {value}")
            
            formatted += ". ".join(criterion_comparison) + "\n\n"
        
        # Add source references
        formatted += "## Sources\n\n"
        sources = set()
        for topic_analysis in analysis.topic_analyses.values():
            sources.update(topic_analysis.document_sources)
        
        formatted += f"This analysis is based on {len(sources)} document sources: {', '.join(sources)}."
        
        return formatted
    
    def _format_bullet(self, analysis: ComparisonAnalysis) -> str:
        """Format comparison as bullet points"""
        formatted = f"# {' vs '.join(analysis.topics)} Comparison\n\n"
        
        # Add overall recommendation
        formatted += f"## Recommendation\n\n{analysis.overall_recommendation}\n\n"
        
        # Format comparison criteria
        formatted += "## Comparison Criteria\n\n"
        for criterion in analysis.criteria:
            formatted += f"• **{criterion.name}**: {criterion.description}\n"
        formatted += "\n"
        
        # Topic summaries
        for topic in analysis.topics:
            topic_analysis = analysis.topic_analyses.get(topic)
            if not topic_analysis:
                continue
                
            formatted += f"## {topic} (Score: {topic_analysis.suitability_score:.1f}/1.0)\n\n"
            
            # Strengths
            formatted += "• **Strengths**:\n"
            for strength in topic_analysis.strengths:
                formatted += f"  - {strength}\n"
            formatted += "\n"
            
            # Weaknesses
            formatted += "• **Weaknesses**:\n"
            for weakness in topic_analysis.weaknesses:
                formatted += f"  - {weakness}\n"
            formatted += "\n"
            
            # Key attributes
            if topic_analysis.key_attributes:
                formatted += "• **Key Attributes**:\n"
                for attr, value in topic_analysis.key_attributes.items():
                    formatted += f"  - {attr}: {value}\n"
                formatted += "\n"
        
        # Direct comparison by criteria
        formatted += "## Direct Comparison\n\n"
        for criterion in analysis.criteria:
            formatted += f"• **{criterion.name}**:\n"
            for topic in analysis.topics:
                topic_analysis = analysis.topic_analyses.get(topic)
                if not topic_analysis:
                    continue
                    
                # Find criterion in key_attributes
                value = "N/A"
                for attr_name, attr_value in topic_analysis.key_attributes.items():
                    if criterion.name.lower() in attr_name.lower():
                        value = str(attr_value)
                        break
                
                formatted += f"  - {topic}: {value}\n"
            formatted += "\n"
        
        # Sources
        formatted += "## Sources\n\n"
        sources = set()
        for topic_analysis in analysis.topic_analyses.values():
            sources.update(topic_analysis.document_sources)
        
        for source in sources:
            formatted += f"• Document ID: {source}\n"
        
        return formatted
