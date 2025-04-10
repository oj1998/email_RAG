from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import logging
import json

logger = logging.getLogger(__name__)

# These would normally be imported from document_variance_analyzer.py
# Including simplified versions here for completeness
class DocumentVariance(BaseModel):
    """Represents a variance between documents on the same topic"""
    topic: str
    aspect: str  # The specific aspect where variance was found (e.g., "installation method")
    variance_description: str  # Description of how sources differ
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence in the identified variance
    source_positions: Dict[str, List[str]]  # Document IDs mapped to their positions
    source_excerpts: Dict[str, str]  # Relevant excerpts from each source
    
class TopicVarianceAnalysis(BaseModel):
    """Complete analysis of variances for a topic"""
    topic: str
    key_variances: List[DocumentVariance]
    agreement_points: List[str]  # Points where sources agree
    general_assessment: str  # Overall assessment of source agreement/disagreement
    source_count: int
    reliability_ranking: Dict[str, float]  # Source ID to reliability score

class VarianceFormatStyle(str, Enum):
    SIDE_BY_SIDE = "side_by_side"
    TABULAR = "tabular"
    NARRATIVE = "narrative"
    VISUAL = "visual"
    JSON = "json"

class VarianceFormatter:
    """Formats variance analysis results for presentation"""
    
    def format_variance_analysis(
        self,
        analysis: TopicVarianceAnalysis,
        format_style: VarianceFormatStyle = VarianceFormatStyle.TABULAR
    ) -> str:
        """Format the variance analysis results in the specified style"""
        if format_style == VarianceFormatStyle.SIDE_BY_SIDE:
            return self._format_side_by_side(analysis)
        elif format_style == VarianceFormatStyle.TABULAR:
            return self._format_tabular(analysis)
        elif format_style == VarianceFormatStyle.VISUAL:
            return self._format_visual(analysis)
        else:  # NARRATIVE is the default fallback
            return self._format_narrative(analysis)
            
    def _format_side_by_side(self, analysis: TopicVarianceAnalysis) -> str:
        """Format analysis with sources side by side for each variance"""
        formatted_content = f"# Source Variance Analysis: {analysis.topic}\n\n"
        formatted_content += f"{analysis.general_assessment}\n\n"
        
        # Add section for agreement points if they exist
        if analysis.agreement_points:
            formatted_content += "## Points of Agreement\n\n"
            for point in analysis.agreement_points:
                formatted_content += f"- {point}\n"
            formatted_content += "\n"
        
        # Format each variance
        if analysis.key_variances:
            formatted_content += "## Key Differences Between Sources\n\n"
            
            for i, variance in enumerate(analysis.key_variances, 1):
                formatted_content += f"### Variance {i}: {variance.aspect}\n\n"
                formatted_content += f"{variance.variance_description}\n\n"
                
                # Side-by-side comparison for each source
                formatted_content += "| Source | Position | Key Excerpt |\n"
                formatted_content += "|--------|----------|-------------|\n"
                
                for doc_id, positions in variance.source_positions.items():
                    position_text = ", ".join(positions)
                    excerpt = variance.source_excerpts.get(doc_id, "No excerpt available")
                    # Truncate excerpt if too long
                    if len(excerpt) > 100:
                        excerpt = excerpt[:100] + "..."
                    
                    formatted_content += f"| {doc_id} | {position_text} | {excerpt} |\n"
                
                formatted_content += "\n"
        else:
            formatted_content += "## No Significant Variances Detected\n\n"
            formatted_content += "The analyzed sources show consistent information about this topic.\n\n"
        
        # Source reliability section
        formatted_content += "## Source Reliability Assessment\n\n"
        # Sort sources by reliability score in descending order
        sorted_sources = sorted(
            analysis.reliability_ranking.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        formatted_content += "| Source | Reliability Score | Notes |\n"
        formatted_content += "|--------|-------------------|-------|\n"
        
        for doc_id, score in sorted_sources:
            # Generate note based on score
            if score >= 0.8:
                note = "High reliability"
            elif score >= 0.6:
                note = "Moderate reliability"
            else:
                note = "Lower reliability"
                
            formatted_content += f"| {doc_id} | {score:.2f} | {note} |\n"
            
        return formatted_content
    
    def _format_tabular(self, analysis: TopicVarianceAnalysis) -> str:
        """Format analysis in a tabular format focusing on the differences"""
        formatted_content = f"# Variance Analysis: {analysis.topic}\n\n"
        formatted_content += f"{analysis.general_assessment}\n\n"
        
        # Create variance table
        if analysis.key_variances:
            formatted_content += "## Comparison of Source Positions\n\n"
            
            # Create table header with aspect and source IDs
            source_ids = set()
            for variance in analysis.key_variances:
                source_ids.update(variance.source_positions.keys())
            
            source_ids = sorted(source_ids)
            
            # Build the table header
            table_header = "| Aspect | " + " | ".join(source_ids) + " |\n"
            table_header += "|--------|" + "|".join(["-" * len(sid) for sid in source_ids]) + "|\n"
            
            formatted_content += table_header
            
            # Add each variance as a row
            for variance in analysis.key_variances:
                row = f"| {variance.aspect} | "
                
                for doc_id in source_ids:
                    if doc_id in variance.source_positions:
                        position = ", ".join(variance.source_positions[doc_id])
                        # Truncate if too long
                        if len(position) > 30:
                            position = position[:27] + "..."
                        row += f"{position} | "
                    else:
                        row += "No position | "
                
                formatted_content += row + "\n"
            
            formatted_content += "\n"
            
            # Add detailed descriptions
            formatted_content += "## Detailed Variance Descriptions\n\n"
            
            for i, variance in enumerate(analysis.key_variances, 1):
                formatted_content += f"### {variance.aspect}\n\n"
                formatted_content += f"{variance.variance_description}\n\n"
                
                # Show confidence
                confidence_indicator = "🟢 High" if variance.confidence > 0.8 else "🟡 Medium" if variance.confidence > 0.6 else "🔴 Low"
                formatted_content += f"**Confidence in this variance**: {confidence_indicator}\n\n"
        
        # Add section for agreement points
        if analysis.agreement_points:
            formatted_content += "## Points of Agreement\n\n"
            for point in analysis.agreement_points:
                formatted_content += f"- {point}\n"
                
        return formatted_content
    
    def _format_narrative(self, analysis: TopicVarianceAnalysis) -> str:
        """Format analysis as a narrative text"""
        formatted_content = f"# {analysis.topic}: Source Variance Analysis\n\n"
        
        # Add general assessment
        formatted_content += f"{analysis.general_assessment}\n\n"
        
        # Add section for agreement points
        if analysis.agreement_points:
            formatted_content += "## Areas of Consensus\n\n"
            formatted_content += "All sources examined agree on the following points:\n\n"
            for point in analysis.agreement_points:
                formatted_content += f"- {point}\n"
            formatted_content += "\n"
        
        # Add section for variances
        if analysis.key_variances:
            formatted_content += "## Areas of Divergence\n\n"
            
            for variance in analysis.key_variances:
                formatted_content += f"### {variance.aspect}\n\n"
                formatted_content += f"{variance.variance_description}\n\n"
                
                # Add source-specific positions
                formatted_content += "#### Source Positions:\n\n"
                for doc_id, positions in variance.source_positions.items():
                    formatted_content += f"**{doc_id}**: {', '.join(positions)}\n\n"
                    # Add excerpt
                    if doc_id in variance.source_excerpts:
                        excerpt = variance.source_excerpts[doc_id]
                        formatted_content += f"> {excerpt}\n\n"
        
        # Add reliability assessment
        formatted_content += "## Source Reliability Assessment\n\n"
        
        # Calculate overall variance level
        variance_count = len(analysis.key_variances)
        if variance_count == 0:
            variance_level = "no significant variances"
        elif variance_count <= 2:
            variance_level = "minor variances"
        elif variance_count <= 5:
            variance_level = "moderate variances"
        else:
            variance_level = "significant variances"
        
        formatted_content += f"The analysis found {variance_level} between the {analysis.source_count} sources examined. "
        
        # Add reliability notes
        sorted_sources = sorted(
            analysis.reliability_ranking.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_sources:
            most_reliable = sorted_sources[0][0]
            formatted_content += f"Based on metadata and content analysis, source **{most_reliable}** appears to be the most reliable, "
            
            if len(sorted_sources) > 1:
                least_reliable = sorted_sources[-1][0]
                formatted_content += f"while **{least_reliable}** shows lower reliability indicators.\n\n"
            else:
                formatted_content += ".\n\n"
                
            formatted_content += "Note that reliability assessment is based on factors such as recency, specificity, and alignment with other sources, and should not be taken as definitive.\n\n"
        
        return formatted_content
    
    def _format_visual(self, analysis: TopicVarianceAnalysis) -> str:
        """Format with visual indicators (emojis, formatting) to highlight differences"""
        formatted_content = f"# 📊 Variance Analysis: {analysis.topic}\n\n"
        
        # Add color-coded confidence bar
        confidence_levels = []
        if analysis.key_variances:
            avg_confidence = sum(v.confidence for v in analysis.key_variances) / len(analysis.key_variances)
            confidence_bar = "**Source Agreement Level**: "
            
            if avg_confidence > 0.8:
                confidence_bar += "🔴 Low Agreement (High Variance)"
            elif avg_confidence > 0.6:
                confidence_bar += "🟠 Moderate Agreement"
            elif avg_confidence > 0.4:
                confidence_bar += "🟡 Good Agreement"
            else:
                confidence_bar += "🟢 Strong Agreement (Low Variance)"
                
            formatted_content += f"{confidence_bar}\n\n"
        else:
            formatted_content += "**Source Agreement Level**: 🟢 Complete Agreement (No Variances)\n\n"
        
        # Add general assessment with visual separation
        formatted_content += "## 📝 Executive Summary\n\n"
        formatted_content += f"{analysis.general_assessment}\n\n"
        formatted_content += "---\n\n"
        
        # Agreement points with checkmarks
        if analysis.agreement_points:
            formatted_content += "## ✅ Points of Agreement\n\n"
            for point in analysis.agreement_points:
                formatted_content += f"✓ {point}\n"
            formatted_content += "\n---\n\n"
        
        # Variances with visual indicators
        if analysis.key_variances:
            formatted_content += "## ⚠️ Key Variances Detected\n\n"
            
            for i, variance in enumerate(analysis.key_variances, 1):
                # Add emoji based on confidence
                emoji = "🔴" if variance.confidence > 0.8 else "🟠" if variance.confidence > 0.6 else "🟡"
                
                formatted_content += f"### {emoji} Variance {i}: {variance.aspect}\n\n"
                formatted_content += f"{variance.variance_description}\n\n"
                
                # Add source comparison with emojis
                formatted_content += "#### Source Positions:\n\n"
                
                # Identify most common position (to mark outliers)
                position_counts = {}
                for positions in variance.source_positions.values():
                    for pos in positions:
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                
                most_common_position = None
                if position_counts:
                    most_common_position = max(position_counts.items(), key=lambda x: x[1])[0]
                
                # Show sources with position indicators
                for doc_id, positions in variance.source_positions.items():
                    position_text = ", ".join(positions)
                    
                    # Add outlier indicator
                    if most_common_position and not any(most_common_position in pos for pos in positions):
                        source_prefix = "⚠️ "  # Mark as outlier
                    else:
                        source_prefix = "📄 "
                        
                    formatted_content += f"{source_prefix}**{doc_id}**: {position_text}\n\n"
                    
                    # Add excerpt in blockquote
                    if doc_id in variance.source_excerpts:
                        excerpt = variance.source_excerpts[doc_id]
                        formatted_content += f"> {excerpt}\n\n"
                
                formatted_content += "---\n\n"
        else:
            formatted_content += "## ✅ No Significant Variances\n\n"
            formatted_content += "The analyzed sources show consistent information on this topic.\n\n"
            formatted_content += "---\n\n"
        
        # Source reliability section with visual indicators
        formatted_content += "## 🔍 Source Reliability Assessment\n\n"
        
        # Sort sources by reliability
        sorted_sources = sorted(
            analysis.reliability_ranking.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for doc_id, score in sorted_sources:
            # Generate reliability indicator
            if score >= 0.8:
                indicator = "🟢 High reliability"
            elif score >= 0.6:
                indicator = "🟡 Moderate reliability"
            else:
                indicator = "🔴 Lower reliability"
                
            formatted_content += f"**{doc_id}**: {indicator} ({score:.2f})\n\n"
