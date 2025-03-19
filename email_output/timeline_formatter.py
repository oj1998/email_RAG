from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TimelineFormat(Enum):
    """Different formats for displaying timelines"""
    CHRONOLOGICAL = "chronological"  # Simple chronological list
    NARRATIVE = "narrative"         # Narrative with key turning points highlighted
    CONTRIBUTOR = "contributor"     # Grouped by contributor
    COMPACT = "compact"             # Compact summary of key events

class TimelineFormatter:
    """Format timeline data for different display needs"""
    
    def format_timeline(
        self,
        timeline_data: Dict[str, Any],
        format_style: TimelineFormat = TimelineFormat.CHRONOLOGICAL,
        include_contributors: bool = True,
        highlight_turning_points: bool = True
    ) -> str:
        """
        Format timeline data into readable text
        
        Args:
            timeline_data: The timeline data structure
            format_style: How to format the timeline
            include_contributors: Whether to include contributor summary
            highlight_turning_points: Whether to highlight turning points
            
        Returns:
            Formatted timeline text
        """
        # Extract components from timeline data
        events = timeline_data.get("events", [])
        contributors = timeline_data.get("contributors", [])
        turning_points = timeline_data.get("turning_points", [])
        query = timeline_data.get("query", "")
        date_range = timeline_data.get("date_range", {})
        
        if not events:
            return "No timeline events found."
        
        # Format header with date range
        header = f"# TIMELINE: {query}\n\n"
        if date_range and date_range.get("start") and date_range.get("end"):
            header += f"**Time period:** {date_range['start']} to {date_range['end']}\n\n"
        
        # Format based on style
        if format_style == TimelineFormat.CHRONOLOGICAL:
            content = self._format_chronological(events, turning_points, highlight_turning_points)
        elif format_style == TimelineFormat.NARRATIVE:
            content = self._format_narrative(events, turning_points)
        elif format_style == TimelineFormat.CONTRIBUTOR:
            content = self._format_by_contributor(events, contributors)
        elif format_style == TimelineFormat.COMPACT:
            content = self._format_compact(events, turning_points)
        else:
            # Default to chronological
            content = self._format_chronological(events, turning_points, highlight_turning_points)
        
        # Add contributor summary if requested
        if include_contributors and contributors:
            contributor_section = self._format_contributors(contributors)
            content += f"\n\n## CONTRIBUTORS\n\n{contributor_section}"
        
        return header + content
    
    def _format_chronological(
        self,
        events: List[Dict[str, Any]],
        turning_points: List[Dict[str, Any]],
        highlight_turning_points: bool
    ) -> str:
        """Format timeline as a chronological list"""
        formatted = "## CHRONOLOGICAL VIEW\n\n"
        
        # Get turning point indices for highlighting
        turning_point_indices = [tp.get("event_index") for tp in turning_points] if highlight_turning_points else []
        
        for i, event in enumerate(events):
            # Determine if this is a turning point
            is_turning_point = i in turning_point_indices
            turning_point_desc = ""
            
            if is_turning_point:
                for tp in turning_points:
                    if tp.get("event_index") == i:
                        turning_point_desc = f" - **TURNING POINT:** {tp.get('description', '')}"
            
            # Format the event
            formatted += f"### {event['date']}{turning_point_desc}\n\n"
            formatted += f"**From:** {event['sender']}\n"
            formatted += f"**Subject:** {event['subject']}\n\n"
            
            if event['contribution']:
                formatted += f"{event['contribution']}\n\n"
            
            if event['key_points']:
                formatted += "**Key points:**\n"
                for point in event['key_points']:
                    formatted += f"- {point}\n"
                formatted += "\n"
            
            if event['references']:
                formatted += "**References:**\n"
                for ref in event['references']:
                    formatted += f"- {ref}\n"
                formatted += "\n"
            
            # Add separator
            formatted += "---\n\n"
        
        return formatted
    
    def _format_narrative(
        self,
        events: List[Dict[str, Any]],
        turning_points: List[Dict[str, Any]]
    ) -> str:
        """Format timeline as a narrative with turning points highlighted"""
        formatted = "## NARRATIVE VIEW\n\n"
        
        # Introduction
        formatted += f"This topic evolved through {len(events)} communications. "
        if turning_points:
            formatted += f"There were {len(turning_points)} key turning points in the discussion.\n\n"
        else:
            formatted += "The discussion progressed linearly without major turning points.\n\n"
        
        # Highlight turning points first
        if turning_points:
            formatted += "### KEY TURNING POINTS\n\n"
            for tp in turning_points:
                event_index = tp.get("event_index")
                if 0 <= event_index < len(events):
                    event = events[event_index]
                    formatted += f"**{event['date']} - {event['subject']}**\n"
                    formatted += f"_From {event['sender']}_\n\n"
                    formatted += f"{tp.get('description', 'Significant change')}\n\n"
            
            formatted += "---\n\n"
        
        # Narrative summary
        formatted += "### FULL NARRATIVE\n\n"
        
        # Group events by week or month for more readable narrative
        current_date = None
        for event in events:
            event_date = event['date']
            
            # Check if this is a new date section
            if event_date != current_date:
                formatted += f"**On {event_date}:**\n\n"
                current_date = event_date
            
            formatted += f"- {event['sender']} sent \"{event['subject']}\" - {event['contribution']}\n"
        
        return formatted
    
    def _format_by_contributor(
        self,
        events: List[Dict[str, Any]],
        contributors: List[Dict[str, Any]]
    ) -> str:
        """Format timeline grouped by contributor"""
        formatted = "## CONTRIBUTOR VIEW\n\n"
        
        # Group events by contributor
        contributor_events = {}
        for event in events:
            sender = event['sender']
            if sender not in contributor_events:
                contributor_events[sender] = []
            contributor_events[sender].append(event)
        
        # Format each contributor's section
        for contributor in contributors:
            name = contributor['name']
            count = contributor['email_count']
            percentage = contributor['participation_percentage']
            
            formatted += f"### {name} ({count} emails, {percentage:.1f}%)\n\n"
            
            if name in contributor_events:
                for event in contributor_events[name]:
                    formatted += f"**{event['date']} - {event['subject']}**\n"
                    formatted += f"{event['contribution']}\n\n"
                    
                    if event['key_points']:
                        key_points = ", ".join(event['key_points'])
                        formatted += f"Key points: {key_points}\n\n"
            
            formatted += "---\n\n"
        
        return formatted
    
    def _format_compact(
        self,
        events: List[Dict[str, Any]],
        turning_points: List[Dict[str, Any]]
    ) -> str:
        """Format timeline as a compact summary"""
        formatted = "## COMPACT TIMELINE\n\n"
        
        for i, event in enumerate(events):
            # Check if this is a turning point
            is_turning_point = False
            turning_point_desc = ""
            
            for tp in turning_points:
                if tp.get("event_index") == i:
                    is_turning_point = True
                    turning_point_desc = tp.get('description', '')
            
            # Format differently based on significance
            if is_turning_point:
                formatted += f"**{event['date']}:** {event['sender']} - {event['subject']}\n"
                formatted += f"➤ **{turning_point_desc}**\n\n"
            else:
                # More compact format for regular events
                formatted += f"**{event['date']}:** {event['sender']} - {event['subject']}\n"
                
                if event['contribution']:
                    # Truncate contribution for compactness
                    contrib = event['contribution']
                    if len(contrib) > 100:
                        contrib = contrib[:97] + "..."
                    formatted += f"- {contrib}\n"
                
                formatted += "\n"
        
        return formatted
    
    def _format_contributors(self, contributors: List[Dict[str, Any]]) -> str:
        """Format contributor summary"""
        if not contributors:
            return "No contributor information available."
        
        formatted = ""
        for contributor in contributors:
            name = contributor['name']
            count = contributor['email_count']
            percentage = contributor['participation_percentage']
            
            formatted += f"- **{name}**: {count} emails ({percentage:.1f}%)\n"
        
        return formatted
