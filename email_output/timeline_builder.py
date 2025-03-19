from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import re
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class TimelineBuilder:
    """Builds email timelines around anchor emails"""
    
    def __init__(self, llm=None):
        """Initialize with optional language model"""
        self.llm = llm or ChatOpenAI(temperature=0.2)
        logger.info("TimelineBuilder initialized")
    
    async def build_timeline(
        self,
        anchor_email: Dict[str, Any],
        related_emails: List[Dict[str, Any]],
        query: str,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a timeline around an anchor email
        
        Args:
            anchor_email: The most relevant email to use as an anchor
            related_emails: Other related emails for context
            query: The original user query
            timeframe: Optional timeframe context
            
        Returns:
            Timeline data structure with events and metadata
        """
        start_time = datetime.now()
        logger.info(f"Building timeline for query: '{query}'")
        logger.info(f"Using anchor email: '{anchor_email.get('subject', 'No subject')}' from {anchor_email.get('sender', 'Unknown')}")
        logger.info(f"With {len(related_emails)} related emails")
        
        # Combine all emails
        all_emails = [anchor_email] + related_emails
        
        # Sort by date
        logger.info("Sorting emails by date")
        sorted_emails = sorted(
            all_emails,
            key=lambda x: self._parse_date(x.get('date', '')),
            reverse=False  # Ascending order (oldest first)
        )
        logger.info(f"Sorted {len(sorted_emails)} emails chronologically")
        
        # Extract timeline events with context
        logger.info("Extracting timeline events with context")
        timeline_events = await self._extract_timeline_events(sorted_emails, query)
        
        # Identify key contributors
        logger.info("Identifying key contributors")
        contributors = self._identify_contributors(sorted_emails)
        logger.info(f"Identified {len(contributors)} contributors")
        
        # Find turning points in the discussion
        logger.info("Finding turning points in the discussion")
        turning_points = await self._identify_turning_points(timeline_events, query)
        
        # Build timeline data structure
        date_range = {
            "start": sorted_emails[0].get('date') if sorted_emails else None,
            "end": sorted_emails[-1].get('date') if sorted_emails else None
        }
        
        timeline_data = {
            "events": timeline_events,
            "contributors": contributors,
            "turning_points": turning_points,
            "query": query,
            "timeframe": timeframe,
            "email_count": len(sorted_emails),
            "date_range": date_range
        }
        
        # Log detailed timeline data for debugging
        logger.info("========== DETAILED TIMELINE DATA ==========")
        logger.info(f"Query: '{query}'")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Date range: {date_range['start']} to {date_range['end']}")
        
        # Log events details
        logger.info(f"Timeline events ({len(timeline_events)}):")
        for i, event in enumerate(timeline_events):
            logger.info(f"  Event {i+1}: {event['date']} - {event['subject']} from {event['sender']}")
            contribution = event['contribution']
            logger.info(f"    Contribution: {contribution[:100]}..." if len(contribution) > 100 else f"    Contribution: {contribution}")
            logger.info(f"    Key points: {event['key_points']}")
        
        # Log turning points
        logger.info(f"Turning points ({len(turning_points)}):")
        for i, point in enumerate(turning_points):
            event_idx = point.get('event_index')
            event_info = f"{timeline_events[event_idx]['date']} - {timeline_events[event_idx]['subject']}" if 0 <= event_idx < len(timeline_events) else "Unknown"
            logger.info(f"  Point {i+1}: {event_info}")
            logger.info(f"    Description: {point.get('description', 'No description')}")
        
        # Log contributors
        logger.info(f"Contributors ({len(contributors)}):")
        for contributor in contributors:
            logger.info(f"  {contributor['name']}: {contributor['email_count']} emails ({contributor['participation_percentage']:.1f}%)")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Timeline building completed in {total_time:.2f} seconds")
        logger.info(f"Timeline contains {len(timeline_events)} events, {len(turning_points)} turning points")
        logger.info(f"Timeline date range: {date_range['start']} to {date_range['end']}")
        
        return timeline_data
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object with fallback"""
        try:
            # Try common formats
            for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%b %d, %Y'):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
            # Last resort, try to parse with dateutil
            from dateutil import parser
            return parser.parse(date_str)
        except Exception as e:
            logger.warning(f"Date parsing failed for '{date_str}': {e}")
            # Return epoch start as fallback
            return datetime(1970, 1, 1)
    
    async def _extract_timeline_events(
        self,
        emails: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Extract meaningful timeline events from emails"""
        events = []
        
        # Add start time tracking
        start_time = datetime.now()
        logger.info(f"Beginning timeline extraction for {len(emails)} emails related to: '{query}'")
        
        for i, email in enumerate(emails):
            # Log each email being processed
            logger.info(f"Processing email {i+1}/{len(emails)}: '{email.get('subject', 'No subject')}' " 
                       f"from {email.get('sender', 'Unknown')} dated {email.get('date', 'Unknown date')}")
            
            # Basic event data
            event = {
                "date": email.get('date', 'Unknown date'),
                "sender": email.get('sender', 'Unknown sender'),
                "subject": email.get('subject', 'No subject'),
                "contribution": "",  # Will be filled by LLM
                "key_points": [],    # Will be filled by LLM
                "references": [],    # Will be filled by LLM
                "email_id": email.get('id', 'unknown')
            }
            
            # Use LLM to extract contribution and key points
            content = email.get('content', email.get('excerpt', ''))
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"Email content preview: {content_preview}")
            
            if content:
                logger.info(f"Extracting contribution and key points using LLM")
                extraction_start = datetime.now()
                extraction = await self._analyze_email_contribution(content, query, event["sender"])
                extraction_time = (datetime.now() - extraction_start).total_seconds()
                
                event.update(extraction)
                
                # Log the extraction results
                contribution_preview = event['contribution'][:100] + "..." if len(event['contribution']) > 100 else event['contribution']
                logger.info(f"Extraction complete in {extraction_time:.2f}s:")
                logger.info(f"  Contribution: {contribution_preview}")
                logger.info(f"  Key points: {event['key_points']}")
                if event['references']:
                    logger.info(f"  References: {event['references']}")
            else:
                logger.warning(f"Email {i+1} has no content to analyze")
                
            events.append(event)
        
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Timeline event extraction completed in {processing_time:.2f} seconds")
        logger.info(f"Extracted {len(events)} events total")
        
        return events
    
    async def _analyze_email_contribution(
        self,
        content: str,
        query: str,
        sender: str
    ) -> Dict[str, Any]:
        """Use LLM to analyze the email's contribution to the timeline"""
        logger.info(f"Analyzing contribution for email from {sender}")
        
        try:
            # Limit content for logging
            log_content = content[:150] + "..." if len(content) > 150 else content
            logger.info(f"Content to analyze: {log_content}")
            
            prompt = PromptTemplate.from_template("""
            Analyze this email content in the context of the query: "{query}"
            
            Email content:
            {content}
            
            Extract:
            1. The main contribution this email makes to the topic
            2. Key points related to the topic (up to 3)
            3. Any references to documents, meetings, or other communications
            
            Format your response as JSON with these keys: contribution, key_points, references
            """)
            
            logger.info("Sending to LLM for analysis")
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "query": query,
                "content": content[:1500]  # Limit content length
            })
            
            # Parse the response
            import json
            import re
            
            # Extract JSON from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.info(f"Received LLM response of length {len(response_text)}")
            logger.info(f"Response preview: {response_text[:150]}...")
            
            # Try to extract JSON object
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                try:
                    matched_json = json_match.group(1)
                    logger.info(f"Found JSON match: {matched_json[:150]}...")
                    result = json.loads(matched_json)
                    
                    # Ensure expected keys exist
                    extraction_result = {
                        "contribution": result.get("contribution", ""),
                        "key_points": result.get("key_points", []),
                        "references": result.get("references", [])
                    }
                    
                    logger.info(f"Successfully extracted structured data from LLM response")
                    return extraction_result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Failed JSON string: {matched_json[:150]}...")
                    pass
            else:
                logger.warning("No JSON object found in LLM response")
            
            # Fallback: extract structured data without JSON parsing
            logger.info("Using fallback extraction method")
            return {
                "contribution": f"Content from {sender}",
                "key_points": [],
                "references": []
            }
                
        except Exception as e:
            logger.error(f"Error analyzing email contribution: {str(e)}", exc_info=True)
            return {
                "contribution": f"Content from {sender}",
                "key_points": [],
                "references": []
            }
    
    def _identify_contributors(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key contributors in the email thread"""
        logger.info(f"Identifying contributors from {len(emails)} emails")
        
        # Count contributions by sender
        contribution_count = {}
        for email in emails:
            sender = email.get('sender', 'Unknown')
            if sender in contribution_count:
                contribution_count[sender] += 1
            else:
                contribution_count[sender] = 1
        
        logger.info(f"Found {len(contribution_count)} unique contributors")
        
        # Convert to list of contributors with metrics
        contributors = [
            {
                "name": sender,
                "email_count": count,
                "participation_percentage": (count / len(emails)) * 100 if emails else 0
            }
            for sender, count in contribution_count.items()
        ]
        
        # Sort by contribution count (descending)
        contributors.sort(key=lambda x: x["email_count"], reverse=True)
        
        # Log contributor details
        for contributor in contributors:
            logger.info(f"Contributor: {contributor['name']} - {contributor['email_count']} emails "
                       f"({contributor['participation_percentage']:.1f}%)")
        
        return contributors
    
    async def _identify_turning_points(
        self,
        events: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Identify turning points or significant changes in the discussion"""
        logger.info(f"Identifying turning points in {len(events)} events")
        
        start_time = datetime.now()
        
        if not events or len(events) < 2:
            logger.info("Not enough events to identify turning points")
            return []
            
        try:
            # Prepare content for LLM analysis
            events_text = ""
            for i, event in enumerate(events):
                events_text += f"Event {i+1} ({event['date']}):\n"
                events_text += f"From: {event['sender']}\n"
                events_text += f"Subject: {event['subject']}\n"
                events_text += f"Contribution: {event['contribution']}\n"
                if event['key_points']:
                    events_text += "Key points: " + ", ".join(event['key_points']) + "\n"
                events_text += "\n"
            
            logger.info(f"Prepared events text of length {len(events_text)} for turning point analysis")
            
            # Use LLM to identify turning points
            prompt = PromptTemplate.from_template("""
            Analyze this sequence of email events related to the query: "{query}"
            
            {events_text}
            
            Identify up to 3 key turning points in this discussion - moments where the direction
            changed, decisions were made, or significant new information was introduced.
            
            For each turning point, provide:
            1. The event number
            2. A brief description of why it's significant
            
            Format your response as JSON with an array of objects with these keys: event_index, description
            """)
            
            logger.info("Sending to LLM for turning point identification")
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "query": query,
                "events_text": events_text
            })
            
            # Parse the response
            import json
            import re
            
            # Extract JSON from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.info(f"Received turning point analysis from LLM, length: {len(response_text)}")
            logger.info(f"Response preview: {response_text[:150]}...")
            
            # Try to extract JSON object
            json_match = re.search(r'(\[{.*}\])', response_text, re.DOTALL)
            if json_match:
                try:
                    matched_json = json_match.group(1)
                    logger.info(f"Found JSON match: {matched_json[:150]}...")
                    turning_points_raw = json.loads(matched_json)
                    
                    # Process turning points - convert event numbers to indices
                    turning_points = []
                    for point in turning_points_raw:
                        if "event_index" in point:
                            event_index = int(point["event_index"]) - 1
                            if 0 <= event_index < len(events):
                                turning_point = {
                                    "event_index": event_index,
                                    "date": events[event_index]["date"],
                                    "sender": events[event_index]["sender"],
                                    "description": point.get("description", "Significant change in discussion")
                                }
                                turning_points.append(turning_point)
                                logger.info(f"Identified turning point at event {event_index+1}: {turning_point['description']}")
                    
                    logger.info(f"Successfully identified {len(turning_points)} turning points")
                    
                    # Log processing time
                    processing_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Turning point identification completed in {processing_time:.2f} seconds")
                    
                    return turning_points
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Failed JSON string: {matched_json[:150]}...")
            else:
                logger.warning("No JSON array found in LLM turning point response")
            
            # Fallback - return empty list
            logger.info("No turning points identified")
            return []
                
        except Exception as e:
            logger.error(f"Error identifying turning points: {str(e)}", exc_info=True)
            return []
