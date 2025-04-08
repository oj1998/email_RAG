from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import re
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

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
        
        # Log relevance scores of all emails
        logger.info("========== EMAIL RELEVANCE SCORES ==========")
        logger.info(f"Anchor email: '{anchor_email.get('subject')}' - Score: {anchor_email.get('confidence', 0):.4f}")
        
        for i, email in enumerate(related_emails):
            logger.info(f"Related email #{i+1}: '{email.get('subject')}' FROM {email.get('sender')} - Score: {email.get('confidence', 0):.4f}")
        logger.info("============================================")
        
        # Combine all emails
        all_emails = [anchor_email] + related_emails
        
        # Log how many emails have zero or near-zero relevance
        low_relevance_count = sum(1 for email in all_emails if email.get('confidence', 0) < 0.1)
        logger.info(f"Number of emails with relevance score < 0.1: {low_relevance_count} out of {len(all_emails)}")
        
        # Filter out emails with low confidence scores
        CONFIDENCE_THRESHOLD = 0.1  # Adjust as needed
        
        # Always keep the anchor email (the most relevant one)
        filtered_emails = [anchor_email]
        
        # Only add related emails if they meet the confidence threshold
        filtered_related = [
            email for email in related_emails 
            if email.get('confidence', 0) >= CONFIDENCE_THRESHOLD
        ]
        
        # Add the filtered related emails to our list
        filtered_emails.extend(filtered_related)
        
        # Log filtering results
        logger.info(f"Filtered out {len(related_emails) - len(filtered_related)} emails with confidence < {CONFIDENCE_THRESHOLD}")
        logger.info(f"Keeping {len(filtered_emails)} emails total (1 anchor + {len(filtered_related)} related)")
        
        # Use filtered emails for the rest of the function
        all_emails = filtered_emails
        
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
        
        # NEW: Identify and categorize important information
        logger.info("Identifying and categorizing important information")
        categories = await self._identify_categories(timeline_events, query)
        logger.info(f"Identified {len(categories)} information categories")
        
        # Build timeline data structure
        date_range = {
            "start": sorted_emails[0].get('date') if sorted_emails else None,
            "end": sorted_emails[-1].get('date') if sorted_emails else None
        }
        
        # Generate a narrative summary of the timeline
        logger.info("Generating narrative summary of the timeline")
        
        # Create a prompt for generating the summary
        summary_prompt = PromptTemplate.from_template("""
        Generate a concise summary of this email timeline in 2-3 paragraphs:
        
        Original query: {query}
        Date range: {start_date} to {end_date}
        Number of emails: {email_count}
        Main contributors: {contributors}
        Key turning points: {turning_points}
        
        Write a narrative summary that captures how this topic evolved over time,
        highlighting the main developments and changes in the conversation.
        Focus on providing context that helps the reader understand the timeline
        at a glance.
        """)
        
        # Prepare the input for the summary
        turning_point_descriptions = [
            tp.get("description", "Significant change") 
            for tp in turning_points if "description" in tp
        ]
        
        top_contributors = [
            c["name"] for c in sorted(
                contributors, 
                key=lambda x: x["email_count"], 
                reverse=True
            )[:3]
        ] if contributors else ["Unknown"]
        
        try:
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            
            summary = await summary_chain.ainvoke({
                "query": query,
                "start_date": date_range["start"] or "unknown",
                "end_date": date_range["end"] or "unknown",
                "email_count": len(sorted_emails),
                "contributors": ", ".join(top_contributors),
                "turning_points": "; ".join(turning_point_descriptions) or "No major turning points identified"
            })
            
            logger.info(f"Generated summary of length {len(summary)}")
            logger.info(f"Summary preview: {summary[:100]}...")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary = f"This timeline shows the evolution of communications related to '{query}' over time, involving {len(sorted_emails)} emails from {len(contributors)} contributors."
            logger.info(f"Using fallback summary: {summary}")
        
        timeline_data = {
            "events": timeline_events,
            "contributors": contributors,
            "turning_points": turning_points,
            "query": query,
            "timeframe": timeframe,
            "email_count": len(sorted_emails),
            "date_range": date_range,
            "summary": summary,  # Add the generated summary to the timeline data
            "categories": categories  # Add the new categories to the timeline data
        }
        
        # Log detailed timeline data for debugging
        logger.info("========== DETAILED TIMELINE DATA ==========")
        logger.info(f"Query: '{query}'")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Date range: {date_range['start']} to {date_range['end']}")
        logger.info(f"Summary: {summary}")
        
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
        
        # Log categories
        logger.info(f"Categories ({len(categories)}):")
        for i, category in enumerate(categories):
            logger.info(f"  Category {i+1}: {category.get('type', 'Unknown')}")
            logger.info(f"    Items count: {len(category.get('items', []))}")
            for j, item in enumerate(category.get('items', [])):
                item_preview = item[:100] + "..." if len(item) > 100 else item
                logger.info(f"      Item {j+1}: {item_preview}")
        
        # Log contributors
        logger.info(f"Contributors ({len(contributors)}):")
        for contributor in contributors:
            logger.info(f"  {contributor['name']}: {contributor['email_count']} emails ({contributor['participation_percentage']:.1f}%)")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Timeline building completed in {total_time:.2f} seconds")
        logger.info(f"Timeline contains {len(timeline_events)} events, {len(turning_points)} turning points, {len(categories)} categories")
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
            # Use LLM to identify turning points with better content
            prompt = PromptTemplate.from_template("""
            Analyze this sequence of email events related to the query: "{query}"
            
            {events_text}
            
            Identify 1-3 key turning points in this discussion - moments where:
            - A significant decision was made
            - New critical information was introduced
            - The project direction changed
            - A problem was identified or resolved
            - A major milestone was reached
            
            For each turning point, provide:
            1. The event number (as an integer)
            2. A VERY SPECIFIC description of why it's significant (3-4 sentences)
            
            Your description should contain SPECIFIC DETAILS about what made this email important:
            - What specific decision was made?
            - What specific information was introduced?
            - How did it specifically change the project direction?
            - What specific problem was identified or solved?
            
            Don't use generic language - include ACTUAL details from the email content.
            
            FORMAT INSTRUCTIONS: Return your response as a JSON array with objects containing these keys: 
            - "event_index" (integer, starting from 1)
            - "description" (string with specific details)
            
            IMPORTANT: Make sure you identify at least one turning point with a DETAILED, SPECIFIC description.
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
            
            # Try to extract JSON array
            json_match = re.search(r'(\[{.*}\])', response_text, re.DOTALL)
            turning_points_raw = []
            
            if json_match:
                try:
                    matched_json = json_match.group(1)
                    logger.info(f"Found JSON match: {matched_json[:150]}...")
                    turning_points_raw = json.loads(matched_json)
                    logger.info(f"Successfully parsed JSON with {len(turning_points_raw)} turning points")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Failed JSON string: {json_match.group(1)[:150]}...")
                    # We'll handle this in the fallback
            else:
                logger.warning("No JSON array found in LLM turning point response")
            
            # Process turning points - convert event numbers to indices
            turning_points = []
            
            # If we have turning points from LLM
            if turning_points_raw:
                for point in turning_points_raw:
                    if "event_index" in point:
                        # Parse event index (adjust from 1-indexed to 0-indexed)
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
            
            # If we didn't get any turning points from the LLM, create a fallback turning point
            if not turning_points and events:
                # Use the most significant email (heuristically)
                # Look for emails with longer contributions, keywords, etc.
                
                # Rank events by potential significance
                ranked_events = []
                for i, event in enumerate(events):
                    score = 0
                    # Longer contributions are potentially more significant
                    score += len(event.get('contribution', '')) * 0.01
                    # More key points suggest significance
                    score += len(event.get('key_points', [])) * 5
                    # Subject line keywords can indicate importance
                    subject = event.get('subject', '').lower()
                    importance_keywords = ['update', 'decision', 'important', 'change', 'new', 'final']
                    for keyword in importance_keywords:
                        if keyword in subject:
                            score += 10
                    
                    ranked_events.append((i, score))
                
                # Sort by score, descending
                ranked_events.sort(key=lambda x: x[1], reverse=True)
                
                # Use the highest-scoring event as a fallback turning point
                if ranked_events:
                    best_index, _ = ranked_events[0]
                    
                    # Create a fallback turning point
                    fallback_turning_point = {
                        "event_index": best_index,
                        "date": events[best_index]["date"],
                        "sender": events[best_index]["sender"],
                        "description": f"This email from {events[best_index]['sender']} represents a key moment in the discussion about '{query}', introducing important information or changing the direction of the conversation."
                    }
                    
                    turning_points.append(fallback_turning_point)
                    logger.info(f"Created fallback turning point at event {best_index+1}: {events[best_index]['subject']}")
                else:
                    # If all else fails, use the first event
                    fallback_turning_point = {
                        "event_index": 0,
                        "date": events[0]["date"],
                        "sender": events[0]["sender"],
                        "description": f"This email begins the discussion about '{query}', setting the context for the conversation that follows."
                    }
                    
                    turning_points.append(fallback_turning_point)
                    logger.info("Created fallback turning point with first event")
            
            # Log processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Turning point identification completed in {processing_time:.2f} seconds")
            logger.info(f"Final turning points count: {len(turning_points)}")
            
            return turning_points

    async def _identify_categories(
        self,
        events: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Identify and categorize important information from the timeline events"""
        logger.info(f"Identifying categories from {len(events)} events")
        
        start_time = datetime.now()
        
        if not events or len(events) < 1:
            logger.info("Not enough events to identify categories")
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
            
            logger.info(f"Prepared events text of length {len(events_text)} for category analysis")
            
            # Use LLM to identify categories
            prompt = PromptTemplate.from_template("""
            Analyze this sequence of email events related to the query: "{query}"
            
            {events_text}
            
            Identify important information that falls into these categories:
            - Critical Information: Essential facts that users must know
            - Risks: Potential issues, concerns, or roadblocks
            - Decisions: Choices made or conclusions reached
            - Action Items: Tasks that need to be completed
            
            For each category, provide specific items extracted from the emails. Be precise and detailed.
            Do not create generic items - extract actual information from the event content.
            
            FORMAT INSTRUCTIONS: Return your response as a JSON array with objects containing these keys: 
            - "type" (string: one of the categories listed above)
            - "items" (array of strings: specific items from the emails)
            - "source_event_indices" (array of integers: the 1-based indices of events where these items were found)
            
            Example output format:
            [
              {
                "type": "Critical Information",
                "items": ["Budget cut by 25% announced on May 12", "VP will attend final presentation"],
                "source_event_indices": [3, 5]
              },
              {
                "type": "Risks",
                "items": ["Server migration delayed by 2 weeks", "Testing resources insufficient"],
                "source_event_indices": [2, 6]
              }
            ]
            
            IMPORTANT: Include at least one item for each category with SPECIFIC details from the emails.
            If you truly cannot find information for a category, you may omit that category.
            """)
            
            logger.info("Sending to LLM for category identification")
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
            
            logger.info(f"Received category analysis from LLM, length: {len(response_text)}")
            logger.info(f"Response preview: {response_text[:150]}...")
            
            # Try to extract JSON array
            json_match = re.search(r'(\[{.*}\])', response_text, re.DOTALL)
            categories_raw = []
            
            if json_match:
                try:
                    matched_json = json_match.group(1)
                    logger.info(f"Found JSON match: {matched_json[:150]}...")
                    categories_raw = json.loads(matched_json)
                    logger.info(f"Successfully parsed JSON with {len(categories_raw)} categories")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Failed JSON string: {json_match.group(1)[:150]}...")
                    # We'll handle this in the fallback
            else:
                logger.warning("No JSON array found in LLM categories response")
            
            # Process categories - adjust indices to be 0-based for frontend
            categories = []
            
            # If we have categories from LLM
            if categories_raw:
                for category in categories_raw:
                    if "type" in category and "items" in category:
                        # Convert source event indices from 1-based to 0-based
                        source_indices = []
                        if "source_event_indices" in category:
                            source_indices = [idx - 1 for idx in category["source_event_indices"] if 0 <= idx - 1 < len(events)]
                        
                        processed_category = {
                            "type": category["type"],
                            "items": category.get("items", []),
                            "source_event_indices": source_indices
                        }
                        categories.append(processed_category)
                        logger.info(f"Identified category: {category['type']} with {len(category.get('items', []))} items")
                        
                        # Log each item for detailed debugging
                        for i, item in enumerate(category.get("items", [])):
                            logger.info(f"  - Item {i+1}: {item}")
            
            # If we didn't get any categories from the LLM, create fallback categories
            if not categories and events:
                # Create basic fallback categories with heuristics
                logger.info("Using heuristic fallback for categories")
                
                # Critical Information - look for important keywords
                critical_items = []
                for i, event in enumerate(events):
                    text = (event.get('contribution', '') + ' ' + 
                           ' '.join(event.get('key_points', []))).lower()
                    if any(kw in text for kw in ['important', 'critical', 'crucial', 'essential', 'key']):
                        critical_items.append(f"From {event['sender']}: {event['subject']} ({event['date']})")
                
                # Risks - look for risk keywords
                risk_items = []
                for i, event in enumerate(events):
                    text = (event.get('contribution', '') + ' ' + 
                           ' '.join(event.get('key_points', []))).lower()
                    if any(kw in text for kw in ['risk', 'issue', 'problem', 'concern', 'delay', 'fail']):
                        risk_items.append(f"Potential issue in {event['subject']} ({event['date']})")
                
                # Decisions - look for decision keywords
                decision_items = []
                for i, event in enumerate(events):
                    text = (event.get('contribution', '') + ' ' + 
                           ' '.join(event.get('key_points', []))).lower()
                    if any(kw in text for kw in ['decide', 'decision', 'concluded', 'agreement', 'approved']):
                        decision_items.append(f"Decision in {event['subject']} ({event['date']})")
                
                # Add fallback categories if we found any items
                if critical_items:
                    categories.append({
                        "type": "Critical Information",
                        "items": critical_items,
                        "source_event_indices": []
                    })
                    logger.info(f"Added fallback 'Critical Information' category with {len(critical_items)} items")
                
                if risk_items:
                    categories.append({
                        "type": "Risks",
                        "items": risk_items,
                        "source_event_indices": []
                    })
                    logger.info(f"Added fallback 'Risks' category with {len(risk_items)} items")
                
                if decision_items:
                    categories.append({
                        "type": "Decisions",
                        "items": decision_items,
                        "source_event_indices": []
                    })
                    logger.info(f"Added fallback 'Decisions' category with {len(decision_items)} items")
                
                # If we still don't have any categories, add a generic one
                if not categories:
                    categories.append({
                        "type": "Key Information",
                        "items": [f"Timeline contains {len(events)} emails related to '{query}'"],
                        "source_event_indices": []
                    })
                    logger.info("Added generic fallback category as last resort")
            
            # Log processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Category identification completed in {processing_time:.2f} seconds")
            logger.info(f"Final categories count: {len(categories)}")
            
            return categories
                
        except Exception as e:
            logger.error(f"Error identifying categories: {str(e)}", exc_info=True)
            # Return a fallback category
            return [{
                "type": "Key Information",
                "items": [f"Timeline contains information about '{query}'"],
                "source_event_indices": []
            }]
