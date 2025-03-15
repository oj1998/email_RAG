from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import logging
import asyncio
from langchain_openai import ChatOpenAI

# Import from our email_intent module
from .email_intent import EmailIntent

logger = logging.getLogger(__name__)

class FormatStyle(Enum):
    """Different formatting styles for email responses"""
    TABULAR = "tabular"         # Table format with rows and columns
    BULLET = "bullet"           # Bullet point list
    NARRATIVE = "narrative"     # Paragraph style narrative
    STRUCTURED = "structured"   # Structured with headings
    SUMMARY = "summary"         # Brief summary format
    CONVERSATIONAL = "conversational"  # Natural conversation style
    JSON_LIKE = "json_like"     # Formatted like JSON for technical users
    CARD = "card"               # Card-style format with clear sections

class IntentFormatMapping(BaseModel):
    """Mapping between intents and preferred format styles"""
    intent: EmailIntent
    primary_style: FormatStyle
    fallback_style: FormatStyle
    include_sources: bool = True
    include_metadata: bool = False
    max_length: Optional[int] = None
    structure_template: Optional[str] = None

class EmailSource(BaseModel):
    """Representation of an email source"""
    id: str
    sender: str
    recipient: Optional[str] = None
    subject: str
    date: str
    confidence: float
    excerpt: Optional[str] = None

class EmailResponse(BaseModel):
    """Structured response for email queries"""
    content: str
    sources: List[EmailSource] = []
    metadata: Dict[str, Any] = {}

class EmailFormatter:
    """Format responses based on detected intent"""
    
    def __init__(self, llm=None):
        """Initialize with optional language model for advanced formatting"""
        self.llm = llm or ChatOpenAI(temperature=0.2)
        self._setup_intent_mappings()
        
    def _setup_intent_mappings(self):
        """Define default intent to format mappings"""
        self.mappings = {
            EmailIntent.SEARCH: IntentFormatMapping(
                intent=EmailIntent.SEARCH,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.BULLET,
                include_sources=True,
                structure_template="# Search Results\n{content}\n\n## Sources\n{sources}"
            ),
            EmailIntent.SUMMARIZE: IntentFormatMapping(
                intent=EmailIntent.SUMMARIZE,
                primary_style=FormatStyle.SUMMARY,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=True,
                structure_template="# Email Summary\n{content}\n\n## Based On\n{sources}"
            ),
            EmailIntent.EXTRACT: IntentFormatMapping(
                intent=EmailIntent.EXTRACT,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.TABULAR,
                include_sources=True,
                structure_template="# Extracted Information\n{content}\n\n## Sources\n{sources}"
            ),
            EmailIntent.ANALYZE: IntentFormatMapping(
                intent=EmailIntent.ANALYZE,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=True,
                include_metadata=True,
                structure_template="# Analysis\n{content}\n\n## Data Sources\n{sources}"
            ),
            EmailIntent.COMPOSE: IntentFormatMapping(
                intent=EmailIntent.COMPOSE,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=False,
                structure_template="# Draft Email\n{content}"
            ),
            EmailIntent.LIST: IntentFormatMapping(
                intent=EmailIntent.LIST,
                primary_style=FormatStyle.TABULAR,
                fallback_style=FormatStyle.BULLET,
                include_sources=True,
                structure_template="# Email List\n{content}\n\n## Query Details\n{metadata}"
            ),
            EmailIntent.COUNT: IntentFormatMapping(
                intent=EmailIntent.COUNT,
                primary_style=FormatStyle.SUMMARY,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=False,
                include_metadata=True,
                structure_template="# Count Results\n{content}"
            ),
            EmailIntent.FORWARD: IntentFormatMapping(
                intent=EmailIntent.FORWARD,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=True,
                structure_template="# Email Content to Forward\n{content}\n\n## Original Email Details\n{sources}"
            ),
            EmailIntent.ORGANIZE: IntentFormatMapping(
                intent=EmailIntent.ORGANIZE,
                primary_style=FormatStyle.STRUCTURED,
                fallback_style=FormatStyle.BULLET,
                include_sources=True,
                structure_template="# Organization Suggestion\n{content}\n\n## Affected Emails\n{sources}"
            ),
            EmailIntent.CONVERSATIONAL: IntentFormatMapping(
                intent=EmailIntent.CONVERSATIONAL,
                primary_style=FormatStyle.CONVERSATIONAL,
                fallback_style=FormatStyle.NARRATIVE,
                include_sources=False,
                max_length=250
            )
        }
    
    async def format_response(self, 
                       content: str,
                       intent: Union[EmailIntent, str],
                       sources: List[Union[Dict, EmailSource]] = None,
                       metadata: Dict[str, Any] = None) -> EmailResponse:
        """
        Format a response based on detected intent
        
        Args:
            content: The main content to format
            intent: The detected intent 
            sources: Source emails that were used
            metadata: Additional metadata about the query/response
            
        Returns:
            Formatted response object
        """
        # Convert string intent to enum if needed
        if isinstance(intent, str):
            try:
                intent = EmailIntent(intent.lower())
            except ValueError:
                logger.warning(f"Unknown intent '{intent}', falling back to CONVERSATIONAL")
                intent = EmailIntent.CONVERSATIONAL
                
        # Get the format mapping for this intent
        mapping = self.mappings.get(intent, self.mappings[EmailIntent.CONVERSATIONAL])
        
        # Process sources if included
        processed_sources = []
        if mapping.include_sources and sources:
            processed_sources = self._process_sources(sources)
            
        # Apply style formatting based on the mapping
        try:
            formatted_content = await self._apply_format_style(
                content, 
                mapping.primary_style,
                intent,
                processed_sources,
                metadata,
                mapping.max_length
            )
        except Exception as e:
            logger.warning(f"Error applying primary style {mapping.primary_style}: {e}")
            # Fall back to secondary style
            formatted_content = await self._apply_format_style(
                content, 
                mapping.fallback_style,
                intent,
                processed_sources,
                metadata,
                mapping.max_length
            )
        
        # Apply structure template if available
        if mapping.structure_template:
            sources_text = self._format_sources_text(processed_sources) if processed_sources else ""
            metadata_text = self._format_metadata_text(metadata) if metadata and mapping.include_metadata else ""
            
            formatted_content = mapping.structure_template.format(
                content=formatted_content,
                sources=sources_text,
                metadata=metadata_text
            )
        
        # Construct the final response
        return EmailResponse(
            content=formatted_content,
            sources=processed_sources,
            metadata=metadata or {}
        )
    
    def _process_sources(self, sources: List[Union[Dict, EmailSource]]) -> List[EmailSource]:
        """Convert various source formats to consistent EmailSource objects"""
        processed = []
        
        for source in sources:
            # Handle dict-like objects
            if isinstance(source, dict):
                processed.append(EmailSource(
                    id=source.get("id", "unknown"),
                    sender=source.get("sender", "Unknown Sender"),
                    recipient=source.get("recipient", None),
                    subject=source.get("subject", source.get("title", "No Subject")),
                    date=source.get("date", "Unknown Date"),
                    confidence=source.get("confidence", source.get("relevance_score", 0.0)),
                    excerpt=source.get("excerpt", source.get("content", None))
                ))
            # Handle EmailSource objects
            elif isinstance(source, EmailSource):
                processed.append(source)
            
        return processed
    
    def _format_sources_text(self, sources: List[EmailSource]) -> str:
        """Format sources into readable text"""
        if not sources:
            return "No sources available"
            
        lines = []
        for i, src in enumerate(sources):
            confidence_pct = f"{src.confidence * 100:.1f}%" if src.confidence else "N/A"
            lines.append(f"{i+1}. **{src.subject}** from {src.sender} on {src.date} (Relevance: {confidence_pct})")
            if src.excerpt:
                lines.append(f"   > {src.excerpt}")
                
        return "\n".join(lines)
    
    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Format metadata into readable text"""
        if not metadata:
            return ""
            
        lines = ["**Query Details**:"]
        for key, value in metadata.items():
            # Skip complex nested structures
            if isinstance(value, (dict, list)):
                continue
            lines.append(f"- {key}: {value}")
            
        return "\n".join(lines)
    
    async def _apply_format_style(self, 
                           content: str, 
                           style: FormatStyle,
                           intent: EmailIntent,
                           sources: List[EmailSource] = None,
                           metadata: Dict[str, Any] = None,
                           max_length: Optional[int] = None) -> str:
        """Apply a specific format style to content"""
        if style == FormatStyle.TABULAR:
            return self._format_tabular(content, sources)
        elif style == FormatStyle.BULLET:
            return self._format_bullet(content)
        elif style == FormatStyle.NARRATIVE:
            return content  # Already narrative
        elif style == FormatStyle.STRUCTURED:
            return self._format_structured(content)
        elif style == FormatStyle.SUMMARY:
            return self._format_summary(content, max_length)
        elif style == FormatStyle.CONVERSATIONAL:
            return await self._format_conversational(content, intent, max_length)
        elif style == FormatStyle.JSON_LIKE:
            return self._format_json_like(content, sources, metadata)
        elif style == FormatStyle.CARD:
            return self._format_card(content, sources)
        else:
            return content  # Default to original content
    
    def _format_tabular(self, content: str, sources: List[EmailSource] = None) -> str:
        """Format content in a tabular style"""
        # If content already has a table structure, return as is
        if "|" in content and "----" in content:
            return content
            
        # Simple tabular formatting - ideally would parse the content
        # and create a proper table, but this is a basic implementation
        if sources:
            # Create a simple table for sources
            table = "| Subject | Sender | Date | Relevance |\n"
            table += "|---------|--------|------|----------|\n"
            
            for src in sources:
                confidence = f"{src.confidence * 100:.1f}%" if src.confidence else "N/A"
                table += f"| {src.subject} | {src.sender} | {src.date} | {confidence} |\n"
                
            return f"{content}\n\n{table}"
        
        # Try to extract data points from content to create a table
        import re
        
        # Look for patterns like "Label: Value" or "Key - Value"
        pairs = re.findall(r'([A-Za-z\s]+)[\s]*[:|-][\s]*([^,\n]+)', content)
        
        if pairs:
            table = "| Key | Value |\n|-----|-------|\n"
            for key, value in pairs:
                table += f"| {key.strip()} | {value.strip()} |\n"
            return table
            
        return content
    
    def _format_bullet(self, content: str) -> str:
        """Format content as bullet points"""
        # If already has bullets, return as is
        if "•" in content or "- " in content:
            return content
            
        # Convert paragraphs to bullet points
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        bullets = [f"• {p}" if not p.startswith('•') and not p.startswith('-') else p 
                  for p in paragraphs]
        
        return "\n".join(bullets)
    
    def _format_structured(self, content: str) -> str:
        """Format content with proper headers and structure"""
        # If already has headers, return as is
        if "#" in content:
            return content
            
        # Try to identify potential sections
        import re
        sections = re.findall(r'^([A-Za-z\s]+):', content, re.MULTILINE)
        
        if sections:
            structured = "# Email Information\n\n"
            # Replace section labels with headers
            current_content = content
            for section in sections:
                current_content = current_content.replace(
                    f"{section}:", 
                    f"## {section}"
                )
            structured += current_content
            return structured
        
        # If no clear sections, add a main header
        return f"# Email Information\n\n{content}"
    
    def _format_summary(self, content: str, max_length: Optional[int] = None) -> str:
        """Format content as a concise summary"""
        if max_length and len(content) > max_length:
            # Truncate at last sentence boundary before max_length
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content[:max_length])
            if sentences:
                # Remove last potentially incomplete sentence
                if len(content) > max_length:
                    sentences = sentences[:-1] 
                return ' '.join(sentences)
        
        return content
    
    async def _format_conversational(self, content: str, intent: EmailIntent, max_length: Optional[int] = None) -> str:
        """Format content in a conversational style using LLM"""
        if not self.llm:
            return self._format_summary(content, max_length)
            
        try:
            from langchain.prompts import PromptTemplate
            
            template = """
            Please reformat the following information about emails in a natural, conversational tone.
            Make it sound like you're having a friendly chat while preserving the key information.
            
            Original content: {content}
            Query intent: {intent}
            
            Conversational response:
            """
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "content": content,
                "intent": intent.value
            })
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                logger.warning("Unexpected LLM response format, using original content")
                return content
                
        except Exception as e:
            logger.error(f"Error in conversational formatting: {e}")
            return content
    
    def _format_json_like(self, content: str, sources: List[EmailSource] = None, metadata: Dict[str, Any] = None) -> str:
        """Format content in a JSON-like structure for technical users"""
        sections = [f"CONTENT: {content}"]
        
        if sources:
            source_lines = []
            for i, src in enumerate(sources):
                source_lines.append(f"  {i+1}. {src.subject} (From: {src.sender}, Date: {src.date})")
            sections.append("SOURCES:\n" + "\n".join(source_lines))
            
        if metadata:
            meta_lines = []
            for k, v in metadata.items():
                if not isinstance(v, (dict, list)):
                    meta_lines.append(f"  {k}: {v}")
            if meta_lines:
                sections.append("METADATA:\n" + "\n".join(meta_lines))
                
        return "\n\n".join(sections)
    
    def _format_card(self, content: str, sources: List[EmailSource] = None) -> str:
        """Format content in a card-like structure with dividers"""
        card = "╭─────────────────────────────────────╮\n"
        card += "│           EMAIL INFORMATION         │\n"
        card += "╰─────────────────────────────────────╯\n\n"
        
        card += content + "\n\n"
        
        if sources:
            card += "╭─────────────────────────────────────╮\n"
            card += "│              SOURCES                │\n" 
            card += "╰─────────────────────────────────────╯\n\n"
            
            for i, src in enumerate(sources):
                card += f"{i+1}. {src.subject}\n"
                card += f"   From: {src.sender} on {src.date}\n"
                if src.excerpt:
                    card += f"   Preview: {src.excerpt}\n"
                card += "\n"
                
        return card


# Simple test function
async def test_formatter():
    """Test the email formatter with a sample response"""
    formatter = EmailFormatter()
    
    sample_content = """
    I found 3 emails from John Smith about the project deadline.
    The most recent email was sent yesterday at 5:30 PM with the subject "Final Project Deadline".
    John mentioned that the deadline has been extended to May 15th due to additional requirements.
    There was also an email from John on Monday discussing the timeline adjustments.
    """
    
    sample_sources = [
        {
            "id": "email123",
            "sender": "john.smith@example.com",
            "subject": "Final Project Deadline",
            "date": "2023-05-01",
            "confidence": 0.95,
            "excerpt": "The deadline has been extended to May 15th due to additional requirements."
        },
        {
            "id": "email124",
            "sender": "john.smith@example.com",
            "subject": "Timeline Adjustments",
            "date": "2023-04-28",
            "confidence": 0.82,
            "excerpt": "We need to adjust the timeline for the following deliverables..."
        }
    ]
    
    # Test different intent formats
    intents_to_test = [
        EmailIntent.SEARCH,
        EmailIntent.SUMMARIZE,
        EmailIntent.CONVERSATIONAL
    ]
    
    for intent in intents_to_test:
        response = await formatter.format_response(
            content=sample_content,
            intent=intent,
            sources=sample_sources
        )
        
        print(f"\n--- FORMATTED FOR {intent.value.upper()} ---\n")
        print(response.content)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_formatter())
