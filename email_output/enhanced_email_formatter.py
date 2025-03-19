from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import re
import logging
from langchain_openai import ChatOpenAI

# Import from our email_intent module
from .email_intent import EmailIntent

logger = logging.getLogger(__name__)

class EmailFormatStyle(Enum):
    """Different formatting styles for email responses"""
    STRUCTURED = "structured"     # Formally structured with headers
    TABULAR = "tabular"           # Table format with rows and columns
    BULLET = "bullet"             # Bullet point list
    NARRATIVE = "narrative"       # Paragraph style narrative
    SUMMARY = "summary"           # Brief summary format
    CONVERSATIONAL = "conversational"  # Natural conversation style
    CARD = "card"                 # Card-style format with clear sections
    ERROR = "error"               # Format for error messages
    TIMELINE = "timeline"

class EmailCategoryFormat(BaseModel):
    """Format specification for email responses"""
    style: EmailFormatStyle
    required_sections: Optional[List[str]] = []
    formatting_rules: Optional[Dict[str, str]] = {}
    validation_rules: Optional[Dict[str, str]] = {}
    template: Optional[str] = None

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

class EnhancedEmailFormatter:
    """Format email responses based on intent with robust formatting options"""
    
    def __init__(self, llm=None):
        """Initialize with optional language model for advanced formatting"""
        self.llm = llm or ChatOpenAI(temperature=0.2)
        self._initialize_format_mappings()
        
    def _initialize_format_mappings(self):
        """Define intent-to-format mappings with detailed formatting rules"""
        self.intent_formats = {
            EmailIntent.SEARCH: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["SEARCH RESULTS", "SOURCES"],
                formatting_rules={
                    "highlight_matches": "true",
                    "include_relevance": "true"
                },
                template="""SEARCH RESULTS:
    {content}
    
    SOURCES:
    {sources}"""
            ),
            
            EmailIntent.SUMMARIZE: EmailCategoryFormat(
                style=EmailFormatStyle.SUMMARY,
                required_sections=["SUMMARY", "SOURCES"],
                formatting_rules={
                    "highlight_key_points": "true",
                    "include_date_range": "true"
                },
                template="""EMAIL SUMMARY:
    {content}
    
    BASED ON:
    {sources}"""
            ),
            
            EmailIntent.EXTRACT: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["EXTRACTED INFO", "SOURCES"],
                formatting_rules={
                    "highlight_extracted_items": "true",
                    "use_clear_labels": "true"
                },
                template="""EXTRACTED INFORMATION:
    {content}
    
    SOURCES:
    {sources}"""
            ),
            
            EmailIntent.ANALYZE: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["ANALYSIS", "FINDINGS", "SOURCES"],
                formatting_rules={
                    "highlight_insights": "true",
                    "use_sections": "true"
                },
                template="""ANALYSIS:
    {content}
    
    DATA SOURCES:
    {sources}"""
            ),
            
            EmailIntent.COMPOSE: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["DRAFT EMAIL"],
                formatting_rules={
                    "format_as_email": "true",
                    "include_subject_line": "true"
                },
                template="""DRAFT EMAIL:
    {content}"""
            ),
            
            EmailIntent.LIST: EmailCategoryFormat(
                style=EmailFormatStyle.BULLET,
                required_sections=["EMAIL LIST", "QUERY DETAILS"],
                formatting_rules={
                    "use_bullet_points": "true",
                    "number_items": "true"
                },
                template="""EMAIL LIST:
    {content}
    
    QUERY DETAILS:
    {metadata}"""
            ),
            
            EmailIntent.COUNT: EmailCategoryFormat(
                style=EmailFormatStyle.SUMMARY,
                required_sections=["COUNT RESULTS"],
                formatting_rules={
                    "highlight_numbers": "true",
                    "include_time_period": "true"
                },
                template="""COUNT RESULTS:
    {content}"""
            ),
            
            EmailIntent.FORWARD: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["EMAIL CONTENT", "ORIGINAL DETAILS"],
                formatting_rules={
                    "preserve_formatting": "true",
                    "include_header_info": "true"
                },
                template="""EMAIL CONTENT TO FORWARD:
    {content}
    
    ORIGINAL EMAIL DETAILS:
    {sources}"""
            ),
            
            EmailIntent.ORGANIZE: EmailCategoryFormat(
                style=EmailFormatStyle.STRUCTURED,
                required_sections=["ORGANIZATION SUGGESTION", "AFFECTED EMAILS"],
                formatting_rules={
                    "use_bullet_points": "true",
                    "group_by_category": "true"
                },
                template="""ORGANIZATION SUGGESTION:
    {content}
    
    AFFECTED EMAILS:
    {sources}"""
            ),
            
            EmailIntent.CONVERSATIONAL: EmailCategoryFormat(
                style=EmailFormatStyle.CONVERSATIONAL,
                required_sections=[],
                formatting_rules={
                    "use_natural_language": "true",
                    "keep_concise": "true"
                },
                template="{content}"
            )
        }
        
        # Also update the error format
        self.error_format = EmailCategoryFormat(
            style=EmailFormatStyle.ERROR,
            required_sections=["ERROR MESSAGE", "ALTERNATIVES"],
            formatting_rules={
                "highlight_error": "true",
                "provide_alternatives": "true"
            },
            template="""SORRY, I COULDN'T PROCESS THAT REQUEST:
    {content}
    
    WHAT YOU CAN TRY INSTEAD:
    {alternatives}"""
        )
    
    async def format_response(self, 
                       content: str,
                       intent: Union[EmailIntent, str],
                       sources: List[Union[Dict, EmailSource]] = None,
                       metadata: Dict[str, Any] = None,
                       is_error: bool = False) -> EmailResponse:
        """
        Format a response based on detected intent
        
        Args:
            content: The main content to format
            intent: The detected intent 
            sources: Source emails that were used
            metadata: Additional metadata about the query/response
            is_error: Whether this is an error response
            
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
        
        # Process sources
        processed_sources = self._process_sources(sources) if sources else []
        
        # Get format specification based on intent (or use error format)
        if is_error:
            format_spec = self.error_format
            # Generate alternatives if this is an error message
            alternatives = await self._generate_alternatives(content, intent)
        else:
            format_spec = self.intent_formats.get(intent, self.default_format)
            alternatives = None
        
        # Validate content against required sections
        missing_sections = self._validate_content(content, format_spec)
        
        # Fix content if needed
        if missing_sections:
            content = await self._fix_content_structure(content, format_spec, missing_sections)
        
        # Apply style-specific formatting
        formatted_content = await self._apply_formatting(content, format_spec)
        
        # Apply template
        if format_spec.template:
            sources_text = self._format_sources_text(processed_sources) if processed_sources else ""
            metadata_text = self._format_metadata_text(metadata) if metadata else ""
            alternatives_text = alternatives if alternatives and is_error else ""
            
            # Apply template with all available parts
            template_vars = {
                "content": formatted_content,
                "sources": sources_text,
                "metadata": metadata_text,
                "alternatives": alternatives_text
            }
            
            for key, value in template_vars.items():
                if "{" + key + "}" in format_spec.template:
                    template_vars[key] = value if value else ""
            
            final_content = format_spec.template.format(**template_vars)
        else:
            # No template, just use the formatted content
            final_content = formatted_content
        
        # Construct the final response
        return EmailResponse(
            content=final_content,
            sources=processed_sources,
            metadata=metadata or {}
        )
    
    def _process_sources(self, sources: List[Union[Dict, EmailSource]]) -> List[EmailSource]:
        """Convert various source formats to consistent EmailSource objects"""
        if not sources:
            return []
            
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
    
    def _validate_content(self, content: str, format_spec: EmailCategoryFormat) -> List[str]:
        """Validate content against required sections"""
        if not format_spec.required_sections:
            return []
            
        missing_sections = []
        for section in format_spec.required_sections:
            # Look for section headers in different formats
            section_pattern = fr"(?:^|\n)(?:#{1,2}\s*{section}|\*\*{section}\*\*|{section}:)"
            if not re.search(section_pattern, content, re.IGNORECASE):
                missing_sections.append(section)
                
        return missing_sections
    
    async def _fix_content_structure(self, content: str, format_spec: EmailCategoryFormat, missing_sections: List[str]) -> str:
        """Fix content to include missing required sections using LLM assistance"""
        if not self.llm or not missing_sections:
            return content
            
        try:
            from langchain.prompts import PromptTemplate
            
            template = """
            Restructure this content to include missing required sections.
            
            Original content: 
            {content}
            
            Missing sections: {missing_sections}
            
            Please restructure the content to include these sections while preserving all existing information.
            
            Restructured content:
            """
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "content": content,
                "missing_sections": ", ".join(missing_sections)
            })
            
            # Extract response text
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
                
        except Exception as e:
            logger.error(f"Error fixing content structure: {e}")
            
        return content
    
    async def _apply_formatting(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Apply style-specific formatting to content"""
        style = format_spec.style
        
        if style == EmailFormatStyle.TABULAR:
            return await self._format_tabular(content, format_spec)
        elif style == EmailFormatStyle.BULLET:
            return self._format_bullet(content, format_spec)
        elif style == EmailFormatStyle.STRUCTURED:
            return self._format_structured(content, format_spec)
        elif style == EmailFormatStyle.SUMMARY:
            return self._format_summary(content, format_spec)
        elif style == EmailFormatStyle.CONVERSATIONAL:
            return await self._format_conversational(content, format_spec)
        elif style == EmailFormatStyle.CARD:
            return self._format_card(content, format_spec)
        elif style == EmailFormatStyle.ERROR:
            return self._format_error(content, format_spec)
        else:  # Default to narrative
            return self._format_narrative(content, format_spec)
    
    def _format_tabular(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content in a tabular style that won't include raw formatting syntax"""
        # For LIST intent, create a clean structured format
        if format_spec.formatting_rules.get("use_table_format") == "true":
            # Try to extract list items or key-value pairs
            list_items = re.findall(r'(?:^|\n)[-•*]\s*([^\n]+)', content)
            
            if list_items:
                # Create a simple section-based structure instead of a table
                formatted = "EMAIL LIST:\n"
                for i, item in enumerate(list_items, 1):
                    formatted += f"{i}. {item.strip()}\n"
                return formatted
            
            # For key-value data, create labeled sections
            key_values = re.findall(r'(?:^|\n)([^:\n]+):\s*([^\n]+)', content)
            
            if key_values:
                formatted = ""
                for key, value in key_values:
                    formatted += f"{key.strip().upper()}:\n{value.strip()}\n\n"
                return formatted
        
        # If content already contains table-like structures, convert to simple format
        if "|" in content and "---" in content:
            # Extract table data while removing markdown syntax
            lines = content.split("\n")
            clean_lines = []
            
            for line in lines:
                if "|" in line and "---" not in line:
                    # This is a data row, extract content without pipes
                    cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                    if len(cells) >= 2:
                        clean_lines.append(f"{cells[0]}: {cells[1]}")
                    else:
                        clean_lines.append(line)
                elif "---" not in line:  # Skip separator rows
                    clean_lines.append(line)
            
            return "\n".join(clean_lines)
        
        return content
    
    def _format_bullet(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content as bullet points"""
        # Check if content already has bullet points
        if re.search(r'(?:^|\n)[-•*]', content):
            return content
            
        # Split content by sentences or line breaks
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Convert to bullet points, skipping short intro sentences
        bullets = []
        intro_added = False
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip very short sentences at the beginning (likely an intro)
            word_count = len(sentence.split())
            if not intro_added and word_count < 10:
                bullets.append(sentence)
                intro_added = True
                continue
                
            # Add bullet point
            bullets.append(f"• {sentence}")
        
        return "\n\n".join(bullets)
    
    def _format_structured(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content with proper headers and structure"""
        # Ensure section headers use markdown formatting
        content = self._ensure_section_formatting(content)
        
        # Highlight extracted items if requested
        if format_spec.formatting_rules.get("highlight_extracted_items") == "true":
            # Find key-value pairs and highlight keys
            content = re.sub(
                r'(?:^|\n)([^:\n]{2,30}):\s*([^\n]+)',
                r'\n**\1:** \2',
                content
            )
        
        # Format as email if requested (for COMPOSE intent)
        if format_spec.formatting_rules.get("format_as_email") == "true":
            if "Subject:" not in content and "To:" not in content:
                # Add basic email structure
                content = f"**To:** [Recipient]\n**Subject:** [Subject]\n\n{content}"
                
            # Make sure email elements are bold
            for field in ["To:", "From:", "Subject:", "Cc:", "Bcc:"]:
                content = content.replace(field, f"**{field}**")
        
        # Highlight matches if requested (for SEARCH intent)
        if format_spec.formatting_rules.get("highlight_matches") == "true":
            # Find likely matches (text following "found" or "matching")
            content = re.sub(
                r'(found|matching|contains)(\s+\d+)?\s+([^.,]+)',
                r'\1\2 **\3**',
                content,
                flags=re.IGNORECASE
            )
        
        return content
    
    def _format_summary(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content as a concise summary"""
        # Highlight key points if requested
        if format_spec.formatting_rules.get("highlight_key_points") == "true":
            # Find sentences with key indicators
            indicators = [
                "most importantly", "key point", "in summary", 
                "highlights", "main topic", "primarily"
            ]
            
            for indicator in indicators:
                pattern = fr'([^.!?]*{indicator}[^.!?]*[.!?])'
                content = re.sub(
                    pattern,
                    r'**\1**',
                    content,
                    flags=re.IGNORECASE
                )
        
        # Highlight numbers if requested (for COUNT intent)
        if format_spec.formatting_rules.get("highlight_numbers") == "true":
            # Find numbers in the content
            content = re.sub(
                r'(\d+)(\s+(?:emails|messages|conversations|threads))',
                r'**\1**\2',
                content,
                flags=re.IGNORECASE
            )
        
        # Include date range if requested
        if format_spec.formatting_rules.get("include_date_range") == "true" and "Date range:" not in content:
            # Look for date information in the content
            date_match = re.search(r'(?:between|from)\s+([A-Za-z]+\s+\d+)(?:\s+to|\s*-\s*)([A-Za-z]+\s+\d+)', content)
            
            if date_match:
                start_date, end_date = date_match.groups()
                date_range = f"\n\n**Date range:** {start_date} to {end_date}"
                content += date_range
        
        return content
    
    async def _format_conversational(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content in a natural, conversational style"""
        if not self.llm:
            return content
            
        if format_spec.formatting_rules.get("use_natural_language") == "true":
            try:
                from langchain.prompts import PromptTemplate
                
                template = """
                Please reformat the following information about emails in a natural, conversational tone.
                Make it sound like you're having a friendly chat while preserving all the key information.
                
                Original content: {content}
                
                Conversational response (maintain all factual information):
                """
                
                prompt = PromptTemplate.from_template(template)
                chain = prompt | self.llm
                
                response = await chain.ainvoke({"content": content})
                
                # Extract response text
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                    
            except Exception as e:
                logger.error(f"Error in conversational formatting: {e}")
        
        # Keep concise if requested
        if format_spec.formatting_rules.get("keep_concise") == "true":
            # Remove filler phrases
            filler_phrases = [
                r'I would like to ',
                r'I am happy to ',
                r'I have found ',
                r'Please note that ',
                r'It is worth mentioning that ',
                r'It is important to remember that '
            ]
            
            for phrase in filler_phrases:
                content = re.sub(phrase, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _format_card(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content in a card-like structure with dividers"""
        # Create a card-like structure with Unicode box drawing characters
        card = "╭─────────────────────────────────────╮\n"
        card += "│           EMAIL INFORMATION         │\n"
        card += "╰─────────────────────────────────────╯\n\n"
        
        card += content + "\n\n"
        
        # Add additional sections if present
        sections = re.findall(r'(?:^|\n)#{1,2}\s+([A-Z\s]+)\s*\n', content)
        
        for section in sections:
            section_pattern = fr'#{1,2}\s+{section}\s*\n(.*?)(?=(?:^|\n)#{1,2}|\Z)'
            section_match = re.search(section_pattern, content, re.DOTALL | re.MULTILINE)
            
            if section_match:
                section_content = section_match.group(1).strip()
                
                card += "╭─────────────────────────────────────╮\n"
                card += f"│{section.center(39)}│\n"
                card += "╰─────────────────────────────────────╯\n\n"
                card += section_content + "\n\n"
        
        return card
    
    def _format_error(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format error message with clear highlighting"""
        if format_spec.formatting_rules.get("highlight_error") == "true":
            # Add error emoji
            content = f"❌ {content}"
        
        return content
    
    def _format_narrative(self, content: str, format_spec: EmailCategoryFormat) -> str:
        """Format content in a simple narrative style"""
        # Just ensure proper paragraph breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content
    
    def _ensure_section_formatting(self, content: str) -> str:
        """Ensure section headers use proper markdown formatting"""
        # Find section-like headers and format with markdown
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines or already formatted headers
            if not line or line.startswith('#'):
                continue
                
            # Check for section headers (ALL CAPS followed by colon)
            if re.match(r'^[A-Z][A-Z\s]+:$', line):
                section_name = line[:-1]  # Remove the colon
                lines[i] = f"## {section_name}"
                
            # Check for section-like capitalized phrases
            elif re.match(r'^[A-Z][A-Z\s]{2,}$', line) and len(line.split()) <= 3:
                lines[i] = f"## {line}"
        
        return '\n'.join(lines)
    
    def _format_sources_text(self, sources: List[EmailSource]) -> str:
        """Format email sources into readable text"""
        if not sources:
            return "No sources available"
            
        text = ""
        for i, src in enumerate(sources, 1):
            confidence_pct = f"{src.confidence * 100:.1f}%" if src.confidence is not None else "N/A"
            text += f"{i}. **{src.subject}** from {src.sender} on {src.date} (Relevance: {confidence_pct})\n"
            
            if src.excerpt:
                text += f"   > {src.excerpt}\n"
            
            text += "\n"
                
        return text.strip()
    
    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Format metadata into readable text"""
        if not metadata:
            return ""
            
        text = ""
        for key, value in metadata.items():
            # Skip complex nested structures
            if isinstance(value, (dict, list)):
                continue
            text += f"- {key}: {value}\n"
            
        return text.strip()
    
    async def _generate_alternatives(self, error_content: str, intent: EmailIntent) -> str:
        """Generate alternative suggestions for error messages"""
        if not self.llm:
            return "- Try rephrasing your query\n- Be more specific about what you're looking for"
            
        try:
            from langchain.prompts import PromptTemplate
            
            template = """
            Based on this error message and the user's original intent,
            suggest 2-3 alternative approaches the user could try.
            
            Error message: {error_content}
            User's intent: {intent}
            
            Provide concrete, helpful alternatives:
            """
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "error_content": error_content,
                "intent": intent.value
            })
            
            # Extract response text
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
                
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            
        # Fallback alternatives
        return "- Try rephrasing your query\n- Be more specific about what you're looking for"
