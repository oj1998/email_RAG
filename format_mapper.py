from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from enum import Enum
import re
from query_intent import QueryIntent, IntentAnalysis

class FormatStyle(Enum):
    WARNING_FIRST = "warning_first"
    STEP_BY_STEP = "step_by_step"
    SPECIFICATION = "specification"
    NARRATIVE = "narrative"

class CategoryFormat(BaseModel):
    style: FormatStyle
    required_sections: Optional[List[str]] = []
    formatting_rules: Optional[Dict[str, str]] = {}
    validation_rules: Optional[Dict[str, str]] = {}

class FormatMapper:
    def __init__(self):
        # Define intent-specific formats
        self.intent_formats = {
            "INSTRUCTION": CategoryFormat(
                style=FormatStyle.STEP_BY_STEP,
                required_sections=["PREPARATION", "STEPS", "VERIFICATION"],
                formatting_rules={
                    "step_numbering": "required",
                    "highlight_cautions": "true"
                }
            ),
            
            "INFORMATION": CategoryFormat(
                style=FormatStyle.SPECIFICATION,
                required_sections=["OVERVIEW", "DETAILS", "RELATED INFORMATION"],
                formatting_rules={
                    "highlight_key_terms": "true",
                    "include_examples": "recommended"
                }
            ),
            
            "CLARIFICATION": CategoryFormat(
                style=FormatStyle.NARRATIVE,
                required_sections=[],  # Simple, direct answer
                formatting_rules={
                    "keep_concise": "true",
                    "highlight_answer": "true"
                }
            ),
            
            "DISCUSSION": CategoryFormat(
                style=FormatStyle.NARRATIVE,
                required_sections=[],
                formatting_rules={
                    "conversational_tone": "true"
                }
            ),
            
            "EMERGENCY": CategoryFormat(
                style=FormatStyle.WARNING_FIRST,
                required_sections=["IMMEDIATE ACTION", "SAFETY PRECAUTIONS", "NEXT STEPS"],
                formatting_rules={
                    "highlight_urgency": "true",
                    "step_numbering": "required",
                    "use_emergency_markers": "true"
                }
            ),
        }
        
        # Category-specific overrides (take precedence over intents)
        self.category_overrides = {
            "SAFETY": CategoryFormat(
                style=FormatStyle.WARNING_FIRST,
                required_sections=["WARNING", "PROCEDURE", "ADDITIONAL NOTES"],
                formatting_rules={
                    "warning_style": "bold_red",
                    "step_numbering": "required",
                    "highlight_cautions": "true"
                }
            ),
            
            "EQUIPMENT": CategoryFormat(
                style=FormatStyle.STEP_BY_STEP,
                required_sections=["PREPARATION", "OPERATION", "MAINTENANCE"],
                formatting_rules={
                    "step_numbering": "required",
                    "highlight_cautions": "true"
                }
            ),
            
            "CODE": CategoryFormat(
                style=FormatStyle.SPECIFICATION,
                required_sections=["CODE REFERENCE", "REQUIREMENTS", "COMPLIANCE NOTES"],
                formatting_rules={
                    "cite_code_sections": "required",
                    "use_block_quotes": "true"
                }
            ),

            "COMPARISON": CategoryFormat(
                style=FormatStyle.SPECIFICATION,
                required_sections=["OPTIONS", "CRITERIA", "RECOMMENDATIONS"],
                formatting_rules={
                    "use_tables": "recommended",
                    "highlight_pros_cons": "true",
                    "color_code_comparison": "optional"
                },
                validation_rules={
                    "min_options": "2",
                    "balanced_criteria": "true"
                }
            )
        }
        
        # Default format for when nothing else matches
        self.default_format = CategoryFormat(
            style=FormatStyle.NARRATIVE,
            required_sections=[]
        )

    def get_format_for_intent(self, intent: str) -> CategoryFormat:
        """Return appropriate format for the primary intent"""
        intent_upper = intent.upper()
        return self.intent_formats.get(intent_upper, self.intent_formats.get("INFORMATION"))
    
    def get_format_for_category(self, category: str) -> CategoryFormat:
        """Return appropriate format for the category"""
        category_upper = category.upper()
        return self.category_overrides.get(category_upper, self.default_format)
    
    def get_format(self, intent: Union[str, QueryIntent], category: Optional[str] = None) -> CategoryFormat:
        """Get format based on intent with possible category override"""
        # Convert QueryIntent enum to string if needed
        if isinstance(intent, QueryIntent):
            intent = intent.value
            
        # Start with intent-based format
        format_spec = self.get_format_for_intent(intent)
        
        # Apply category override if it exists
        if category and category.upper() in self.category_overrides:
            category_format = self.category_overrides[category.upper()]
            
            # Merge the formats, with category taking precedence
            merged_formatting_rules = {**(format_spec.formatting_rules or {})}
            merged_formatting_rules.update(**(category_format.formatting_rules or {}))
            
            merged_validation_rules = {**(format_spec.validation_rules or {})}
            merged_validation_rules.update(**(category_format.validation_rules or {}))
            
            format_spec = CategoryFormat(
                style=category_format.style,
                required_sections=category_format.required_sections or format_spec.required_sections,
                formatting_rules=merged_formatting_rules,
                validation_rules=merged_validation_rules
            )
            
        return format_spec

    def validate_content(self, content: str, format_spec_or_category: Optional[Union[CategoryFormat, str]] = None, 
                        category: Optional[str] = None) -> List[str]:
        """
        Validate that the content contains all required sections.
        Returns a list of missing section names, or an empty list if all required sections are present.
        """
        missing_sections = []
                            
        # Handle the case where a string (category) is passed instead of a CategoryFormat
        if isinstance(format_spec_or_category, str):
            format_spec = self.get_format_for_category(format_spec_or_category)
        else:
            format_spec = format_spec_or_category
            
        # If no format_spec provided, try to get one from category parameter
        if not format_spec and category:
            format_spec = self.get_format_for_category(category)
        
        if not format_spec or not hasattr(format_spec, 'required_sections') or not format_spec.required_sections:
            return []  # No validation needed
        
        # Check for required sections using regex to be more flexible with formatting
        for section in format_spec.required_sections:
            # Look for section headers in different formats (## Section, # Section, Section:)
            section_pattern = fr"(?:^|\n)(?:#{1,2}\s*{section}|\*\*{section}\*\*|{section}:)"
            if not re.search(section_pattern, content, re.IGNORECASE):
                missing_sections.append(section)
        
        return missing_sections

    def apply_formatting(self, content: str, format_spec: Optional[CategoryFormat] = None, 
                        category: Optional[str] = None, intent: Optional[Union[str, QueryIntent]] = None) -> str:
        """Apply appropriate formatting to the content"""
        # Determine format spec if not provided or if it's a string
        if isinstance(format_spec, str):
            # If format_spec is actually a string (likely a category name), convert it
            category = format_spec
            format_spec = None
            
        if not format_spec:
            if intent and category:
                format_spec = self.get_format(intent, category)
            elif category:
                format_spec = self.get_format_for_category(category)
            elif intent:
                format_spec = self.get_format_for_intent(intent if isinstance(intent, str) else intent.value)
            else:
                format_spec = self.default_format
        
        # Apply style-specific formatting
        if format_spec.style == FormatStyle.WARNING_FIRST:
            return self._apply_warning_first_format(content, format_spec)
        elif format_spec.style == FormatStyle.STEP_BY_STEP:
            return self._apply_step_by_step_format(content, format_spec)
        elif format_spec.style == FormatStyle.SPECIFICATION:
            return self._apply_specification_format(content, format_spec)
        else:  # Narrative style (default)
            return self._apply_narrative_format(content, format_spec)
    
    def _extract_section_content(self, content: str, section_name: str) -> Optional[str]:
        """Extract content from a specific section"""
        # Flexible matching for section headers
        section_patterns = [
            fr"## {section_name}\s*(.*?)(?=\n##|\Z)",  # ## Section
            fr"# {section_name}\s*(.*?)(?=\n#|\Z)",    # # Section
            fr"\*\*{section_name}\*\*:?\s*(.*?)(?=\n\*\*|\Z)",  # **Section**
            fr"{section_name}:\s*(.*?)(?=\n\w+:|\Z)"   # Section:
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _apply_warning_first_format(self, content: str, format_spec: CategoryFormat) -> str:
        """Format with warnings at the top, clearly highlighted"""
        # Extract warning section if present
        warning_content = None
        warning_section_names = ["WARNING", "IMMEDIATE ACTION", "DANGER", "CAUTION"]
        
        for section_name in warning_section_names:
            warning_content = self._extract_section_content(content, section_name)
            if warning_content:
                break
        
        # If warning found, move to top and format
        if warning_content:
            # Remove original warning section
            for section_name in warning_section_names:
                pattern = fr"(?:^|\n)(?:#{1,2}\s*{section_name}|\*\*{section_name}\*\*|{section_name}:).*?(?=\n#|\n\*\*|\n\w+:|\Z)"
                content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
                
            # Format warning and add to top
            formatted_warning = f"⚠️ **WARNING** ⚠️\n\n{warning_content}"
            content = formatted_warning + "\n\n" + content.strip()
        
        # Apply step formatting if required
        if "step_numbering" in format_spec.formatting_rules and format_spec.formatting_rules["step_numbering"] == "required":
            content = self._add_step_numbering(content)
            
        # Format emergency content if needed
        if "use_emergency_markers" in format_spec.formatting_rules and format_spec.formatting_rules["use_emergency_markers"] == "true":
            content = self._add_emergency_formatting(content)
            
        # Highlight cautions if needed
        if "highlight_cautions" in format_spec.formatting_rules and format_spec.formatting_rules["highlight_cautions"] == "true":
            content = self._highlight_cautions(content)
            
        return content
    
    def _apply_step_by_step_format(self, content: str, format_spec: CategoryFormat) -> str:
        """Format with clear, numbered steps"""
        # Add step numbering
        if "step_numbering" in format_spec.formatting_rules and format_spec.formatting_rules["step_numbering"] == "required":
            content = self._add_step_numbering(content)
        
        # Highlight cautions if needed
        if "highlight_cautions" in format_spec.formatting_rules and format_spec.formatting_rules["highlight_cautions"] == "true":
            content = self._highlight_cautions(content)
            
        # Ensure section headers are properly formatted
        content = self._ensure_section_formatting(content)
        
        return content
    
    def _apply_specification_format(self, content: str, format_spec: CategoryFormat) -> str:
        """Format with specifications highlighted and structured"""
        # Ensure section headers are properly formatted
        content = self._ensure_section_formatting(content)
        
        # Highlight key terms
        if "highlight_key_terms" in format_spec.formatting_rules and format_spec.formatting_rules["highlight_key_terms"] == "true":
            content = self._highlight_key_terms(content)
        
        # Add block quotes for code sections if required
        if "use_block_quotes" in format_spec.formatting_rules and format_spec.formatting_rules["use_block_quotes"] == "true":
            content = self._add_block_quotes(content)
            
        # Format specifications in key-value format
        content = self._format_specifications(content)
        
        return content
    
    def _apply_narrative_format(self, content: str, format_spec: CategoryFormat) -> str:
        """Apply minimal formatting for narrative style responses"""
        # Ensure proper paragraph breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Highlight answer if requested
        if "highlight_answer" in format_spec.formatting_rules and format_spec.formatting_rules["highlight_answer"] == "true":
            # Find the first sentence or paragraph that looks like a direct answer
            answer_match = re.search(r'^([^.!?]*[.!?])', content)
            if answer_match:
                answer = answer_match.group(1)
                content = f"**{answer}**\n\n" + content[len(answer):].lstrip()
        
        # Make concise if requested
        if "keep_concise" in format_spec.formatting_rules and format_spec.formatting_rules["keep_concise"] == "true":
            # Remove any redundant phrases
            redundant_phrases = [
                r'It is important to note that ',
                r'It should be noted that ',
                r'Keep in mind that ',
                r'Remember that ',
                r'To summarize, ',
                r'In conclusion, '
            ]
            
            for phrase in redundant_phrases:
                content = re.sub(phrase, '', content, flags=re.IGNORECASE)
        
        return content
    
    # Helper formatting methods
    
    def _add_step_numbering(self, content: str) -> str:
        """Add or fix step numbering in procedural sections"""
        # Find sections that should have numbered steps
        procedural_sections = ["PROCEDURE", "OPERATION", "STEPS", "INSTALLATION STEPS", "IMMEDIATE ACTION"]
        
        for section in procedural_sections:
            section_content = self._extract_section_content(content, section)
            if not section_content:
                continue
                
            # Split into lines
            lines = section_content.split('\n')
            
            # Add numbers to lines that look like steps
            numbered_lines = []
            step_number = 1
            in_paragraph = False
            
            for line in lines:
                stripped = line.strip()
                
                # Skip empty lines
                if not stripped:
                    numbered_lines.append(line)
                    in_paragraph = False
                    continue
                
                # If already in a paragraph, continue it
                if in_paragraph:
                    numbered_lines.append(line)
                    continue
                    
                # Check if line looks like a step (starts with bullet or number)
                if stripped.startswith("-") or stripped.startswith("*") or re.match(r'^\d+[.)]', stripped):
                    # Replace with proper numbering
                    indentation = len(line) - len(stripped)
                    space = " " * indentation
                    
                    # Find where the text starts after the bullet/number
                    text_start = re.search(r'^[-*\d.)]+ *', stripped).end()
                    numbered_line = f"{space}{step_number}. {stripped[text_start:]}"
                    
                    numbered_lines.append(numbered_line)
                    step_number += 1
                    in_paragraph = False
                elif re.match(r'^[A-Z]', stripped) and len(stripped.split()) <= 5:
                    # Short line starting with uppercase - likely a step
                    indentation = len(line) - len(stripped)
                    space = " " * indentation
                    numbered_line = f"{space}{step_number}. {stripped}"
                    
                    numbered_lines.append(numbered_line)
                    step_number += 1
                    in_paragraph = False
                else:
                    # Normal text, not a step
                    numbered_lines.append(line)
                    in_paragraph = True
            
            # Replace section content with numbered content
            numbered_content = "\n".join(numbered_lines)
            
            # Replace in original content using regex with specific section name
            pattern = fr"((?:^|\n)(?:#{1,2}\s*{section}|\*\*{section}\*\*|{section}:).*?)((?=\n#{1,2}|\n\*\*|\n\w+:|\Z))"
            replacement = fr"\1{numbered_content}\2"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
        
        return content
    
    def _highlight_cautions(self, content: str) -> str:
        """Highlight caution and warning phrases"""
        # Find caution phrases and highlight them
        caution_patterns = [
            (r'((?:^|\s)caution:?[^.!?]*[.!?])', r'⚠️ **\1**'),
            (r'((?:^|\s)warning:?[^.!?]*[.!?])', r'⚠️ **\1**'),
            (r'((?:^|\s)danger:?[^.!?]*[.!?])', r'🚫 **\1**'),
            (r'((?:^|\s)important:?[^.!?]*[.!?])', r'❗ **\1**'),
            (r'((?:^|\s)be careful[^.!?]*[.!?])', r'⚠️ **\1**'),
            (r'((?:^|\s)take care[^.!?]*[.!?])', r'⚠️ **\1**')
        ]
        
        for pattern, replacement in caution_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            
        return content
    
    def _add_emergency_formatting(self, content: str) -> str:
        """Add emergency formatting elements"""
        # Add emergency banner at top if not already present
        if not re.search(r'EMERGENCY|URGENT|IMMEDIATE ACTION', content[:200], re.IGNORECASE):
            content = "🚨 **EMERGENCY SITUATION - IMMEDIATE ACTION REQUIRED** 🚨\n\n" + content
        
        # Format any phone numbers or emergency contacts
        content = re.sub(
            r'(\d{3}[-.]?\d{3}[-.]?\d{4})', 
            r'📞 **\1**', 
            content
        )
        
        # Highlight time-sensitive phrases
        time_patterns = [
            (r'(immediately)', r'**\1**'),
            (r'(as soon as possible)', r'**\1**'),
            (r'(without delay)', r'**\1**'),
            (r'(right now)', r'**\1**')
        ]
        
        for pattern, replacement in time_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    

    def _add_block_quotes(self, content: str) -> str:
        """Add block quotes for references and code citations"""
        # Find code or standard references and add block quotes
        patterns = [
            r'((?:Section|Code|Standard|Regulation)\s+[\d\.]+[^.]*\.)',
            r'(According to [^.,]+ Standard[^.]*\.)',
            r'(As per [^.,]+ Code[^.]*\.)',
            r'(The [^.,]+ Code specifies[^.]*\.)'
        ]
        
        for pattern in patterns:
            content = re.sub(
                pattern,
                r'> \1',
                content
            )
            
        return content
    

    def _ensure_section_formatting(self, content: str) -> str:
        """Ensure section headers are consistently formatted"""
        # Find section-like headers that aren't properly formatted
        section_pattern = r'^([A-Z][A-Z\s]+)(?::|\s*$)'
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            match = re.match(section_pattern, line.strip())
            if match and not line.strip().startswith(('#', '**')):
                section_name = match.group(1).strip()
                lines[i] = f"## {section_name}"
        
        return '\n'.join(lines)

    def _format_comparisons(self, content: str) -> str:
        """Format comparison content with tables, pros/cons highlighting, and color coding"""
        
        # 1. Detect and format pros/cons lists
        pros_pattern = r'(?:Pros|Advantages|Benefits)[:;]\s*(.*?)(?=\n\n|\n(?:Cons|Disadvantages|Drawbacks):|\Z)'
        cons_pattern = r'(?:Cons|Disadvantages|Drawbacks)[:;]\s*(.*?)(?=\n\n|\Z)'
        
        # Format pros with checkmarks
        content = re.sub(
            pros_pattern,
            lambda m: "**Pros:**\n" + "\n".join([f"✅ {item.strip()}" for item in re.split(r'[-•*]\s*', m.group(1)) if item.strip()]),
            content,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # Format cons with x marks
        content = re.sub(
            cons_pattern,
            lambda m: "**Cons:**\n" + "\n".join([f"❌ {item.strip()}" for item in re.split(r'[-•*]\s*', m.group(1)) if item.strip()]),
            content,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # 2. Format option comparisons in tabular format
        # Look for potential table-like comparisons
        comparison_pattern = r'(?:Option|Alternative|Material|Method|Type)\s*\d+\s*:\s*(.*?)(?=(?:Option|Alternative|Material|Method|Type)\s*\d+\s*:|$)'
        options = re.findall(comparison_pattern, content, re.IGNORECASE)
        
        if len(options) >= 2:
            # Convert option descriptions to a more tabular format
            table_content = "| Option | Description | Key Features |\n| --- | --- | --- |\n"
            
            for i, option in enumerate(options, 1):
                # Split option text on first period to separate description from details
                parts = option.split('.', 1)
                desc = parts[0].strip()
                features = parts[1].strip() if len(parts) > 1 else ""
                
                # Extract key features if present
                features_formatted = ", ".join([f.strip() for f in features.split(',')])
                
                table_content += f"| Option {i} | {desc} | {features_formatted} |\n"
            
            # Replace the options section with the table
            for i, option in enumerate(options, 1):
                option_pattern = re.escape(f"Option {i}: {option}")
                content = re.sub(option_pattern, "", content)
                
            # Add table at the start of the OPTIONS section
            options_section_pattern = r'(## OPTIONS)'
            content = re.sub(options_section_pattern, r'\1\n\n' + table_content, content)
        
        # 3. Format criteria comparison
        criteria_pattern = r'## CRITERIA\s*(.*?)(?=\n##|\Z)'
        criteria_match = re.search(criteria_pattern, content, re.DOTALL)
        
        if criteria_match:
            criteria_text = criteria_match.group(1)
            criteria_items = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\Z)', criteria_text, re.DOTALL)
            
            if criteria_items:
                # Format criteria as a structured list
                formatted_criteria = "\n\n"
                
                for i, item in enumerate(criteria_items, 1):
                    item = item.strip()
                    if ':' in item:
                        # Format criterion with name and description
                        name, desc = item.split(':', 1)
                        formatted_criteria += f"**{i}. {name.strip()}**: {desc.strip()}\n\n"
                    else:
                        formatted_criteria += f"**{i}.** {item}\n\n"
                
                # Replace original criteria text
                content = content.replace(criteria_text, formatted_criteria)
        
        # 4. Enhance recommendations with visual markers
        recommendations_pattern = r'## RECOMMENDATIONS\s*(.*?)(?=\n##|\Z)'
        recommendations_match = re.search(recommendations_pattern, content, re.DOTALL)
        
        if recommendations_match:
            recommendations_text = recommendations_match.group(1)
            
            # Add star emoji to the best/recommended option
            best_option_pattern = r'(?:best|recommended|preferred) (?:option|choice|alternative) (?:is|would be)\s*(.*?)(?=\.|\n)'
            content = re.sub(
                best_option_pattern,
                r'best option is ⭐ \1',
                content,
                flags=re.IGNORECASE
            )
            
            # Format any decision criteria summaries
            criteria_summary_pattern = r'based on\s*(.*?),\s*(.*?)(?=\.|\n)'
            content = re.sub(
                criteria_summary_pattern,
                r'based on **\1**, \2',
                content,
                flags=re.IGNORECASE
            )
        
        return content

    def _apply_specification_format(self, content: str, format_spec: CategoryFormat) -> str:
        """Format with specifications highlighted and structured"""
        # Ensure section headers are properly formatted
        content = self._ensure_section_formatting(content)
        
        # Apply comparison formatting if needed
        if format_spec.formatting_rules.get("highlight_pros_cons") == "true":
            content = self._format_comparisons(content)
        
        # Highlight key terms
        if format_spec.formatting_rules.get("highlight_key_terms") == "true":
            content = self._highlight_key_terms(content)
        
        # Add block quotes for code sections if required
        if format_spec.formatting_rules.get("use_block_quotes") == "true":
            content = self._add_block_quotes(content)
            
        # Format specifications in key-value format
        content = self._format_specifications(content)
        
        return content
        
    def _format_specifications(self, content: str) -> str:
        """Format specification items in key-value format"""
        # Look for specification patterns like "Key: Value" or "Parameter: Value"
        spec_pattern = r'(?:^|\n)([A-Z][a-zA-Z\s]+):\s*([^\n]+)'
        
        content = re.sub(
            spec_pattern,
            r'\n**\1:** \2',
            content
        )
        
        # Format dimensions and measurements
        dimension_pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)'
        content = re.sub(
            dimension_pattern,
            r'**\1\2 × \3\4**',
            content
        )
        
        # Format numeric specifications
        numeric_spec_pattern = r'(\d+(?:\.\d+)?)\s*(PSI|kPa|MPa|kg|lb|°C|°F)'
        content = re.sub(
            numeric_spec_pattern,
            r'**\1\2**',
            content
        )
        
        return content

    def _highlight_key_terms(self, content: str) -> str:
        """Placeholder for highlighting key terms - currently does nothing"""
        return content
