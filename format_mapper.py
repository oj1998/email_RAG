# format_mapper.py
from typing import Dict, List, Optional
from pydantic import BaseModel
from enum import Enum

class FormatStyle(Enum):
    WARNING_FIRST = "warning_first"
    STEP_BY_STEP = "step_by_step"
    SPECIFICATION = "specification"
    NARRATIVE = "narrative"

class CategoryFormat(BaseModel):
    style: FormatStyle
    required_sections: Optional[List[str]] = None
    formatting_rules: Optional[Dict[str, str]] = None
    validation_rules: Optional[Dict[str, str]] = None

class FormatMapper:
    def __init__(self):
        # Initialize with default formatting rules
        pass

    def get_format_for_category(self, category: str) -> CategoryFormat:
        # Return appropriate format for the category
        return CategoryFormat(
            style=FormatStyle.NARRATIVE,  # default style
            required_sections=[],
            formatting_rules={},
            validation_rules={}
        )

    def apply_formatting(self, content: str, format_spec: CategoryFormat, classification: Dict) -> str:
        # Apply formatting rules
        return content

    def validate_content(self, content: str, category: str) -> List[str]:
        """
        Validate that the content contains all required sections for the given category.
        Returns a list of missing section names, or an empty list if all required sections are present.
        """
        # Get the format specification for this category
        format_spec = self.get_format_for_category(category)
        if not format_spec:
            return []  # No format specification, so no validation needed
        
        missing_sections = []
        
        # Check for required sections
        if hasattr(format_spec, 'required_sections') and format_spec.required_sections:
            for section in format_spec.required_sections:
                # Simple check for section headers in content
                section_marker = f"## {section}"
                if section_marker not in content:
                    missing_sections.append(section)
        
        return missing_sections

# query_intent.py
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel
