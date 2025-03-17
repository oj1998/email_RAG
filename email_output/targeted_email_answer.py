# email_output/targeted_email_answer.py
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from .email_intent import EmailIntent
import logging

logger = logging.getLogger(__name__)

class TargetedEmailAnswerGenerator:
    """Generates focused answers from pre-retrieved email sources"""
    
    def __init__(self, llm=None):
        """Initialize with optional language model"""
        self.llm = llm or ChatOpenAI(temperature=0.2)
    
    async def generate_targeted_answer(
        self, 
        query: str, 
        email_sources: List[Dict[str, Any]], 
        intent: EmailIntent
    ) -> Dict[str, Any]:
        """
        Generate a targeted answer based on specific email sources
        
        Args:
            query: The user's original query
            email_sources: Pre-retrieved relevant emails
            intent: The detected intent of the query
            
        Returns:
            Dict containing the answer and any formatting metadata
        """
        try:
            # Prepare source content to feed into the LLM
            source_texts = []
            for i, email in enumerate(email_sources):
                source_text = f"EMAIL {i+1}:\n"
                source_text += f"Subject: {email.get('subject', 'No subject')}\n"
                source_text += f"From: {email.get('sender', 'Unknown')}\n"
                source_text += f"Date: {email.get('date', 'Unknown date')}\n"
                source_text += f"Content: {email.get('content', email.get('excerpt', 'No content'))}\n\n"
                source_texts.append(source_text)
            
            # Build the appropriate prompt based on intent
            intent_prompts = {
                EmailIntent.SEARCH: "Find information in these emails that directly answers the query.",
                EmailIntent.SUMMARIZE: "Summarize the key information from these emails that relates to the query.",
                EmailIntent.EXTRACT: "Extract specific details from these emails that answer the query.",
                EmailIntent.ANALYZE: "Analyze patterns or insights from these emails to answer the query.",
                EmailIntent.LIST: "List the most relevant information from these emails.",
                EmailIntent.COUNT: "Provide numerical information requested in the query based on these emails.",
                EmailIntent.FORWARD: "Identify which email(s) would be most relevant to forward based on the query.",
                EmailIntent.ORGANIZE: "Suggest how these emails could be organized based on the query.",
                EmailIntent.COMPOSE: "Draft a response that addresses the query using information from these emails.",
                EmailIntent.CONVERSATIONAL: "Answer the query conversationally using information from these emails."
            }
            
            intent_instruction = intent_prompts.get(intent, "Answer the query using information from these emails.")
            
            # Create the prompt
            template = """
            Based on the following emails, answer this query: "{query}"
            
            {intent_instruction}
            
            SOURCE EMAILS:
            {sources}
            
            ANSWER:
            """
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "query": query,
                "intent_instruction": intent_instruction,
                "sources": "\n".join(source_texts)
            })
            
            # Extract response text
            answer_text = ""
            if hasattr(response, 'content'):
                answer_text = response.content
            elif isinstance(response, str):
                answer_text = response
            else:
                answer_text = "Unable to generate a targeted answer from the sources."
            
            # Define formatting rules based on intent
            formatting_rules = self._get_formatting_rules_for_intent(intent)
            
            return {
                "content": answer_text,
                "intent": intent.value,
                "formatting_rules": formatting_rules
            }
            
        except Exception as e:
            logger.error(f"Error generating targeted answer: {e}")
            return {
                "content": f"Unable to generate a targeted answer due to an error: {str(e)}",
                "intent": intent.value,
                "formatting_rules": {"error": "true"}
            }
    
    def _get_formatting_rules_for_intent(self, intent: EmailIntent) -> Dict[str, str]:
        """Get formatting rules specific to an intent"""
        
        # Define default formatting rules
        default_rules = {
            "section_title": "Answer",
            "highlight_key_points": "false",
            "format_style": "narrative"
        }
        
        # Intent-specific rules
        intent_rules = {
            EmailIntent.SEARCH: {
                "section_title": "Search Results",
                "highlight_matches": "true",
                "format_style": "structured"
            },
            EmailIntent.SUMMARIZE: {
                "section_title": "Summary",
                "highlight_key_points": "true",
                "format_style": "summary"
            },
            EmailIntent.EXTRACT: {
                "section_title": "Extracted Information",
                "highlight_extracted_items": "true",
                "format_style": "structured"
            },
            EmailIntent.ANALYZE: {
                "section_title": "Analysis",
                "highlight_insights": "true",
                "format_style": "structured",
                "use_sections": "true"
            },
            EmailIntent.LIST: {
                "section_title": "Email List",
                "use_bullet_points": "true",
                "format_style": "bullet"
            },
            EmailIntent.COUNT: {
                "section_title": "Count Results",
                "highlight_numbers": "true",
                "format_style": "summary"
            },
            EmailIntent.FORWARD: {
                "section_title": "Forward Recommendation",
                "format_style": "structured",
                "include_header_info": "true"
            },
            EmailIntent.ORGANIZE: {
                "section_title": "Organization Suggestion",
                "use_bullet_points": "true",
                "format_style": "structured"
            },
            EmailIntent.CONVERSATIONAL: {
                "section_title": "",  # No section title for conversational
                "use_natural_language": "true",
                "format_style": "conversational"
            }
        }
        
        # Get intent-specific rules or default to empty dict
        specific_rules = intent_rules.get(intent, {})
        
        # Merge default rules with intent-specific rules
        rules = {**default_rules, **specific_rules}
        
        return rules
