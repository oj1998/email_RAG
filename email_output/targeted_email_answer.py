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
    ) -> str:
        """
        Generate a targeted answer based on specific email sources
        
        Args:
            query: The user's original query
            email_sources: Pre-retrieved relevant emails
            intent: The detected intent of the query
            
        Returns:
            A focused answer addressing the query using the provided sources
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
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return "Unable to generate a targeted answer from the sources."
        except Exception as e:
            logger.error(f"Error generating targeted answer: {e}")
            return f"Unable to generate a targeted answer due to an error: {str(e)}"
