from typing import Dict, Optional
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import logging
import re

logger = logging.getLogger(__name__)

class QuestionType(BaseModel):
    category: str
    confidence: float
    reasoning: str
    suggested_format: Optional[Dict] = None

class ConstructionClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a construction site question classifier. 
            Analyze questions from construction workers and classify them by type.
            Consider the full context, implied urgency, and safety implications.
            
            Categories:
            - SAFETY: Questions about safety procedures, emergencies, or risks
            - EQUIPMENT: Questions about tool/equipment usage or maintenance
            - MATERIALS: Questions about specifications or compatibility
            - INSTALLATION: Questions about assembly or installation procedures
            - CODE: Questions about building codes or compliance
            - TROUBLESHOOTING: Questions about problems or issues
            - PLANNING: Questions about project sequence or timing
            - CALCULATIONS: Questions about measurements or quantities
            - ENVIRONMENTAL: Questions about weather or conditions
            - COMPARISON: Questions comparing different options, materials, or methods
            
            Return JSON with:
            {
                "category": "chosen category",
                "confidence": confidence score (0-1),
                "reasoning": "brief explanation",
                "suggested_format": {
                    "style": "step_by_step or narrative",
                    "includes": ["key elements to include"],
                    "urgency": urgency level (1-5)
                }
            }
            """),
            ("user", """Question: {question}
            Previous Context: {conversation_context}
            Current Context: {current_context}""")
        ])

    async def is_comparison_query(self, question: str) -> bool:
        """Detect if a query is asking for a comparison between sources."""
        comparison_patterns = [
            r"compare", r"difference between", r"versus", r"vs\.", 
            r"which is better", r"how does .+ differ", r"pros and cons",
            r"advantages (?:and|&) disadvantages", r"(or|vs|versus)"
        ]
        return any(re.search(pattern, question.lower()) for pattern in comparison_patterns)
        
    def is_emergency_query(self, question: str) -> bool:
        """Detect if a query is related to an emergency situation."""
        emergency_patterns = [
            r"emergency", r"urgent", r"immediately", r"right now",
            r"gas leak", r"fire", r"explosion", r"collapse", r"accident",
            r"injury", r"injured", r"hurt", r"bleeding", r"danger",
            r"hazard", r"evacuate", r"evacuation"
        ]
        return any(re.search(pattern, question.lower()) for pattern in emergency_patterns)

    async def classify_question(
        self, 
        question: str,
        conversation_context: Optional[Dict] = None,
        current_context: Optional[Dict] = None
    ) -> QuestionType:
        # Fast-track for emergency queries to ensure they always get proper classification
        if self.is_emergency_query(question):
            logger.info(f"Emergency query detected: '{question}'")
            return QuestionType(
                category="SAFETY",
                confidence=0.95,
                reasoning="Emergency-related query detected",
                suggested_format={
                    "style": "step_by_step",
                    "includes": ["immediate actions", "safety precautions", "follow-up steps"],
                    "urgency": 5,
                    "is_comparison": False
                }
            )
            
        try:
            # Check if this is a comparison question first
            is_comparison = await self.is_comparison_query(question)
            
            # Convert ConversationContext to dict if needed
            if conversation_context and hasattr(conversation_context, 'dict'):
                context_dict = conversation_context.dict()
            elif conversation_context and isinstance(conversation_context, dict):
                context_dict = conversation_context
            else:
                context_dict = None
                
            # Format the prompt with all context
            prompt = self.base_prompt.format_messages(
                question=question,
                conversation_context=json.dumps(context_dict) if context_dict else "None",
                current_context=json.dumps(current_context) if current_context else "None"
            )
            
            # Get classification from LLM
            response = await self.llm.ainvoke(prompt)
            
            # Improved JSON parsing with error handling
            try:
                # Clean and parse the JSON response
                content = response.content.strip()
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL).group(1)
                elif "```" in content:
                    content = re.search(r'```\n(.*?)\n```', content, re.DOTALL).group(1)
                
                classification = json.loads(content)
                
            except (json.JSONDecodeError, AttributeError) as json_err:
                logger.error(f"JSON parsing error: {str(json_err)}")
                logger.debug(f"Raw response content: {response.content}")
                
                # Provide a reasonable fallback classification
                return QuestionType(
                    category="GENERAL",
                    confidence=0.5,
                    reasoning="Classification parsing error, using general handling",
                    suggested_format={
                        "style": "narrative",
                        "includes": ["key information"],
                        "urgency": 3,
                        "is_comparison": is_comparison
                    }
                )
            
            # Add comparison flag to the classification if detected
            if is_comparison:
                if 'suggested_format' not in classification:
                    classification['suggested_format'] = {}
                classification['suggested_format']['is_comparison'] = True
                
                # If not already categorized as COMPARISON, consider overriding
                if classification['category'] != 'COMPARISON' and classification['confidence'] < 0.8:
                    classification['category'] = 'COMPARISON'
                    if 'reasoning' in classification:
                        classification['reasoning'] += " Detected comparison query patterns."
            
            # Validate the classification format
            if not all(key in classification for key in ["category", "confidence", "reasoning"]):
                logger.warning(f"Incomplete classification returned: {classification}")
                # Fill in missing fields
                classification["category"] = classification.get("category", "GENERAL")
                classification["confidence"] = classification.get("confidence", 0.6)
                classification["reasoning"] = classification.get("reasoning", "Partial classification")
            
            # Parse into QuestionType and apply enhancements
            result = QuestionType(**classification)
            return self._enhance_safety_confidence(result)
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            # Enhanced fallback classification with context awareness
            if self.is_emergency_query(question):
                return QuestionType(
                    category="SAFETY",
                    confidence=0.9,
                    reasoning="Emergency detected during error recovery",
                    suggested_format={
                        "style": "step_by_step", 
                        "includes": ["immediate actions", "safety precautions"],
                        "urgency": 5,
                        "is_comparison": False
                    }
                )
            
            # Regular fallback for non-emergency queries
            return QuestionType(
                category="GENERAL",
                confidence=0.3,
                reasoning="Classification error, using general handling",
                suggested_format={"is_comparison": is_comparison if 'is_comparison' in locals() else False}
            )

    def _enhance_safety_confidence(self, classification: QuestionType) -> QuestionType:
        """Boost confidence for safety-related queries"""
        if classification.category == 'SAFETY':
            classification.confidence = min(1.0, classification.confidence + 0.1)
            if classification.suggested_format:
                current_urgency = classification.suggested_format.get('urgency', 1)
                classification.suggested_format['urgency'] = max(current_urgency, 4)
        return classification
