from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel
import re
from datetime import datetime
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np
import logging

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    INSTRUCTION = "instruction"      # Needs detailed steps/procedures
    INFORMATION = "information"      # Wants to understand concepts
    CLARIFICATION = "clarification"  # Quick verification
    DISCUSSION = "discussion"        # Casual inquiry/conversation
    EMERGENCY = "emergency"          # Urgent/immediate needs
    GREETING = "greeting"            # Simple hello/hi/greeting
    SMALL_TALK = "small_talk"        # Casual conversation not requiring sources
    ACKNOWLEDGMENT = "acknowledgment"  # Thank you, ok, I understand, etc.

class IntentMetadata(BaseModel):
    confidence: float
    markers_found: List[str]
    context_signals: List[str]
    urgency_level: int  # 1-5
    reasoning: Optional[str] = None

# In query_intent.py, IntentAnalysis class
class IntentAnalysis(BaseModel):
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent] = []
    confidence: float
    reasoning: str
    metadata: Optional[Dict] = {}  # Make optional with default empty dict

class SmartQueryIntentAnalyzer:
    def __init__(self, use_embeddings: bool = False, use_llm: bool = True):
        """
        Initialize the analyzer with options for using embeddings and/or LLM.
        
        Args:
            use_embeddings: Whether to use embeddings for semantic similarity
            use_llm: Whether to use LLM for complex intent analysis
        """
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        
        # Still keep some core patterns for backup and hybrid approach
        self._initialize_pattern_dictionary()
        
        # Intent examples for semantic comparison
        self.intent_examples = {
            QueryIntent.INSTRUCTION: [
                "How do I operate this excavator?",
                "What are the steps to install drywall?",
                "Guide me through connecting these pipes.",
                "Show me how to use this drill properly.",
                "What's the correct way to pour concrete?"
            ],
            QueryIntent.INFORMATION: [
                "What is R-value in insulation?",
                "Explain the difference between load-bearing and non-load-bearing walls.",
                "Tell me about foundation types.",
                "Describe how steel reinforcement works in concrete.",
                "What are the code requirements for stairway width?"
            ],
            QueryIntent.CLARIFICATION: [
                "Is this the right screw size?",
                "Should I use galvanized nails here?",
                "Just checking - can I mix these paint colors?",
                "Verify this measurement for me.",
                "Is it OK to cut this beam?"
            ],
            QueryIntent.DISCUSSION: [
                "I'm curious about why we use copper pipes instead of PVC.",
                "What do you think about LED vs fluorescent lighting?",
                "I've been wondering about passive solar design.",
                "Tell me more about sustainable construction materials.",
                "I'm interested in learning about building automation systems."
            ],
            QueryIntent.EMERGENCY: [
                "There's a gas leak! What do I do?",
                "The scaffold is wobbling dangerously. Help!",
                "I need immediate help with a worker injury.",
                "Power line down on the site! What's the procedure?",
                "Chemical spill on the second floor. Need emergency response!"
            ],
            # Adding examples for new intent types
            QueryIntent.GREETING: [
                "Hello there",
                "Good morning",
                "Hi",
                "Hey, how's it going?",
                "Greetings"
            ],
            QueryIntent.SMALL_TALK: [
                "How are you doing today?",
                "Nice weather we're having",
                "Did you see the game last night?",
                "Just taking a break",
                "How's your day going?"
            ],
            QueryIntent.ACKNOWLEDGMENT: [
                "Thanks for the information",
                "I understand now",
                "Got it",
                "That makes sense",
                "Thank you for your help"
            ]
        }
        
        # Set up embeddings if enabled
        if self.use_embeddings:
            try:
                self.embeddings = OpenAIEmbeddings()
                self._initialize_intent_embeddings()
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")
                self.use_embeddings = False
        
        # Set up LLM if enabled
        if self.use_llm:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    temperature=0, 
                    model_name="gpt-3.5-turbo-0125"  # Model that supports function calling
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False
        
        # Context signal weights
        self.context_weights = {
            'time_pressure': 0.3,
            'previous_queries': 0.2,
            'equipment_active': 0.15, 
            'weather_conditions': 0.1,
            'noise_level': 0.05
        }
        
        

    def _initialize_pattern_dictionary(self):
        """Initialize a reduced set of core patterns for hybrid approach"""
        self.intent_patterns = {
            QueryIntent.INSTRUCTION: {
                'high_weight': [
                    r"how (do|should|can) (i|we|you)",
                    r"what (are|is) the (steps|procedure)",
                    r"guide me|show me how"
                ]
            },
            QueryIntent.INFORMATION: {
                'high_weight': [
                    r"what (is|are)",
                    r"explain|tell me about|describe"
                ]
            },
            QueryIntent.CLARIFICATION: {
                'high_weight': [
                    r"verify|confirm|is it true",
                    r"(right|correct)\?"
                ]
            },
            QueryIntent.DISCUSSION: {
                'high_weight': [
                    r"curious about|wonder why|interested in",
                    r"what do you think|how come"
                ]
            },
            QueryIntent.EMERGENCY: {
                'high_weight': [
                    r"emergency|immediate|urgent|right now|asap",
                    r"help|danger|accident"
                ]
            },
            # Adding patterns for new intent types
            QueryIntent.GREETING: {
                'high_weight': [
                    r"^(hi|hello|hey|greetings|good (morning|afternoon|evening))",
                    r"how('s| is) it going",
                    r"^what's up"
                ]
            },
            QueryIntent.SMALL_TALK: {
                'high_weight': [
                    r"how are you",
                    r"nice (day|weather)",
                    r"(how's|how is) your day",
                    r"did you (see|watch|hear)",
                    r"just chatting"
                ]
            },
            QueryIntent.ACKNOWLEDGMENT: {
                'high_weight': [
                    r"^(thanks|thank you)",
                    r"^(got it|ok|okay|alright|i see)",
                    r"^that makes sense",
                    r"^(understood|I understand)",
                    r"^appreciate (it|that)"
                ]
            }
        }

    def _initialize_intent_embeddings(self):
            """Precompute embeddings for intent examples"""
            self.example_embeddings = {}
            self.intent_centroids = {}
            
            # Compute embeddings for each example
            for intent, examples in self.intent_examples.items():
                try:
                    example_embeds = []
                    for example in examples:
                        embedding = self.embeddings.embed_query(example)
                        example_embeds.append(embedding)
                    
                    self.example_embeddings[intent] = example_embeds
                    
                    # Compute centroid (average) embedding for each intent
                    if example_embeds:
                        centroid = np.mean(example_embeds, axis=0)
                        self.intent_centroids[intent] = centroid
                except Exception as e:
                    logger.error(f"Error computing embeddings for {intent}: {e}")

    async def analyze(self, query: str, context: Dict) -> IntentAnalysis:
        """
        Analyze query intent using multiple methods and considering context.
        
        Args:
            query: The user's question
            context: Dictionary with contextual information
            
        Returns:
            IntentAnalysis: The analyzed intent with metadata
        """
        # Initialize scores for each approach
        pattern_scores = {}
        embedding_scores = {}
        llm_analysis = None
        
        # 1. Pattern-based analysis (as fallback)
        pattern_scores = self._score_with_patterns(query.lower())
        
        # 2. Embedding-based semantic similarity
        if self.use_embeddings:
            try:
                embedding_scores = await self._score_with_embeddings(query)
            except Exception as e:
                logger.warning(f"Embedding scoring failed: {e}")
                embedding_scores = {}
        
        # 3. LLM-based advanced analysis
        if self.use_llm:
            try:
                llm_analysis = await self._analyze_with_llm(query, context)
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                llm_analysis = None
        
        # 4. Combine results with weighted ensemble
        final_scores = self._combine_scores(
            pattern_scores, 
            embedding_scores, 
            llm_analysis,
            query
        )
        
        # 5. Adjust for context
        adjusted_scores = self._adjust_for_context(final_scores, context)
        
        # 6. Determine primary and secondary intents
        primary_intent, secondary_intent = self._determine_intents(adjusted_scores)
        
        # 7. Calculate confidence and collect context signals
        confidence = self._calculate_confidence(adjusted_scores, primary_intent)
        context_signals = self._collect_context_signals(context)
        urgency = self._determine_urgency(query.lower(), context)
        
        # 8. Get or generate reasoning
        reasoning = self._get_reasoning(llm_analysis, primary_intent, query)
        
        # Create and return the analysis
        logger.info(f"Intent analysis complete - Primary: {primary_intent.value}, Confidence: {confidence:.2f}, Urgency: {urgency}")
        return IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intent=secondary_intent,
            metadata=IntentMetadata(
                confidence=confidence,
                markers_found=self._get_markers(query, primary_intent),
                context_signals=context_signals,
                urgency_level=urgency,
                reasoning=reasoning
            )
        )

    def _score_with_patterns(self, query: str) -> Dict[QueryIntent, float]:
        """Use pattern matching for baseline scoring"""
        scores = {intent: 0.0 for intent in QueryIntent}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns.get('high_weight', []):
                if re.search(pattern, query, re.IGNORECASE):
                    scores[intent] += 0.4
        
        return scores

    async def _score_with_embeddings(self, query: str) -> Dict[QueryIntent, float]:
        """Use semantic similarity to example queries"""
        scores = {intent: 0.0 for intent in QueryIntent}
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarity to centroids
        for intent, centroid in self.intent_centroids.items():
            similarity = np.dot(query_embedding, centroid)
            # Normalize to 0-1 range (cosine similarity is between -1 and 1)
            normalized_similarity = (similarity + 1) / 2
            scores[intent] = normalized_similarity
            
        return scores

    async def _analyze_with_llm(self, query: str, context: Dict) -> Dict[str, Any]:
            """Use function calling for reliable structured output"""
            try:
                # Define the function schema
                function_def = {
                    "name": "analyze_intent",
                    "description": "Analyze the intent of a construction query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": ["instruction", "information", "clarification", "discussion", 
                                         "emergency", "greeting", "small_talk", "acknowledgment"],
                                "description": "The primary intent of the query"
                            },
                            "secondary_intent": {
                                "type": "string",
                                "enum": ["instruction", "information", "clarification", "discussion", 
                                         "emergency", "greeting", "small_talk", "acknowledgment", "none"],
                                "description": "Optional secondary intent"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence in the classification (0-1)"
                            },
                            "urgency": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Urgency level of the query"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation for the classification"
                            }
                        },
                        "required": ["intent", "confidence", "urgency", "reasoning"]
                    }
                }
                
                # Create the messages for the API call
                messages = [
                    {"role": "system", "content": """You analyze construction and general queries to determine user intent.
                    For simple greetings, acknowledgments, or small talk, identify those accordingly rather than 
                    trying to fit them into construction categories."""},
                    {"role": "user", "content": f"Query: {query}\nContext: {str(context)}"}
                ]
                
                # Call the OpenAI API using the correct method syntax
                response = self.llm.invoke(
                    messages,
                    functions=[function_def],
                    function_call={"name": "analyze_intent"}
                )
                
                # Extract the function call arguments
                function_call = response.additional_kwargs.get('function_call')
                if function_call and function_call.get('name') == "analyze_intent":
                    result = json.loads(function_call.get('arguments'))
                    
                    # Process result
                    result["intent"] = QueryIntent(result["intent"])
                    if result.get("secondary_intent") and result["secondary_intent"].lower() not in ("none", "null"):
                        result["secondary_intent"] = QueryIntent(result["secondary_intent"])
                    else:
                        result["secondary_intent"] = None
                        
                    return result
                else:
                    raise ValueError("No function call in response")
                    
            except Exception as e:
                logger.warning(f"LLM analysis failed: {str(e)}")
                return {
                    "intent": QueryIntent.INFORMATION,
                    "secondary_intent": None,
                    "confidence": 0.5,
                    "urgency": 1,
                    "reasoning": "Fallback due to analysis error"
                }
                
    def _combine_scores(
        self, 
        pattern_scores: Dict[QueryIntent, float],
        embedding_scores: Dict[QueryIntent, float],
        llm_analysis: Optional[Dict[str, Any]],
        query: str
    ) -> Dict[QueryIntent, float]:
        """Combine scores from multiple analysis methods"""
        # Start with empty scores
        combined_scores = {intent: 0.0 for intent in QueryIntent}
        
        # Weight factors (can be adjusted)
        weights = {
            "pattern": 0.2,    # Lower weight for pattern matching
            "embedding": 0.3,  # Medium weight for embeddings
            "llm": 0.5         # Higher weight for LLM analysis
        }
        
        # Add pattern scores
        if pattern_scores:
            for intent, score in pattern_scores.items():
                combined_scores[intent] += score * weights["pattern"]
        
        # Add embedding scores
        if embedding_scores:
            for intent, score in embedding_scores.items():
                combined_scores[intent] += score * weights["embedding"]
        
        # Add LLM scores
        if llm_analysis and "intent" in llm_analysis:
            primary_intent = llm_analysis["intent"]
            combined_scores[primary_intent] += llm_analysis.get("confidence", 0.8) * weights["llm"]
            
            # Add secondary intent if present
            if "secondary_intent" in llm_analysis and llm_analysis["secondary_intent"]:
                secondary_intent = llm_analysis["secondary_intent"]
                combined_scores[secondary_intent] += (llm_analysis.get("confidence", 0.8) * 0.5) * weights["llm"]
        
        # Special case for emergency detection - check for urgent terms directly
        emergency_terms = ["emergency", "urgent", "immediately", "danger", "accident", "injured", "fire", "now"]
        if any(term in query.lower() for term in emergency_terms):
            combined_scores[QueryIntent.EMERGENCY] = max(combined_scores[QueryIntent.EMERGENCY], 0.7)
        
        # Special case for greetings and acknowledgments - check for common terms
        greeting_terms = ["hi", "hello", "hey", "morning", "afternoon", "evening", "greetings"]
        if any(term == query.lower() or query.lower().startswith(f"{term} ") for term in greeting_terms):
            combined_scores[QueryIntent.GREETING] = max(combined_scores[QueryIntent.GREETING], 0.8)
            
        acknowledgment_terms = ["thanks", "thank you", "got it", "ok", "okay", "understood", "appreciate it"]
        if any(term == query.lower() or query.lower().startswith(f"{term} ") for term in acknowledgment_terms):
            combined_scores[QueryIntent.ACKNOWLEDGMENT] = max(combined_scores[QueryIntent.ACKNOWLEDGMENT], 0.8)
            
        logger.debug(f"Combined scores: {combined_scores}")
        return combined_scores

    def _adjust_for_context(
        self, 
        scores: Dict[QueryIntent, float], 
        context: Dict
    ) -> Dict[QueryIntent, float]:
        """Adjust scores based on contextual factors"""
        adjusted = scores.copy()
        
        # Time pressure adjustment
        if context.get('timeOfDay'):
            time_of_day = context['timeOfDay'].lower() if isinstance(context['timeOfDay'], str) else str(context['timeOfDay'])
            
            # Handle text-based time descriptions
            if time_of_day in ['morning', 'afternoon', 'evening', 'night']:
                # Apply shift-based logic for morning and afternoon/evening
                if time_of_day in ['morning']:
                    adjusted[QueryIntent.CLARIFICATION] *= 1.2  # Start of shift
                elif time_of_day in ['afternoon', 'evening']:
                    adjusted[QueryIntent.CLARIFICATION] *= 1.2  # End of shift
            else:
                # Try to parse as HH:MM format
                try:
                    time_now = datetime.strptime(time_of_day, "%H:%M").time()
                    if time_now.hour in [7, 8, 16, 17]:  # Start/end of typical shift
                        adjusted[QueryIntent.CLARIFICATION] *= 1.2
                except ValueError:
                    pass
                
        # Equipment context
        if context.get('activeEquipment'):
            adjusted[QueryIntent.INSTRUCTION] *= 1.1
            if any(equip.lower() in ['crane', 'excavator', 'heavymachinery', 'lift', 'scaffold'] 
                  for equip in context['activeEquipment']):
                adjusted[QueryIntent.EMERGENCY] *= 1.2
                
        # Weather conditions
        if context.get('weather') and isinstance(context['weather'], dict):
            conditions = context['weather'].get('conditions', '').lower()
            if conditions in ['rain', 'snow', 'storm', 'high wind', 'lightning']:
                adjusted[QueryIntent.INSTRUCTION] *= 1.2
                adjusted[QueryIntent.EMERGENCY] *= 1.1
                
        # Noise level context
        if context.get('noiseLevel', 0) > 80:  # High noise environment
            adjusted[QueryIntent.CLARIFICATION] *= 0.8  # Less likely to be casual
            
        # Consider previous queries
        if context.get('previousQueries') and isinstance(context['previousQueries'], list):
            # If previous queries were instructions, slightly boost instruction intent
            instruction_keywords = ["how", "steps", "procedure", "guide"]
            if any(any(kw in prev.lower() for kw in instruction_keywords) 
                  for prev in context['previousQueries'][-3:]):
                adjusted[QueryIntent.INSTRUCTION] *= 1.1

        logger.debug(f"Context-adjusted scores: {adjusted}")
        return adjusted

    def _determine_intents(
        self, 
        scores: Dict[QueryIntent, float]
    ) -> Tuple[QueryIntent, Optional[QueryIntent]]:
        """Determine primary and secondary intents from scores"""
        sorted_intents = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        primary = sorted_intents[0][0]
        
        # Only include secondary if it's significantly strong
        secondary = None
        if len(sorted_intents) > 1 and sorted_intents[1][1] > 0.3:
            if sorted_intents[1][1] / sorted_intents[0][1] > 0.6:  # At least 60% as strong
                secondary = sorted_intents[1][0]
                
        logger.debug(f"Selected primary intent: {primary}, secondary intent: {secondary}")        
        return primary, secondary

    def _calculate_confidence(
        self, 
        scores: Dict[QueryIntent, float], 
        primary_intent: QueryIntent
    ) -> float:
        """Calculate confidence in the intent classification"""
        primary_score = scores[primary_intent]
        total_score = sum(scores.values())
        
        if total_score == 0:
            return 0.5  # Default confidence
            
        # Confidence based on primary score dominance
        confidence = primary_score / total_score
        
        # Normalize to 0.5-1.0 range (higher minimum confidence)
        return max(0.5, min(1.0, confidence * 1.5))

    def _collect_context_signals(self, context: Dict) -> List[str]:
        """Collect relevant signals from context"""
        signals = []
        
        if context.get('timeOfDay'):
            signals.append(f"Time: {context['timeOfDay']}")
            
        if context.get('activeEquipment'):
            signals.append(f"Active equipment: {', '.join(context['activeEquipment'])}")
            
        if context.get('weather') and isinstance(context['weather'], dict):
            condition = context['weather'].get('conditions')
            temp = context['weather'].get('temperature')
            if condition and temp:
                signals.append(f"Weather: {condition}, {temp}°")
            elif condition:
                signals.append(f"Weather: {condition}")
            
        if context.get('noiseLevel'):
            signals.append(f"Noise level: {context['noiseLevel']}dB")
            
        return signals

    def _determine_urgency(self, query: str, context: Dict) -> int:
        """Determine urgency level (1-5) based on query and context"""
        urgency = 1  # Default urgency level
        
        # Check for urgent terms with weighted importance
        urgent_terms = {
            'emergency': 5,
            'immediate': 4,
            'urgent': 4,
            'danger': 5,
            'accident': 5,
            'injured': 5,
            'asap': 3,
            'quickly': 2,
            'now': 3,
            'fast': 2
        }
        
        for term, level in urgent_terms.items():
            if term in query:
                urgency = max(urgency, level)

        # Context-based urgency adjustments
        if context.get('weather') and isinstance(context['weather'], dict):
            conditions = context['weather'].get('conditions', '').lower()
            if conditions in ['storm', 'severe', 'lightning', 'high wind']:
                urgency += 1
            
        if context.get('noiseLevel', 0) > 90:  # Very high noise
            urgency += 1
            
        # Equipment-based urgency
        if context.get('activeEquipment'):
            high_risk_equipment = ['crane', 'excavator', 'lift', 'scaffold', 'forklift']
            if any(equip.lower() in high_risk_equipment for equip in context['activeEquipment']):
                urgency += 1

        # Social intents always have low urgency
        if query.lower() in ["hi", "hello", "thanks", "thank you", "ok", "got it"]:
            urgency = 1

        return min(5, urgency)  # Cap at 5
        
    def _get_markers(self, query: str, primary_intent: QueryIntent) -> List[str]:
        """Extract key phrases that indicate the intent"""
        markers = []
        
        # Find matches for the primary intent patterns
        if primary_intent in self.intent_patterns:
            for pattern in self.intent_patterns[primary_intent].get('high_weight', []):
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    markers.append(match.group(0))
        
        # Add emergency markers regardless of intent if present
        if primary_intent != QueryIntent.EMERGENCY:
            for pattern in self.intent_patterns[QueryIntent.EMERGENCY].get('high_weight', []):
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    markers.append(match.group(0))
                    
        return list(set(markers))  # Remove duplicates
    
    def _get_reasoning(self, llm_analysis: Optional[Dict], primary_intent: QueryIntent, query: str) -> str:
        """Get reasoning for the intent classification"""
        # Use LLM reasoning if available
        if llm_analysis and "reasoning" in llm_analysis:
            return llm_analysis["reasoning"]
            
        # Otherwise generate basic reasoning
        intent_reasons = {
            QueryIntent.INSTRUCTION: "The query seeks step-by-step guidance or instructions.",
            QueryIntent.INFORMATION: "The query seeks factual information or conceptual understanding.",
            QueryIntent.CLARIFICATION: "The query seeks to verify or confirm information.",
            QueryIntent.DISCUSSION: "The query invites exploration or opinion on a topic.",
            QueryIntent.EMERGENCY: "The query indicates an urgent situation requiring immediate attention.",
            QueryIntent.GREETING: "The query is a simple greeting or salutation.",
            QueryIntent.SMALL_TALK: "The query is casual conversation not requiring technical information.",
            QueryIntent.ACKNOWLEDGMENT: "The query acknowledges previous information or expresses thanks."
        }
        
        return intent_reasons.get(primary_intent, "Based on query pattern analysis")
