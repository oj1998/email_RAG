from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import re
import json
from datetime import datetime
import logging
import asyncio

# Reuse OpenAI components from your existing system
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np

logger = logging.getLogger(__name__)

class EmailIntent(Enum):
    """Classification of email query intents"""
    SEARCH = "search"           # Looking for specific emails
    SUMMARIZE = "summarize"     # Wanting a summary of emails
    EXTRACT = "extract"         # Extract specific information
    ANALYZE = "analyze"         # Analyze patterns or trends
    COMPOSE = "compose"         # Help with writing emails
    LIST = "list"               # List emails matching criteria
    COUNT = "count"             # Count emails matching criteria
    FORWARD = "forward"         # Forward or share emails
    ORGANIZE = "organize"       # Organize or categorize emails
    CONVERSATIONAL = "conversational"  # General chat about emails
    TIMELINE = "timeline"

class EmailIntentMetadata(BaseModel):
    """Metadata for email intent analysis"""
    confidence: float
    markers_found: List[str]
    context_signals: List[str]
    reasoning: Optional[str] = None

class EmailIntentAnalysis(BaseModel):
    """Complete analysis of email query intent"""
    primary_intent: EmailIntent
    secondary_intent: Optional[EmailIntent] = None
    metadata: EmailIntentMetadata

class EmailIntentDetector:
    """Detect the intent of email-related queries using multiple methods"""
    
    def __init__(self, use_embeddings: bool = False, use_llm: bool = True):
        """
        Initialize the detector with analysis options
        
        Args:
            use_embeddings: Whether to use embeddings for semantic similarity
            use_llm: Whether to use LLM for advanced intent analysis
        """
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        
        # Initialize pattern dictionary for baseline detection
        self._initialize_pattern_dictionary()
        
        # Intent examples for semantic comparison
        self._initialize_intent_examples()
        
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
                self.llm = ChatOpenAI(
                    temperature=0, 
                    model_name="gpt-3.5-turbo"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False
        
        # Context signal weights
        self.context_weights = {
            'timeframe': 0.3,
            'previous_queries': 0.2,
            'sender_filter': 0.15,
            'subject_filter': 0.15,
            'attachment_filter': 0.1
        }

    def _initialize_pattern_dictionary(self):
        """Initialize patterns for detecting email intents"""
        self.intent_patterns = {
            EmailIntent.SEARCH: {
                'high_weight': [
                    r"find (email|mail|message)",
                    r"search (for|through) (email|mail|message)",
                    r"look for (email|mail|message)"
                ]
            },
            EmailIntent.SUMMARIZE: {
                'high_weight': [
                    r"summarize (email|mail|message)",
                    r"summary of (email|mail|message)",
                    r"overview of (correspondence|communication)",
                    r"brief (on|about) (email|mail|message)"
                ]
            },
            EmailIntent.EXTRACT: {
                'high_weight': [
                    r"extract (from|the)",
                    r"get the .* from (email|mail|message)",
                    r"pull .* from (email|mail|message)",
                    r"what (did|does) .* say about"
                ]
            },
            EmailIntent.ANALYZE: {
                'high_weight': [
                    r"analyze (email|mail|message|communication)",
                    r"(pattern|trend) (in|of) (email|mail|message)",
                    r"statistics (on|about) (email|mail|message)",
                    r"frequency of (email|mail|message)"
                ]
            },
            EmailIntent.COMPOSE: {
                'high_weight': [
                    r"(write|draft|compose) (an |a )?(email|mail|message)",
                    r"help (me )?(write|draft|compose)",
                    r"create (an |a )?(email|mail|message)",
                    r"send (an |a )?(email|mail|message)"
                ]
            },
            EmailIntent.LIST: {
                'high_weight': [
                    r"list (email|mail|message)",
                    r"show (me |all )?(email|mail|message)",
                    r"display (email|mail|message)",
                    r"what (email|mail|message) (do I have|are there)"
                ]
            },
            EmailIntent.COUNT: {
                'high_weight': [
                    r"count (email|mail|message)",
                    r"how many (email|mail|message)",
                    r"number of (email|mail|message)",
                    r"total (email|mail|message)"
                ]
            },
            EmailIntent.FORWARD: {
                'high_weight': [
                    r"forward (email|mail|message)",
                    r"share (email|mail|message)",
                    r"send .* to",
                    r"pass (email|mail|message) (to|along)"
                ]
            },
            EmailIntent.ORGANIZE: {
                'high_weight': [
                    r"organize (email|mail|message)",
                    r"(categorize|sort|label|folder) (email|mail|message)",
                    r"move (email|mail|message) to",
                    r"file (email|mail|message)"
                ]
            },
            EmailIntent.CONVERSATIONAL: {
                'high_weight': [
                    r"^(tell me about|talk about) (email|mail|message)",
                    r"(what|how) (do|can|should) (i|you) .* (email|mail|message)",
                    r"^(hi|hello|hey|thanks|thank you)",
                    r"^(got it|ok|okay|I understand|understood)"
                ]
            },
            EmailIntent.TIMELINE: {
                'high_weight': [
                    r"timeline (of|for)",
                    r"(evolution|history) of",
                    r"how did .* (evolve|change|develop)",
                    r"track .* over time",
                    r"(decision|discussion) (history|progress)"
                ]
            }
        }

    def _initialize_intent_examples(self):
        """Initialize examples for each intent"""
        self.intent_examples = {
            EmailIntent.SEARCH: [
                "Find emails from John about the project deadline",
                "Search for emails containing the word 'contract'",
                "Look for messages with attachments from last week",
                "Find all emails about the budget meeting",
                "Search my inbox for emails from marketing department"
            ],
            EmailIntent.SUMMARIZE: [
                "Summarize my emails from yesterday",
                "Give me a summary of communications with the client",
                "Summarize all threads related to the Johnson project",
                "Provide an overview of emails from the finance team",
                "Brief me on recent emails about the merger"
            ],
            EmailIntent.EXTRACT: [
                "Extract the meeting time from Jane's latest email",
                "Get the phone number from the email David sent yesterday",
                "Pull the project deadline from the client's message",
                "What did Mark say about the budget in his last email?",
                "Extract all URLs from emails sent by marketing"
            ],
            EmailIntent.ANALYZE: [
                "Analyze email patterns from our biggest client",
                "Show me trends in communication with the support team",
                "Analyze response times for customer inquiries",
                "Give me statistics on email volume by department",
                "Analyze frequency of emails about the new product"
            ],
            EmailIntent.COMPOSE: [
                "Write an email to schedule a meeting with the team",
                "Draft a response to the client's inquiry",
                "Help me compose a follow-up email to yesterday's meeting",
                "Create an email requesting budget approval",
                "Write a polite rejection email to the vendor"
            ],
            EmailIntent.LIST: [
                "List all unread emails from yesterday",
                "Show me emails with attachments from the legal team",
                "Display all emails about project Alpha",
                "List emails from Google received this month",
                "Show all emails marked as important"
            ],
            EmailIntent.COUNT: [
                "How many emails did I receive from clients this month?",
                "Count unread messages in my inbox",
                "Total number of emails with attachments",
                "How many messages did I exchange with Sarah last week?",
                "Count emails by sender domain"
            ],
            EmailIntent.FORWARD: [
                "Forward John's email about budget to the finance team",
                "Share that email about the new policy with my team",
                "Forward the meeting invitation to Sarah",
                "Share Marcus's presentation with the marketing department",
                "Pass along the client feedback to product development"
            ],
            EmailIntent.ORGANIZE: [
                "Organize emails from clients into folders",
                "Move all emails from Apple to the Vendors folder",
                "Sort messages by project name",
                "Create labels for different types of customer inquiries",
                "Help me categorize unread messages by priority"
            ],
            EmailIntent.CONVERSATIONAL: [
                "Do you think I get too many emails?",
                "Tell me about email best practices",
                "How should I handle overwhelming email volume?",
                "What's a good strategy for email management?",
                "Thanks for helping with my emails"
            ],

            EmailIntent.TIMELINE: [
                "Show me the timeline of the budget discussions",
                "How did the decision about the crane rental evolve?",
                "Give me a history of communications about the site safety issue",
                "Track the development of the material substitution discussion",
                "Timeline of project delay discussions with the client"
            ]
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

    async def detect_intent(self, query: str, context: Dict[str, Any] = None) -> EmailIntentAnalysis:
        """
        Detect email query intent using multiple methods
        
        Args:
            query: The user's query about emails
            context: Additional context about the conversation
            
        Returns:
            EmailIntentAnalysis object with detected intent and metadata
        """
        # Initialize default context if none provided
        if context is None:
            context = {}
            
        # 1. Pattern-based analysis (baseline)
        pattern_scores = self._score_with_patterns(query.lower())
        
        # 2. Embedding-based semantic similarity (if enabled)
        embedding_scores = {}
        if self.use_embeddings:
            try:
                embedding_scores = await self._score_with_embeddings(query)
            except Exception as e:
                logger.warning(f"Embedding scoring failed: {e}")
        
        # 3. LLM-based advanced analysis (if enabled)
        llm_analysis = None
        if self.use_llm:
            try:
                llm_analysis = await self._analyze_with_llm(query, context)
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # 4. Combine results with weighted ensemble
        combined_scores = self._combine_scores(
            pattern_scores, 
            embedding_scores, 
            llm_analysis
        )
        
        # 5. Adjust for context
        adjusted_scores = self._adjust_for_context(combined_scores, context)
        
        # 6. Determine primary and secondary intents
        primary_intent, secondary_intent = self._determine_intents(adjusted_scores)
        
        # 7. Calculate confidence and collect context signals
        confidence = self._calculate_confidence(adjusted_scores, primary_intent)
        context_signals = self._collect_context_signals(context)
        
        # 8. Get or generate reasoning
        reasoning = self._get_reasoning(llm_analysis, primary_intent, query)
        
        # 9. Log the detected intent
        logger.info(f"Email intent detected - Primary: {primary_intent.value}, Confidence: {confidence:.2f}")
        
        # Create and return the analysis
        return EmailIntentAnalysis(
            primary_intent=primary_intent,
            secondary_intent=secondary_intent,
            metadata=EmailIntentMetadata(
                confidence=confidence,
                markers_found=self._get_markers(query, primary_intent),
                context_signals=context_signals,
                reasoning=reasoning
            )
        )

    def _score_with_patterns(self, query: str) -> Dict[EmailIntent, float]:
        """Score intents based on pattern matching"""
        scores = {intent: 0.0 for intent in EmailIntent}

        timeline_matches = []
        for pattern in self.intent_patterns[EmailIntent.TIMELINE].get('high_weight', []):
            if re.search(pattern, query, re.IGNORECASE):
                match = re.search(pattern, query, re.IGNORECASE).group(0)
                timeline_matches.append(match)
                scores[EmailIntent.TIMELINE] += 0.4
        
        if timeline_matches:
            logger.info(f"Timeline intent detected through patterns: {timeline_matches}")
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns.get('high_weight', []):
                if re.search(pattern, query, re.IGNORECASE):
                    scores[intent] += 0.4
                    
        return scores

    async def _score_with_embeddings(self, query: str) -> Dict[EmailIntent, float]:
        """Score intents based on semantic similarity to examples"""
        scores = {intent: 0.0 for intent in EmailIntent}
        
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
        """Use LLM for advanced intent analysis"""
        try:
            # Define the function schema for structured output
            function_def = {
                "name": "analyze_email_intent",
                "description": "Analyze the intent of an email-related query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [intent.value for intent in EmailIntent],
                            "description": "The primary intent of the email query"
                        },
                        "secondary_intent": {
                            "type": "string",
                            "enum": [intent.value for intent in EmailIntent] + ["none"],
                            "description": "Optional secondary intent"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in the classification (0-1)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation for the classification"
                        }
                    },
                    "required": ["intent", "confidence", "reasoning"]
                }
            }
            
            # Create system message
            system_message = """You analyze email-related queries to determine user intent.
            Classify based on what the user wants to do with emails (search, summarize, etc.).
            For general questions about email use 'conversational' intent."""
            
            # Create user message with query and context
            user_message = f"Query: {query}\nContext: {str(context)}"
            
            # Call the LLM with function calling
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                functions=[function_def],
                function_call={"name": "analyze_email_intent"}
            )
            
            # Extract the function call arguments
            function_call = response.additional_kwargs.get('function_call')
            if function_call and function_call.get('name') == "analyze_email_intent":
                result = json.loads(function_call.get('arguments'))
                
                # Convert string intents to enum values
                result["intent"] = EmailIntent(result["intent"])
                if result.get("secondary_intent") and result["secondary_intent"].lower() not in ("none", "null"):
                    result["secondary_intent"] = EmailIntent(result["secondary_intent"])
                else:
                    result["secondary_intent"] = None
                    
                return result
            else:
                # Fallback if no function call in response
                return {
                    "intent": EmailIntent.CONVERSATIONAL,
                    "secondary_intent": None,
                    "confidence": 0.6,
                    "reasoning": "Fallback due to missing function call"
                }
                
        except Exception as e:
            logger.warning(f"LLM analysis failed: {str(e)}")
            # Return fallback analysis
            return {
                "intent": EmailIntent.CONVERSATIONAL,
                "secondary_intent": None,
                "confidence": 0.5,
                "reasoning": f"Fallback due to analysis error: {str(e)}"
            }

    def _combine_scores(
        self, 
        pattern_scores: Dict[EmailIntent, float],
        embedding_scores: Dict[EmailIntent, float],
        llm_analysis: Optional[Dict[str, Any]]
    ) -> Dict[EmailIntent, float]:
        """Combine scores from multiple analysis methods"""
        # Initialize combined scores
        combined_scores = {intent: 0.0 for intent in EmailIntent}
        
        # Define weights for each method
        weights = {
            "pattern": 0.3,    # Pattern matching weight
            "embedding": 0.3,  # Embedding similarity weight
            "llm": 0.4         # LLM analysis weight
        }
        
        # Add pattern scores
        for intent, score in pattern_scores.items():
            combined_scores[intent] += score * weights["pattern"]
        
        # Add embedding scores
        for intent, score in embedding_scores.items():
            combined_scores[intent] += score * weights["embedding"]
        
        # Add LLM scores
        if llm_analysis and "intent" in llm_analysis:
            primary_intent = llm_analysis["intent"]
            confidence = llm_analysis.get("confidence", 0.8)
            combined_scores[primary_intent] += confidence * weights["llm"]
            
            # Add secondary intent if present
            if "secondary_intent" in llm_analysis and llm_analysis["secondary_intent"]:
                secondary_intent = llm_analysis["secondary_intent"]
                combined_scores[secondary_intent] += (confidence * 0.5) * weights["llm"]
                
        return combined_scores

    def _adjust_for_context(
        self, 
        scores: Dict[EmailIntent, float], 
        context: Dict
    ) -> Dict[EmailIntent, float]:
        """Adjust scores based on contextual factors"""
        adjusted = scores.copy()
        
        # Adjust for timeframe mentions
        if context.get('timeframe'):
            # If timeframe is mentioned, slightly boost list/search intents
            adjusted[EmailIntent.LIST] *= 1.1
            adjusted[EmailIntent.SEARCH] *= 1.1
                
        # Adjust for filter mentions
        if context.get('email_filters'):
            filters = context['email_filters']
            if isinstance(filters, dict):
                # If specific sender/subject filters are set, boost search intent
                if filters.get('from_email') or filters.get('subject_contains'):
                    adjusted[EmailIntent.SEARCH] *= 1.2
                    
                # If date filters are set, boost analysis intent
                if filters.get('after_date') or filters.get('before_date'):
                    adjusted[EmailIntent.ANALYZE] *= 1.1
                    
                # If attachment filter is set, boost extract intent
                if filters.get('has_attachment'):
                    adjusted[EmailIntent.EXTRACT] *= 1.1
        
        # Consider conversation history (if available)
        if context.get('conversation_history'):
            history = context['conversation_history']
            if isinstance(history, list) and len(history) > 0:
                # Check last few exchanges for continuity
                try:
                    recent_messages = history[-3:]
                    
                    # If recent messages were about organizing, boost organize intent
                    organize_keywords = ["folder", "label", "categorize", "organize"]
                    if any(any(kw in msg.get('content', '').lower() for kw in organize_keywords) 
                          for msg in recent_messages if isinstance(msg, dict)):
                        adjusted[EmailIntent.ORGANIZE] *= 1.2
                        
                    # Similar for compose-related continuity  
                    compose_keywords = ["write", "draft", "compose", "email to"]
                    if any(any(kw in msg.get('content', '').lower() for kw in compose_keywords)
                          for msg in recent_messages if isinstance(msg, dict)):
                        adjusted[EmailIntent.COMPOSE] *= 1.2
                except Exception as e:
                    logger.debug(f"Error processing conversation history: {e}")
                    
        return adjusted

    def _determine_intents(
        self, 
        scores: Dict[EmailIntent, float]
    ) -> tuple[EmailIntent, Optional[EmailIntent]]:
        """Determine primary and secondary intents from scores"""
        # Sort intents by score
        sorted_intents = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get the highest scoring intent
        primary = sorted_intents[0][0]
        
        # Only set secondary intent if it's somewhat strong
        secondary = None
        if len(sorted_intents) > 1:
            # Secondary intent must be at least 75% as strong as primary
            # and have a minimum score of 0.3
            if (sorted_intents[1][1] / sorted_intents[0][1] > 0.75 and 
                sorted_intents[1][1] > 0.3):
                secondary = sorted_intents[1][0]
                
        return primary, secondary

    def _calculate_confidence(
        self, 
        scores: Dict[EmailIntent, float], 
        primary_intent: EmailIntent
    ) -> float:
        """Calculate confidence in the intent classification"""
        primary_score = scores[primary_intent]
        total_score = sum(scores.values())
        
        if total_score == 0:
            return 0.5  # Default moderate confidence
            
        # Calculate confidence based on how dominant the primary score is
        confidence = primary_score / total_score
        
        # Scale confidence to useful range (0.6-1.0)
        scaled_confidence = 0.6 + (confidence * 0.4)
        
        return min(1.0, scaled_confidence)

    def _collect_context_signals(self, context: Dict) -> List[str]:
        """Collect relevant signals from context"""
        signals = []
        
        # Extract timeframe signals
        if context.get('timeframe'):
            signals.append(f"Timeframe: {context['timeframe']}")
            
        # Extract filter signals
        if context.get('email_filters'):
            filters = context['email_filters']
            if isinstance(filters, dict):
                for k, v in filters.items():
                    if v:
                        signals.append(f"Filter: {k}={v}")
        
        # Extract device type if available
        if context.get('deviceType'):
            signals.append(f"Device: {context['deviceType']}")
            
        return signals

    def _get_markers(self, query: str, primary_intent: EmailIntent) -> List[str]:
        """Extract key phrases that indicate the intent"""
        markers = []
        
        # Find matches for the primary intent patterns
        if primary_intent in self.intent_patterns:
            for pattern in self.intent_patterns[primary_intent].get('high_weight', []):
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    markers.append(match.group(0))
                    
        return list(set(markers))  # Remove duplicates

    def _get_reasoning(
        self, 
        llm_analysis: Optional[Dict], 
        primary_intent: EmailIntent, 
        query: str
    ) -> str:
        """Get reasoning for the intent classification"""
        # Use LLM reasoning if available
        if llm_analysis and "reasoning" in llm_analysis:
            return llm_analysis["reasoning"]
            
        # Otherwise provide default reasoning
        intent_reasons = {
            EmailIntent.SEARCH: "The query is about finding specific emails.",
            EmailIntent.SUMMARIZE: "The query asks for a summary or overview of emails.",
            EmailIntent.EXTRACT: "The query wants to extract specific information from emails.",
            EmailIntent.ANALYZE: "The query asks for analysis of email patterns or trends.",
            EmailIntent.COMPOSE: "The query is about writing or creating a new email.",
            EmailIntent.LIST: "The query wants to list or display emails matching criteria.",
            EmailIntent.COUNT: "The query wants to count emails or get statistical information.",
            EmailIntent.FORWARD: "The query is about forwarding or sharing emails.",
            EmailIntent.ORGANIZE: "The query is about organizing, categorizing, or filing emails.",
            EmailIntent.CONVERSATIONAL: "The query is a general question or statement about emails."
        }
        
        return intent_reasons.get(
            primary_intent, 
            "Based on the pattern and structural analysis of the query."
        )

# Simple test function
async def test_intent_detector():
    """Test the email intent detector with sample queries"""
    detector = EmailIntentDetector(use_embeddings=False, use_llm=True)
    
    test_queries = [
        "Find emails from John about the project",
        "Summarize my emails from yesterday",
        "Extract the meeting time from Jane's email",
        "Write an email to schedule a meeting with the team",
        "How many emails did I receive last week?",
        "Tell me about email best practices"
    ]
    
    for query in test_queries:
        intent = await detector.detect_intent(query)
        print(f"Query: '{query}'")
        print(f"Intent: {intent.primary_intent.value}")
        print(f"Confidence: {intent.metadata.confidence:.2f}")
        print(f"Reasoning: {intent.metadata.reasoning}")
        print("-" * 50)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_intent_detector())
