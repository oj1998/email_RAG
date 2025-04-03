from typing import Dict, Optional, List, Any
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import logging
import re
import time
from datetime import datetime, timedelta
import tiktoken
import asyncio
import uuid
import os

# Set up enhanced logging
logger = logging.getLogger(__name__)

# Configure a file handler for cost tracking
cost_logger = logging.getLogger("cost_tracking")
cost_logger.setLevel(logging.INFO)

# Create handler for cost logs if it doesn't exist
if not any(isinstance(h, logging.FileHandler) for h in cost_logger.handlers):
    try:
        cost_handler = logging.FileHandler("gpt4_costs.log")
        cost_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        cost_handler.setFormatter(cost_formatter)
        cost_logger.addHandler(cost_handler)
    except Exception as e:
        logger.warning(f"Could not set up cost logging file: {e}")
        
# Token counter using tiktoken
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string"""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except Exception as e:
        # Fallback to approximate counting if tiktoken fails
        logger.warning(f"Token counting error: {e}, using approximation")
        words = text.split()
        return int(len(words) * 1.3)  # Rough approximation

class ClassificationRequest(BaseModel):
    """Track details of a classification request"""
    id: str
    question: str
    context_size: int
    start_time: datetime
    completion_time: Optional[datetime] = None
    model: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    processing_time: Optional[float] = None
    estimated_cost: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None

class QuestionType(BaseModel):
    category: str
    confidence: float
    reasoning: str
    suggested_format: Optional[Dict] = None

class ConstructionClassifier:
    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        """
        Initialize the construction classifier with enhanced logging and cost tracking
        
        Args:
            use_cache: Whether to use classification caching
            cache_ttl_hours: How long to keep cache entries (in hours)
        """
        # Log initialization
        model_name = os.environ.get("CLASSIFIER_MODEL", "gpt-4")
        logger.info(f"Initializing ConstructionClassifier with model: {model_name}")
        
        # Set up the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
        # Initialize cost calculation rates
        self.cost_rates = {
            "gpt-4": {
                "input": 0.03,    # $0.03 per 1K input tokens
                "output": 0.06    # $0.06 per 1K output tokens
            },
            "gpt-3.5-turbo": {
                "input": 0.0015,  # $0.0015 per 1K input tokens
                "output": 0.002   # $0.002 per 1K output tokens
            }
        }
        
        # Usage statistics
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "error_count": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "calls_by_category": {},
            "avg_response_time": 0,
            "last_reset": datetime.now(),
            "peak_token_count": 0,
        }
        
        # Set up caching
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.classification_cache = {}  # {question_hash: (QuestionType, timestamp)}
        
        # Request tracking
        self.recent_requests = []  # Keep track of recent requests for analysis
        self.max_recent_requests = 100
        
        # Initialize the prompt template
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
        
        logger.info(f"Classifier initialized with caching={'enabled' if use_cache else 'disabled'}, TTL={cache_ttl_hours}h")

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
    
    def _check_cache(self, question: str) -> Optional[QuestionType]:
        """Check if we have a cached classification for this question"""
        if not self.use_cache:
            return None
            
        # Normalize question for cache lookup
        cache_key = question.strip().lower()
        
        # Check cache
        if cache_key in self.classification_cache:
            cached_result, timestamp = self.classification_cache[cache_key]
            
            # Check if cache entry is still valid
            if datetime.now() - timestamp < self.cache_ttl:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for question: '{question[:30]}...'")
                return cached_result
            else:
                # Remove expired entry
                del self.classification_cache[cache_key]
        
        return None
    
    def _update_cache(self, question: str, result: QuestionType) -> None:
        """Update the classification cache"""
        if not self.use_cache:
            return
            
        # Normalize question for cache lookup
        cache_key = question.strip().lower()
        
        # Add to cache with current timestamp
        self.classification_cache[cache_key] = (result, datetime.now())
        
        # Clean up old cache entries
        self._clean_cache()
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.classification_cache.items()
            if now - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.classification_cache[key]
            
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired cache entries")
    
    def _is_simple_query(self, question: str) -> Optional[QuestionType]:
        """Check if this is a simple query that doesn't need GPT-4"""
        question_lower = question.lower().strip()
        
        # Check for greetings
        greeting_patterns = [
            "hello", "hi there", "good morning", "hey", "hi", 
            "howdy", "greetings"
        ]
        if any(question_lower.startswith(pattern) for pattern in greeting_patterns):
            return QuestionType(
                category="GREETING",
                confidence=0.95,
                reasoning="Pattern-based greeting detection"
            )
            
        # Check for acknowledgments
        acknowledgment_patterns = [
            "thanks", "thank you", "appreciated", "got it", "understood",
            "okay", "ok", "sounds good", "perfect"
        ]
        if any(question_lower.startswith(pattern) for pattern in acknowledgment_patterns):
            return QuestionType(
                category="ACKNOWLEDGMENT",
                confidence=0.95,
                reasoning="Pattern-based acknowledgment detection"
            )
        
        return None
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
        """
        Estimate the cost of a GPT API call
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (gpt-4 or gpt-3.5-turbo)
            
        Returns:
            float: Estimated cost in USD
        """
        # Get correct rates for model
        model_rates = self.cost_rates.get(model, self.cost_rates["gpt-4"])
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * model_rates["input"]
        output_cost = (output_tokens / 1000) * model_rates["output"]
        
        return input_cost + output_cost

    def _update_stats(self, request: ClassificationRequest) -> None:
        """Update usage statistics based on a completed request"""
        if not request.completion_time:
            return
            
        # Update counters
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += (request.input_tokens + request.output_tokens)
        self.stats["total_input_tokens"] += request.input_tokens
        self.stats["total_output_tokens"] += request.output_tokens
        self.stats["total_cost"] += request.estimated_cost
        
        # Track peak token usage
        if request.input_tokens + request.output_tokens > self.stats["peak_token_count"]:
            self.stats["peak_token_count"] = request.input_tokens + request.output_tokens
        
        # Update category stats
        if request.category:
            self.stats["calls_by_category"][request.category] = self.stats["calls_by_category"].get(request.category, 0) + 1
        
        # Calculate running average for response time
        if request.processing_time:
            avg_time = self.stats["avg_response_time"]
            count = self.stats["total_calls"]
            self.stats["avg_response_time"] = ((avg_time * (count - 1)) + request.processing_time) / count
        
        # Update cache hits if this was a cache hit
        if request.cache_hit:
            self.stats["cache_hits"] += 1
            
        # Update error count
        if request.error:
            self.stats["error_count"] += 1
            
        # Add to recent requests
        self.recent_requests.append(request)
        
        # Keep recent requests list at max size
        if len(self.recent_requests) > self.max_recent_requests:
            self.recent_requests.pop(0)
            
        # Log detailed cost information
        cost_logger.info(
            f"CLASSIFICATION,{request.id},{request.model},{request.category or 'unknown'}," +
            f"{request.input_tokens},{request.output_tokens}," +
            f"{request.processing_time or 0:.2f},{request.estimated_cost:.6f}," +
            f"{request.cache_hit},{request.error or 'none'}"
        )

    async def classify_question(
        self, 
        question: str,
        conversation_context: Optional[Dict] = None,
        current_context: Optional[Dict] = None
    ) -> QuestionType:
        """
        Classify a construction question with enhanced logging and cost tracking
        
        Args:
            question: The question to classify
            conversation_context: Optional conversation history context
            current_context: Optional current context
            
        Returns:
            QuestionType: The classification result
        """
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())[:8]
        
        # Start timing
        start_time = datetime.now()
        
        # Create request tracking object
        context_str = json.dumps(conversation_context) if conversation_context else "None"
        current_str = json.dumps(current_context) if current_context else "None"
        
        request = ClassificationRequest(
            id=request_id,
            question=question,
            context_size=len(context_str) + len(current_str),
            start_time=start_time,
            model=self.llm.model_name,
            input_tokens=0,
            output_tokens=0
        )
        
        logger.info(f"[CLASS-{request_id}] Starting classification for: '{question[:50]}...'")
        
        # Fast-track for emergency queries to ensure they always get proper classification
        if self.is_emergency_query(question):
            logger.info(f"[CLASS-{request_id}] Emergency query detected: '{question}'")
            
            result = QuestionType(
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
            
            # Update request tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            request.completion_time = datetime.now()
            request.category = "SAFETY"
            request.confidence = 0.95
            request.processing_time = processing_time
            request.estimated_cost = 0  # No API call made
            
            # Log completion
            logger.info(f"[CLASS-{request_id}] Emergency fast-track completed in {processing_time:.3f}s")
            self._update_stats(request)
            
            return result
            
        # Check for simple queries that don't need GPT-4
        simple_result = self._is_simple_query(question)
        if simple_result:
            logger.info(f"[CLASS-{request_id}] Simple query pattern detected: '{question}'")
            
            # Update request tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            request.completion_time = datetime.now()
            request.category = simple_result.category
            request.confidence = simple_result.confidence
            request.processing_time = processing_time
            request.estimated_cost = 0  # No API call made
            
            # Log completion
            logger.info(f"[CLASS-{request_id}] Simple pattern match completed in {processing_time:.3f}s")
            self._update_stats(request)
            
            return simple_result
            
        # Check cache for existing classification
        cached_result = self._check_cache(question)
        if cached_result:
            logger.info(f"[CLASS-{request_id}] Cache hit for: '{question[:50]}...'")
            
            # Update request tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            request.completion_time = datetime.now()
            request.category = cached_result.category
            request.confidence = cached_result.confidence
            request.processing_time = processing_time
            request.estimated_cost = 0  # No API call made
            request.cache_hit = True
            
            # Log completion
            logger.info(f"[CLASS-{request_id}] Cached classification retrieved in {processing_time:.3f}s")
            self._update_stats(request)
            
            return cached_result
            
        try:
            # Check if this is a comparison question
            is_comparison = await self.is_comparison_query(question)
            logger.debug(f"[CLASS-{request_id}] Comparison query check: {is_comparison}")
            
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
            
            # Convert prompt to string for token counting
            prompt_str = "\n".join([msg.content for msg in prompt])
            request.prompt_text = prompt_str
            
            # Count input tokens
            input_tokens = count_tokens(prompt_str, model=self.llm.model_name)
            request.input_tokens = input_tokens
            
            # Log before API call
            logger.info(f"[CLASS-{request_id}] Sending to {self.llm.model_name} ({input_tokens} tokens)")
            
            # Track API call start time for just the API call
            api_start_time = time.time()
            
            # Get classification from LLM
            response = await self.llm.ainvoke(prompt)
            
            # Track API call duration
            api_duration = time.time() - api_start_time
            
            # Store response
            response_text = response.content.strip()
            request.response_text = response_text
            
            # Count output tokens
            output_tokens = count_tokens(response_text, model=self.llm.model_name)
            request.output_tokens = output_tokens
            
            # Calculate cost
            cost = self._estimate_cost(input_tokens, output_tokens, model=self.llm.model_name)
            request.estimated_cost = cost
            
            # Log after API call
            logger.info(f"[CLASS-{request_id}] Received response from {self.llm.model_name} " +
                       f"in {api_duration:.3f}s ({output_tokens} tokens, est. cost: ${cost:.6f})")
            
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
                
                # Log parsed classification
                logger.info(f"[CLASS-{request_id}] Parsed classification: {classification['category']} " +
                           f"with confidence {classification['confidence']}")
                
            except (json.JSONDecodeError, AttributeError) as json_err:
                logger.error(f"[CLASS-{request_id}] JSON parsing error: {str(json_err)}")
                logger.debug(f"[CLASS-{request_id}] Raw response content: {response.content}")
                
                # Save error details
                request.error = f"JSON parsing error: {str(json_err)}"
                
                # Provide a reasonable fallback classification
                classification = {
                    "category": "GENERAL",
                    "confidence": 0.5,
                    "reasoning": "Classification parsing error, using general handling",
                    "suggested_format": {
                        "style": "narrative",
                        "includes": ["key information"],
                        "urgency": 3,
                        "is_comparison": is_comparison
                    }
                }
                
                logger.info(f"[CLASS-{request_id}] Using fallback classification: {classification['category']}")
            
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
                logger.warning(f"[CLASS-{request_id}] Incomplete classification returned: {classification}")
                # Fill in missing fields
                classification["category"] = classification.get("category", "GENERAL")
                classification["confidence"] = classification.get("confidence", 0.6)
                classification["reasoning"] = classification.get("reasoning", "Partial classification")
            
            # Parse into QuestionType and apply enhancements
            result = QuestionType(**classification)
            result = self._enhance_safety_confidence(result)
            
            # Update request tracking with final details
            processing_time = (datetime.now() - start_time).total_seconds()
            request.completion_time = datetime.now()
            request.category = result.category
            request.confidence = result.confidence
            request.processing_time = processing_time
            
            # Update cache with result
            self._update_cache(question, result)
            
            # Log success
            logger.info(f"[CLASS-{request_id}] Classification completed in {processing_time:.3f}s: " +
                       f"{result.category} (confidence: {result.confidence:.2f})")
            
            # Update stats
            self._update_stats(request)
            
            return result
            
        except Exception as e:
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save error details
            request.error = str(e)
            request.completion_time = datetime.now()
            request.processing_time = processing_time
            
            logger.error(f"[CLASS-{request_id}] Classification error: {str(e)}")
            
            # Enhanced fallback classification with context awareness
            if self.is_emergency_query(question):
                result = QuestionType(
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
                request.category = "SAFETY"
            else:
                # Regular fallback for non-emergency queries
                result = QuestionType(
                    category="GENERAL",
                    confidence=0.3,
                    reasoning="Classification error, using general handling",
                    suggested_format={"is_comparison": is_comparison if 'is_comparison' in locals() else False}
                )
                request.category = "GENERAL"
                
            request.confidence = result.confidence
            
            # Update stats including error
            self._update_stats(request)
            
            logger.info(f"[CLASS-{request_id}] Error recovery generated fallback classification: {result.category}")
            
            return result

    def _enhance_safety_confidence(self, classification: QuestionType) -> QuestionType:
        """Boost confidence for safety-related queries"""
        if classification.category == 'SAFETY':
            classification.confidence = min(1.0, classification.confidence + 0.1)
            if classification.suggested_format:
                current_urgency = classification.suggested_format.get('urgency', 1)
                classification.suggested_format['urgency'] = max(current_urgency, 4)
        return classification
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics with cost and performance metrics"""
        # Calculate additional stats
        cache_hit_rate = (self.stats["cache_hits"] / max(1, self.stats["total_calls"])) * 100
        error_rate = (self.stats["error_count"] / max(1, self.stats["total_calls"])) * 100
        
        # Calculate time since last reset
        uptime = (datetime.now() - self.stats["last_reset"]).total_seconds() / 3600  # in hours
        
        # Return comprehensive stats
        return {
            **self.stats,
            "cache_hit_rate_percent": cache_hit_rate,
            "error_rate_percent": error_rate,
            "uptime_hours": uptime,
            "cache_size": len(self.classification_cache),
            "avg_cost_per_call": self.stats["total_cost"] / max(1, self.stats["total_calls"]),
            "calls_per_hour": self.stats["total_calls"] / max(1, uptime),
            "cost_per_hour": self.stats["total_cost"] / max(1, uptime),
            "recent_calls": [
                {
                    "id": req.id,
                    "question": req.question[:30] + "..." if len(req.question) > 30 else req.question,
                    "category": req.category,
                    "tokens": req.input_tokens + req.output_tokens,
                    "cost": req.estimated_cost,
                    "time": req.processing_time,
                    "cache_hit": req.cache_hit
                }
                for req in list(reversed(self.recent_requests))[:10]  # Get 10 most recent
            ]
        }
    
    def reset_stats(self) -> Dict[str, Any]:
        """Reset usage statistics and return the previous values"""
        old_stats = self.get_stats()
        
        # Save stats to log file before reset
        cost_logger.info(f"STATS_RESET,{old_stats['total_calls']},{old_stats['total_cost']:.6f}," + 
                        f"{old_stats['total_tokens']},{old_stats['cache_hit_rate_percent']:.2f}%," +
                        f"{old_stats['error_rate_percent']:.2f}%,{old_stats['avg_response_time']:.3f}s")
        
        # Create new stats dictionary while preserving some values
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "error_count": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "calls_by_category": {},
            "avg_response_time": 0,
            "last_reset": datetime.now(),
            "peak_token_count": 0,
        }
        
        return old_stats
    
    def get_cached_categories(self) -> Dict[str, int]:
        """Get statistics about cached categories"""
        category_counts = {}
        
        for _, (classification, _) in self.classification_cache.items():
            category = classification.category
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return category_counts
