from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import re
from datetime import datetime

class QueryIntent(Enum):
    INSTRUCTION = "instruction"      # Needs detailed steps/procedures
    INFORMATION = "information"      # Wants to understand concepts
    CLARIFICATION = "clarification"  # Quick verification
    DISCUSSION = "discussion"        # Casual inquiry/conversation
    EMERGENCY = "emergency"          # Urgent/immediate needs

class IntentMetadata(BaseModel):
    confidence: float
    markers_found: List[str]
    context_signals: List[str]
    urgency_level: int  # 1-5

class IntentAnalysis(BaseModel):
    primary_intent: QueryIntent
    secondary_intent: Optional[QueryIntent]
    metadata: IntentMetadata

class QueryIntentAnalyzer:
    def __init__(self):
        # Intent patterns with weighted scores
        self.intent_patterns = {
            QueryIntent.INSTRUCTION: {
                'high_weight': [
                    r"how (do|should|can) (i|we|you)",
                    r"what (are|is) the (steps|procedure)",
                    r"guide me through",
                    r"show me how",
                    r"proper way to",
                ],
                'medium_weight': [
                    r"need to",
                    r"supposed to",
                    r"correct way",
                    r"help me",
                ],
                'low_weight': [
                    r"steps",
                    r"process",
                    r"procedure",
                ]
            },
            QueryIntent.INFORMATION: {
                'high_weight': [
                    r"what (is|are)",
                    r"explain",
                    r"tell me about",
                    r"describe",
                    r"definition of",
                ],
                'medium_weight': [
                    r"mean(s|ing)?",
                    r"understand",
                    r"learn about",
                ],
                'low_weight': [
                    r"info",
                    r"details",
                    r"basics",
                ]
            },
            QueryIntent.CLARIFICATION: {
                'high_weight': [
                    r"quick question",
                    r"just checking",
                    r"verify",
                    r"confirm",
                    r"is it true",
                ],
                'medium_weight': [
                    r"right\?",
                    r"correct\?",
                    r"make sure",
                ],
                'low_weight': [
                    r"\?",
                    r"wondering",
                ]
            },
            QueryIntent.DISCUSSION: {
                'high_weight': [
                    r"curious about",
                    r"wonder why",
                    r"interested in",
                    r"thoughts on",
                    r"just wondering",
                ],
                'medium_weight': [
                    r"what do you think",
                    r"how come",
                    r"tell me more",
                ],
                'low_weight': [
                    r"interesting",
                    r"wondering",
                    r"general",
                ]
            },
            QueryIntent.EMERGENCY: {
                'high_weight': [
                    r"emergency",
                    r"immediate",
                    r"urgent",
                    r"right now",
                    r"asap",
                ],
                'medium_weight': [
                    r"quickly",
                    r"hurry",
                    r"fast",
                ],
                'low_weight': [
                    r"soon",
                    r"priority",
                ]
            }
        }

        # Context signal weights
        self.context_weights = {
            'time_pressure': 0.3,
            'previous_queries': 0.2,
            'equipment_active': 0.15,
            'weather_conditions': 0.1,
            'noise_level': 0.05
        }

    def analyze(self, query: str, context: Dict) -> IntentAnalysis:
        """
        Analyze query intent considering both content and context.
        """
        query_lower = query.lower()
        
        # Score each intent based on patterns
        intent_scores = self._score_intents(query_lower)
        
        # Adjust scores based on context
        adjusted_scores = self._adjust_for_context(intent_scores, context)
        
        # Get primary and secondary intents
        primary_intent, secondary_intent, markers = self._determine_intents(adjusted_scores)
        
        # Calculate confidence and collect context signals
        confidence = self._calculate_confidence(adjusted_scores, primary_intent)
        context_signals = self._collect_context_signals(context)
        urgency = self._determine_urgency(query_lower, context)

        return IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intent=secondary_intent,
            metadata=IntentMetadata(
                confidence=confidence,
                markers_found=markers,
                context_signals=context_signals,
                urgency_level=urgency
            )
        )

    def _score_intents(self, query: str) -> Dict[QueryIntent, float]:
        """
        Score each intent based on pattern matching.
        """
        scores = {intent: 0.0 for intent in QueryIntent}
        
        for intent, patterns in self.intent_patterns.items():
            # High weight patterns (0.4)
            for pattern in patterns['high_weight']:
                if re.search(pattern, query):
                    scores[intent] += 0.4
                    
            # Medium weight patterns (0.2)
            for pattern in patterns['medium_weight']:
                if re.search(pattern, query):
                    scores[intent] += 0.2
                    
            # Low weight patterns (0.1)
            for pattern in patterns['low_weight']:
                if re.search(pattern, query):
                    scores[intent] += 0.1
        
        return scores

    def _adjust_for_context(
        self, 
        scores: Dict[QueryIntent, float], 
        context: Dict
    ) -> Dict[QueryIntent, float]:
        """
        Adjust intent scores based on context.
        """
        adjusted = scores.copy()

        # Time pressure adjustment
        if context.get('timeOfDay'):
            time_now = datetime.strptime(context['timeOfDay'], "%H:%M").time()
            if time_now.hour in [7, 8, 16, 17]:  # Start/end of typical shift
                adjusted[QueryIntent.CLARIFICATION] *= 1.2

        # Equipment context
        if context.get('activeEquipment'):
            adjusted[QueryIntent.INSTRUCTION] *= 1.1
            if any(equip in ['crane', 'excavator', 'heavyMachinery'] 
                  for equip in context['activeEquipment']):
                adjusted[QueryIntent.EMERGENCY] *= 1.2

        # Weather conditions
        if context.get('weather', {}).get('conditions') in ['rain', 'snow', 'storm']:
            adjusted[QueryIntent.INSTRUCTION] *= 1.2
            adjusted[QueryIntent.EMERGENCY] *= 1.1

        # Noise level context
        if context.get('noiseLevel', 0) > 80:  # High noise environment
            adjusted[QueryIntent.CLARIFICATION] *= 0.8  # Less likely to be casual

        return adjusted

    def _determine_intents(
        self, 
        scores: Dict[QueryIntent, float]
    ) -> Tuple[QueryIntent, Optional[QueryIntent], List[str]]:
        """
        Determine primary and secondary intents from scores.
        """
        sorted_intents = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        markers = [
            pattern
            for intent in scores.keys()
            for weight in ['high_weight', 'medium_weight']
            for pattern in self.intent_patterns[intent][weight]
        ]

        primary = sorted_intents[0][0]
        secondary = sorted_intents[1][0] if sorted_intents[1][1] > 0.3 else None

        return primary, secondary, markers

    def _calculate_confidence(
        self, 
        scores: Dict[QueryIntent, float], 
        primary_intent: QueryIntent
    ) -> float:
        """
        Calculate confidence in the intent classification.
        """
        primary_score = scores[primary_intent]
        total_score = sum(scores.values())
        
        if total_score == 0:
            return 0.3  # Default confidence
            
        # Confidence based on primary score dominance
        confidence = primary_score / total_score
        
        # Normalize to 0.3-1.0 range
        return max(0.3, min(1.0, confidence))

    def _collect_context_signals(self, context: Dict) -> List[str]:
        """
        Collect relevant signals from context that influenced the decision.
        """
        signals = []
        
        if context.get('timeOfDay'):
            signals.append(f"Time: {context['timeOfDay']}")
            
        if context.get('activeEquipment'):
            signals.append(f"Active equipment: {', '.join(context['activeEquipment'])}")
            
        if context.get('weather'):
            signals.append(f"Weather: {context['weather'].get('conditions')}")
            
        if context.get('noiseLevel'):
            signals.append(f"Noise level: {context['noiseLevel']}dB")
            
        return signals

    def _determine_urgency(self, query: str, context: Dict) -> int:
        """
        Determine urgency level (1-5) based on query and context.
        """
        urgency = 1  # Default urgency level
        
        # Check for urgent terms
        urgent_terms = {
            'emergency': 5,
            'immediate': 4,
            'urgent': 4,
            'asap': 3,
            'quickly': 2
        }
        
        for term, level in urgent_terms.items():
            if term in query:
                urgency = max(urgency, level)

        # Context-based urgency adjustments
        if context.get('weather', {}).get('conditions') in ['storm', 'severe']:
            urgency += 1
            
        if context.get('noiseLevel', 0) > 90:  # Very high noise
            urgency += 1

        return min(5, urgency)  # Cap at 5
