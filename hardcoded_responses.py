# hardcoded_responses.py - Condensed for Simplified 4-Step Drilling Story
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

class HardcodedResponse:
    def __init__(
        self,
        query_pattern: str,
        is_regex: bool,
        response_data: Dict[str, Any],
        exact_match: bool = False,
        priority: int = 0
    ):
        self.query_pattern = query_pattern
        self.is_regex = is_regex
        self.response_data = response_data
        self.exact_match = exact_match
        self.priority = priority
    
    def matches(self, query: str) -> bool:
        """Check if the query matches this hardcoded response"""
        if self.exact_match:
            return query.strip().lower() == self.query_pattern.lower()
        elif self.is_regex:
            return bool(re.search(self.query_pattern, query, re.IGNORECASE))
        else:
            return self.query_pattern.lower() in query.lower()

# Create a registry of hardcoded responses
HARDCODED_RESPONSES: List[HardcodedResponse] = []

# ====== SIMPLIFIED 4-STEP DRILLING STORY ======

# Step 0: Initial Drilling Interface
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(drill|drilling|bore|boring).*(log|logging|record|document|start|setup|begin|assistant)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """I'm your drilling assistant, ready to document your operation step by step.

**Quick Start Examples:**
- "Starting bore at station 15+50" (Setup)
- "At 75 feet, hitting clay layer" (Progress) 
- "Steering off target, drifting left" (Issues)
- "Pullback complete, 350 feet total" (Completion)

Just describe what's happening and I'll guide you through the proper documentation.""",
            "classification": {
                "category": "DRILL_LOGGING_INTERFACE",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_LOGGING_INTERFACE",
                "query_type": "drilling_workflow",
                "render_type": "drill_logging_interface"
            }
        },
        priority=85
    )
)

# Step 1: Setup Check - "Starting new bore at station 15+50"
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(starting|begin|entry|station).*(bore|drill|drilling).*(station|at)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """Setup confirmed at station 15+50.

To complete the setup log, I need:
- Entry angle (degrees)
- Target depth (feet)

This ensures we have complete project documentation from the start.""",
            "classification": {
                "category": "DRILL_SETUP_CHECK",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_SETUP_CHECK",
                "query_type": "drilling_workflow",
                "render_type": "drill_setup_check",
                "setup_data": {
                    "station": "15+50",
                    "status": "confirmed"
                }
            }
        },
        priority=90
    )
)

# Step 2: Progress Milestone - "At 75 feet, hitting clay layer"
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(at|depth).*(feet|ft|\d+).*(clay|soil|pressure|conditions|penetration|hitting)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """Progress logged at 75 feet - clay layer encountered.

Current conditions recorded. To complete the log entry:
- Current mud pressure (psi)
- Drilling speed (good/slow/fast)

This helps track drilling efficiency and soil conditions.""",
            "classification": {
                "category": "DRILL_PROGRESS_MILESTONE",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PROGRESS_MILESTONE",
                "query_type": "drilling_workflow",
                "render_type": "drill_progress_milestone",
                "progress_data": {
                    "depth": 75,
                    "soil_type": "clay",
                    "status": "active"
                }
            }
        },
        priority=88
    )
)

# Step 3: Issue Documentation - "Steering off target, drifting left"
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(steering|correction|drift|off.target|alignment|problem|issue)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """Steering issue documented - bore drifting left of target.

To log the correction:
- Correction amount (degrees)
- Direction (left/right)

This ensures proper documentation of all steering adjustments.""",
            "classification": {
                "category": "DRILL_ISSUE_DOCUMENTATION",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_ISSUE_DOCUMENTATION",
                "query_type": "drilling_workflow",
                "render_type": "drill_issue_documentation",
                "issue_data": {
                    "type": "steering_drift",
                    "direction": "left",
                    "status": "documented"
                }
            }
        },
        priority=92
    )
)

# Step 4: Completion Summary - "Pullback complete, 350 feet total"
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(pullback|complete|completed|finished|done|total|success)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """🎉 Drilling operation complete! 

Installation successful - 350 feet total length.

**Final Documentation Options:**
- Generate comprehensive drilling report
- Add final project notes
- Start new drilling project

All drilling data has been captured and is ready for final reporting.""",
            "classification": {
                "category": "DRILL_COMPLETION_SUMMARY",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_COMPLETION_SUMMARY",
                "query_type": "drilling_workflow",
                "render_type": "drill_completion_summary",
                "completion_data": {
                    "total_length": 350,
                    "status": "complete",
                    "success": True
                }
            }
        },
        priority=87
    )
)

# ====== OTHER ESSENTIAL RESPONSES ======

# Emergency Gas Leak Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(gas.+smell|smell.+gas|gas.+leak|what.+do.+gas.+leak)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# EMERGENCY: GAS LEAK SAFETY PROTOCOL

## Immediate Actions to Take
1. **Evacuate the Area:** Get everyone out of the building immediately
2. **Call for Help:** Once at a safe distance, call emergency services (911)
3. **Do Not Create Sparks:** Don't turn on/off electrical switches, use phones inside, or light matches

## What NOT to Do
* Do not attempt to locate the leak yourself
* Do not turn any electrical equipment on or off
* Do not use your phone until you're safely away from the area

## After Evacuation
* Wait at a safe distance for emergency responders
* Follow all instructions from emergency personnel
* Do not return to the building until professionals declare it safe""",
            "classification": {
                "category": "EMERGENCY_GAS_LEAK",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "EMERGENCY_GAS_LEAK",
                "query_type": "document",
                "render_type": "safety_protocol",
                "is_gas_leak_emergency": True
            }
        },
        priority=100
    )
)

# Drywall Protection Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(gypsum|drywall).+(protect|rain|water|moisture|wet)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Protecting Installed Gypsum Board During Sudden Rainfall

## Required Equipment
- 8-mil reinforced polyethylene sheeting
- Spring clamps with rubber guards
- Waterproof tape (commercial grade)
- J-channel water diverters

## Execution Process
1. Retrieve emergency rain kit from designated weather station
2. Unroll polyethylene sheeting from ceiling track to floor
3. Secure top edge to metal framing using spring clamps at 18-inch intervals
4. Create 2-inch air gap between sheeting and installed panels
5. Secure bottom edge with water-diverting J-channel
6. Seal all penetrations with waterproof tape

## Critical Notes
- Type X fire-rated assemblies must receive priority protection
- Drywall wet for more than 24 hours typically requires replacement
- Document all affected areas for insurance purposes""",
            "classification": {
                "category": "DRYWALL_PROTECTION_PROTOCOL",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRYWALL_PROTECTION_PROTOCOL",
                "query_type": "document",
                "render_type": "safety_protocol"
            }
        },
        priority=75
    )
)

def get_hardcoded_response(query: str) -> Optional[Dict[str, Any]]:
    """Check if we have a hardcoded response for this query"""
    
    matching_responses = []
    
    for response in HARDCODED_RESPONSES:
        if response.matches(query):
            matching_responses.append(response)
    
    if not matching_responses:
        return None
    
    # Return the highest priority response
    selected = sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0]
    
    return selected.response_data
