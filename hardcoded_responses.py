# hardcoded_responses.py
from typing import Dict, Any, List, Optional
import re

# Define a structure for hardcoded responses
class HardcodedResponse:
    def __init__(
        self,
        query_pattern: str,
        is_regex: bool,
        response_data: Dict[str, Any],
        exact_match: bool = False,
        priority: int = 0  # Higher number = higher priority
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
2. **Call for Help:** Once at a safe distance, call emergency services (911) or your gas company's emergency line
3. **Do Not Create Sparks:** Don't turn on/off any electrical switches, use phones inside, or light matches

## What NOT to Do
* Do not attempt to locate the leak yourself
* Do not turn any electrical equipment on or off
* Do not use your phone until you're safely away from the area

## After Evacuation
* Wait at a safe distance for emergency responders
* Follow all instructions from emergency personnel
* Do not return to the building until professionals declare it safe
""",
            "classification": {
                "category": "EMERGENCY_GAS_LEAK",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "EMERGENCY_GAS_LEAK",
                "query_type": "document",
                "render_type": "safety_protocol",
                "is_gas_leak_emergency": True,
                "emergency_details": {
                    "severity": "high",
                    "requires_immediate_action": True
                }
            }
        },
        priority=100  # Highest priority for emergency
    )
)

# Commercial Drywall Protection Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(gypsum|drywall).+(protect|rain|water|moisture|wet)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Protecting Installed Gypsum Board and Fire-Rated Assemblies During Sudden Rainfall

## Required Equipment
- 8-mil reinforced polyethylene sheeting
- Spring clamps with rubber guards
- Waterproof tape (commercial grade)
- J-channel water diverters
- Commercial-grade water diversion booms
- Red marking tape for fire-rated assemblies
- Digital camera for documentation

## Execution Process
1. Retrieve the emergency rain kit from the designated weather station (marked with yellow stripes)

2. Unroll 8-mil reinforced polyethylene sheeting from ceiling track to floor

3. Secure top edge to metal framing using spring clamps with rubber guards at 18-inch intervals

4. Pull sheeting taut and create an air gap of 2 inches between sheeting and installed panels

5. Secure bottom edge with water-diverting J-channel to direct water away from base track

6. Seal all penetrations (electrical boxes, HVAC openings) with waterproof tape

7. Pay special attention to fire-rated assemblies - mark these with red tape to ensure inspection after rainfall

8. Place commercial-grade water diversion booms in doorways and at perimeter walls

9. Document protected areas with time-stamped photos referencing room numbers per construction drawings

10. Monitor mechanical and electrical penetrations every 30 minutes until rainfall subsides

## Critical Considerations
- Type X fire-rated assemblies must receive priority protection
- Drywall that has been wet for more than 24 hours typically requires replacement
- Document all affected areas for insurance and quality control purposes
- Follow up with moisture meter testing after rainfall event

## Safety Notes
- Do not use metal ladders during active rainfall
- Ensure proper fall protection when working near openings
- Report any electrical concerns immediately to site supervisor
- Evacuate areas where ceiling materials show signs of saturation""",
            "classification": {
                "category": "DRYWALL_PROTECTION_PROTOCOL",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "drywall-protection-guide",
                    "title": "Commercial Gypsum Board Weather Protection",
                    "page": 12,
                    "confidence": 0.96,
                    "excerpt": "Proper protection of installed gypsum board can save significant costs in commercial and industrial projects."
                },
                {
                    "id": "field-manual-protocols",
                    "title": "Emergency Weather Response Protocols",
                    "page": 35,
                    "confidence": 0.94,
                    "excerpt": "Type X fire-rated assemblies must receive priority protection. Drywall that has been wet for more than 24 hours typically requires replacement to maintain warranty coverage."
                }
            ],
            "metadata": {
                "category": "DRYWALL_PROTECTION_PROTOCOL",
                "query_type": "document",
                "special_query_details": {
                    "type": "gypsum_protection",
                    "application": "commercial_construction",
                    "severity": "high_priority"
                },
                "context": {
                    "weather": {
                        "currentTemp": "57°F",
                        "forecast": "Rain expected in 24 hours",
                        "humidity": "72%"
                    }
                }
            }
        },
        priority=75  # High priority
    )
)

# Contract Dispute Analysis Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(who|what|when|why).+(approved|approval|change|HVAC|system).+(Westside|project)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# HVAC System Change Dispute Analysis

## Communication Timeline Analysis

After reviewing all project communications, I've identified conflicting information regarding the HVAC system change on the Westside project.

### Key Findings:
* Original specification was clearly for Airflow 3000 model (James Rodriguez, April 1)
* Supply chain issues were discussed in weekly meeting (Procore meeting minutes, April 5)
* Alternative system (ThermalTech Pro) was proposed to Maria Chen, not James (April 6)
* No formal change order was created or distributed
* Final installation approval was given under time pressure without specific system confirmation

### Contract Implications:
According to Section 12.3 of the contract, substitutions require written approval from the original requester and proper documentation through a change order process.

## Recommended Actions:
1. Create retroactive change order documentation
2. Conduct technical comparison of installed vs. specified system
3. Negotiate cost adjustment based on documented benefits
4. Revise approval protocols to prevent future issues
""",
            "classification": {
                "category": "CONTRACT_DISPUTE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "contract-doc-123",
                    "title": "Westside Project Master Contract",
                    "page": 87,
                    "confidence": 0.95,
                    "excerpt": "Section 12.3: All substitutions of specified materials, equipment, or systems must be approved in writing by the original requester and documented with a formal change order as specified in Section 8.2."
                }
            ],
            "metadata": {
                "category": "CONTRACT_DISPUTE_ANALYSIS",
                "query_type": "document",
                "special_query_details": {
                    "type": "communication_analysis",
                    "project": "Westside",
                    "issue": "hvac_system_change"
                },
                "timeline_events": [
                    {
                        "id": "email-001",
                        "date": "2025-04-01T09:23:15",
                        "sender": "James Rodriguez",
                        "recipient": "HVAC Contractors Inc.",
                        "subject": "HVAC System Requirements for Westside Project",
                        "content": "As per our discussions during the planning phase, please ensure the installed system meets the efficiency ratings specified in the contract. We need the Airflow 3000 model as agreed.",
                        "system": "Outlook",
                        "tags": ["requirements", "specifications"],
                        "relevance": 0.95
                    },
                    {
                        "id": "meeting-002",
                        "date": "2025-04-05T14:30:00",
                        "sender": "Project Meeting",
                        "recipient": "All Team Members",
                        "subject": "Weekly Progress Meeting - Minutes",
                        "content": "During discussion of HVAC installation timeline, Sam mentioned potential supply chain issues with the Airflow 3000. Alternative options were discussed but no decision was made. Action item: James to follow up with HVAC Contractors about alternatives if needed.",
                        "system": "Procore",
                        "tags": ["meeting-minutes", "supply-issues"],
                        "relevance": 0.88
                    },
                    {
                        "id": "email-003",
                        "date": "2025-04-06T11:42:53",
                        "sender": "HVAC Contractors Inc.",
                        "recipient": "Maria Chen",
                        "subject": "RE: Westside Project - HVAC Options",
                        "content": "Following yesterday's meeting, we're having trouble sourcing the Airflow 3000 within your timeline. We recommend the ThermalTech Pro which actually has better efficiency ratings and we can get it installed by the original deadline. It's a bit more expensive but a superior system. Let me know if you want to proceed with this option.",
                        "system": "Outlook",
                        "tags": ["alternative", "recommendation"],
                        "relevance": 0.97
                    },
                    {
                        "id": "chat-004",
                        "date": "2025-04-06T15:17:22",
                        "sender": "Maria Chen",
                        "recipient": "HVAC Contractors Inc.",
                        "subject": "Direct Message",
                        "content": "The ThermalTech Pro sounds like a good alternative. If it's better quality and meets the deadline, it might be worth considering. Let me check with the team about the additional cost.",
                        "system": "Teams",
                        "tags": ["confirmation", "inquiry"],
                        "relevance": 0.92
                    },
                    {
                        "id": "sms-007",
                        "date": "2025-04-12T09:23:11",
                        "sender": "Site Supervisor",
                        "recipient": "Project Manager",
                        "subject": "SMS Message",
                        "content": "HVAC team arrived with ThermalTech Pro units. Different from what I expected but they said it was approved. Should I let them proceed with installation?",
                        "system": "Text Message",
                        "tags": ["urgent", "decision-needed"],
                        "relevance": 0.99
                    },
                    {
                        "id": "chat-008",
                        "date": "2025-04-12T09:31:47",
                        "sender": "Project Manager",
                        "recipient": "Site Supervisor",
                        "subject": "Teams Chat",
                        "content": "If they're ready to go and it meets our specs, let them proceed. We can't afford delays on the HVAC installation as it's on the critical path.",
                        "system": "Teams",
                        "tags": ["approval", "schedule-priority"],
                        "relevance": 0.99
                    }
                ],
                "conflict_points": [
                    {
                        "description": "Original system specification clearly stated as Airflow 3000 in initial communication",
                        "eventIds": ["email-001"],
                        "severity": "medium"
                    },
                    {
                        "description": "Alternative system (ThermalTech Pro) discussed with Maria Chen, not the original requester (James)",
                        "eventIds": ["email-003", "chat-004"],
                        "severity": "high"
                    },
                    {
                        "description": "No formal change order documentation created or distributed to project team",
                        "eventIds": ["email-006", "document-005"],
                        "severity": "high"
                    },
                    {
                        "description": "Ambiguous approval given for installation without specific system confirmation",
                        "eventIds": ["sms-007", "chat-008"],
                        "severity": "medium"
                    }
                ]
            }
        },
        priority=70  # High priority but below emergency responses
    )
)

# Function to check for hardcoded responses
def get_hardcoded_response(query: str) -> Optional[Dict[str, Any]]:
    """Check if we have a hardcoded response for this query"""
    matching_responses = []
    
    for response in HARDCODED_RESPONSES:
        if response.matches(query):
            matching_responses.append(response)
    
    if not matching_responses:
        return None
    
    # Return the highest priority response
    return sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0].response_data
