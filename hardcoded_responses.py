# hardcoded_responses.py - SIMPLIFIED VERSION - MOLD & BUILDERTREND ONLY
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

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

# ============= MOLD-RELATED RESPONSES =============

# 1. Mold Discovery at Sea Cliff Project
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(sea cliff|seacliff).*(demo|demolition).*(mold|mould).*(information|info|documented|reports|data)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Mold Discovery - Sea Cliff Project

## Documentation Search Result: NO

After comprehensive review of all Sea Cliff project documentation, **no prior mold conditions were documented** during pre-demolition assessments or inspection reports.

## Immediate Response Protocol

This appears to be a new discovery requiring immediate safety protocols:

**STOP WORK ORDER INITIATED**
- Halt all demolition work in affected area
- Establish containment protocols  
- Contact environmental specialist for professional assessment
- Document with photos and GPS coordinates

## Required Next Steps
1. Schedule professional mold assessment within 24 hours
2. Update safety protocols for demolition crew
3. Prepare change order for remediation scope
4. Review insurance coverage for environmental discovery

## Documentation Sources Reviewed
- Sea Cliff Project Inspection Report (Page 12): "No visible mold or moisture damage observed"
- STOW Environmental Assessment (Page 7): "Pre-demolition environmental screening complete"

This discovery was not anticipated based on available project documentation.""",
            "classification": {
                "category": "ENVIRONMENTAL_DISCOVERY",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "sea-cliff-inspection-2024",
                    "title": "Sea Cliff Project Inspection Report",
                    "page": 12,
                    "confidence": 0.94,
                    "excerpt": "No visible mold or moisture damage observed during initial structural assessment. All areas appeared dry with no signs of water intrusion."
                },
                {
                    "id": "stow-environmental-assessment", 
                    "title": "STOW Environmental Assessment",
                    "page": 7,
                    "confidence": 0.91,
                    "excerpt": "Pre-demolition environmental screening complete. No hazardous materials identified during visual inspection and air quality testing."
                }
            ],
            "metadata": {
                "category": "MOLD_DISCOVERY_SEA_CLIFF",
                "query_type": "document",
                "render_type": "mold_discovery_renderer",
                "project_name": "Sea Cliff",
                "discovery_type": "environmental_hazard",
                "safety_priority": "high",
                "work_stoppage_required": True
            }
        },
        priority=95,
        exact_match=False
    )
)

# 2. Mold SOP Procedures for Severe Mold Discovery
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(SOP|standard operating procedure|procedures?).*(severe mold|unexpected mold|mold).*(wall|demo|demolition)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Standard Operating Procedures - Severe Mold Discovery

Our Standard Operating Procedures require **immediate work stoppage and specialist intervention** when severe mold is discovered during demolition.

According to SOP Section 4.2 - Environmental Hazards, all demolition work must cease immediately upon discovery of unexpected mold conditions. A certified mold specialist must assess the situation before work can resume.

This protocol was established following lessons learned from the Harbor View project where initial mold discovery led to extensive remediation costs due to delayed specialist involvement.

## Required Immediate Actions:
1. Stop all demolition work in affected area
2. Contact certified mold specialist within 4 hours
3. Establish temporary containment if advised by specialist
4. Document discovery with photos and location mapping
5. Notify project manager and safety coordinator

## Next Steps:
- Await specialist assessment before resuming any work
- Prepare for potential scope and schedule adjustments
- Review insurance coverage for environmental discoveries""",
            "classification": {
                "category": "MOLD_SOP_PROCEDURES",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "safety-field-guide-environmental",
                    "title": "Safety Field Guide - Environmental Protocols",
                    "page": "Section 4.2",
                    "confidence": 0.96,
                    "excerpt": "Immediate work stoppage required upon discovery of unexpected mold conditions during demolition. Certified mold specialist must assess before work resumption."
                },
                {
                    "id": "harbor-view-lessons-learned",
                    "title": "Harbor View Project - Lessons Learned Report", 
                    "page": 15,
                    "confidence": 0.92,
                    "excerpt": "Delayed specialist response resulted in $47,000 additional remediation costs and 12-day schedule impact due to contamination spread."
                }
            ],
            "metadata": {
                "category": "MOLD_SOP_PROCEDURES",
                "query_type": "document",
                "render_type": "mold_sop_renderer",
                "sop_section": "4.2_environmental_hazards",
                "safety_priority": "high",
                "work_stoppage_required": True,
                "specialist_required": True
            }
        },
        priority=96,
        exact_match=False
    )
)

# 3. Mold Remediation Change Order
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(create|generate|can we).*(change order|change-order).*(mold|mould).*(remediation|removal|cleanup).*(100|sqft|square feet)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Mold Remediation Change Order Generated

Change order **CO-2025-0259** has been generated for mold remediation at the Sea Cliff project.

**Scope:** 100 square feet of mold remediation including containment, removal, and surface treatment per industry standards. Work includes proper disposal and air quality testing.

**Cost Impact:** $4,850 additional to contract
**Schedule Impact:** 3-day extension for remediation

## Work Breakdown:
- Containment setup and negative air pressure system
- Mold removal and affected material disposal  
- HEPA filtration and air scrubbing
- Surface treatment and antimicrobial application
- Post-remediation air quality testing
- Documentation and certification

## Regulatory Compliance:
- IICRC S520 Standard procedures followed
- EPA mold remediation guidelines compliance
- Local health department notification requirements
- Proper hazardous waste disposal protocols

Change order ready for client approval and implementation.""",
            "classification": {
                "category": "MOLD_CHANGE_ORDER_GENERATION",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "iicrc-s520-standards",
                    "title": "Mold Remediation Standards - IICRC S520",
                    "page": "Section 3.2",
                    "confidence": 0.94,
                    "excerpt": "Containment and removal procedures for areas up to 100 square feet require negative air pressure, HEPA filtration, and certified disposal methods."
                },
                {
                    "id": "sea-cliff-contract-amendment",
                    "title": "Sea Cliff Project Contract - Amendment Protocol", 
                    "page": 8,
                    "confidence": 0.89,
                    "excerpt": "Environmental discoveries during demolition require immediate change order processing with scope documentation and regulatory compliance verification."
                }
            ],
            "metadata": {
                "category": "MOLD_CHANGE_ORDER_GENERATION",
                "query_type": "document",
                "render_type": "mold_change_order_renderer",
                "project_name": "Sea Cliff",
                "change_order_number": "CO-2025-0259",
                "remediation_area": "100_sqft",
                "cost_impact": 4850,
                "schedule_impact_days": 3
            }
        },
        priority=93,
        exact_match=False
    )
)

# ============= BUILDERTREND RESPONSE =============

# 4. BuilderTrend Client Portal Update
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(thanks|let's|now).*(update|notify).*(client portal|buildertrend|BuilderTrend).*(in the loop|loop|informed)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# BuilderTrend Client Portal Update

BuilderTrend client portal update prepared for Sea Cliff project to notify stakeholders of mold remediation status.

## Client Portal Updates:
- Project timeline updated with 3-day extension
- Change order CO-2025-0259 added to financial summary  
- Daily log entry: Mold discovery and remediation plan
- Photo gallery: Before/during remediation documentation
- Next milestone: Remediation completion (estimated May 23)

## Notifications to Send:
- Email alert to project stakeholders
- SMS notification to key contacts
- Mobile app push notification

## Communication Strategy:
The portal update will provide complete transparency about the environmental discovery, remediation plan, and revised project timeline. All stakeholders will receive immediate notification through their preferred communication channels.

Portal updates maintain project transparency and keep all parties informed of environmental remediation progress and schedule adjustments.""",
            "classification": {
                "category": "BUILDERTREND_PORTAL_UPDATE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "buildertrend-api-documentation",
                    "title": "BuilderTrend API Documentation",
                    "page": "Section 4.1",
                    "confidence": 0.92,
                    "excerpt": "Client portal updates require project timeline synchronization, change order integration, and automated stakeholder notification workflows."
                },
                {
                    "id": "sea-cliff-communication-protocol",
                    "title": "Sea Cliff Project Communication Protocol", 
                    "page": 3,
                    "confidence": 0.89,
                    "excerpt": "Environmental issues require immediate client notification through all available channels including portal updates, email alerts, and direct communication."
                }
            ],
            "metadata": {
                "category": "BUILDERTREND_PORTAL_UPDATE",
                "query_type": "document",
                "render_type": "buildertrend_portal_renderer",
                "project_name": "Sea Cliff",
                "portal_updates": [
                    "timeline_extension",
                    "change_order_integration", 
                    "daily_log_entry",
                    "photo_documentation",
                    "milestone_updates"
                ],
                "notifications": [
                    "email_alerts",
                    "sms_notifications", 
                    "mobile_push"
                ]
            }
        },
        priority=92,
        exact_match=False
    )
)

# Function to check for hardcoded responses
def get_hardcoded_response(query: str) -> Optional[Dict[str, Any]]:
    """Check if we have a hardcoded response for this query"""
    
    matching_responses = []
    
    for response in HARDCODED_RESPONSES:
        if response.matches(query):
            matching_responses.append(response)
            print(f"🎯 MATCH FOUND: {response.response_data['metadata']['category']} (Priority: {response.priority})")
    
    if not matching_responses:
        print(f"❌ No hardcoded response found for query: {query}")
        return None
    
    # Return the highest priority response
    selected = sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0]
    print(f"✅ SELECTED RESPONSE: {selected.response_data['metadata']['category']}")
    
    return selected.response_data
