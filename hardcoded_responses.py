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

# Climate Data Visualization
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern="show me climate data variance",
        is_regex=False,
        response_data={
            "status": "success",
            "answer": """# Climate Impact Analysis

## Temperature Variations

Global measurements show significant variations across regions. Northern areas experience more pronounced warming, with some regions showing temperature increases of up to 2.1°C above pre-industrial levels. Southern hemisphere changes are more moderate, averaging 0.8-1.2°C increases.

## Precipitation Patterns

Precipitation patterns show high regional variability:
- Increased rainfall intensity in equatorial regions
- Prolonged drought conditions in mid-latitudes
- More frequent extreme precipitation events globally

## Source Conflicts

Different climate models project varying outcomes for certain regions, particularly in predicting monsoon pattern shifts across Southeast Asia and agricultural impacts in transitional climate zones.
""",
            "classification": {
                "category": "CLIMATE_DATA",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "doc-123",
                    "title": "Climate Report 2024",
                    "page": 42,
                    "confidence": 0.95,
                    "excerpt": "Temperature variations across regions indicate..."
                }
            ],
            "metadata": {
                "category": "CUSTOM_RENDERER_REQUIRED",
                "query_type": "special_visualization",
                "render_type": "climate_data_visual",
                "custom_renderer_data": {
                    "chart_type": "temperature_variance",
                    "regions": ["North America", "Europe", "Asia"],
                    "time_period": "2020-2024"
                }
            }
        },
        exact_match=True,
        priority=50
    )
)

# Construction Aggregates Alternative
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(alternative|replacement|substitute|other).+(construction aggregate|aggregates|crushed stone|stone aggregate)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Alternative Construction Aggregates

## Recycled Alternatives
- **Recycled Concrete Aggregate (RCA):** Processed from demolished concrete structures, providing similar properties to natural aggregates
- **Crushed Brick:** Derived from demolition waste, suitable for non-structural applications
- **Glass Aggregate:** Processed from recycled glass, effective for decorative concrete and special applications

## Natural Alternatives
- **Gravel:** Natural river or pit-sourced aggregate with rounded edges
- **Sand:** Fine aggregate for mortar and concrete mixes
- **Slag:** Byproduct of steel production, excellent for road construction

## Environmental Benefits
Using alternative aggregates reduces quarrying impacts, decreases landfill waste, and lowers the carbon footprint of construction projects.

## Application Considerations
Always verify compliance with local building codes and structural requirements before substituting traditional aggregates.""",
            "classification": {
                "category": "MATERIALS",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "CONSTRUCTION_AGGREGATES_SPECIAL",
                "query_type": "document",
                "is_special_aggregate_query": True,
                "special_query_details": {
                    "type": "alternative_aggregates",
                    "is_exact_match": False
                }
            }
        },
        exact_match=False,
        priority=60
    )
)

# StormShield Installation Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(stormshield|storm shield).+(install|requirements|facade|west|commercial|rain)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# StormShield Installation Standards

## Key Requirements for West-Facing Commercial Facades

### Temperature and Environmental Requirements
* Application temperature must be between 45-85°F with surface moisture reading below 12%
* Do not apply if rain is expected within 48 hours

### Proper Installation Sequence
* Primary membrane must cure for 24 hours before secondary layer application
* West-facing installations require double-layer application at seams with 6" overlap (increased from 4" in previous standard)
* Use BlueBond-X adhesive formula for any installation within 15 miles of saltwater

### Quality Assurance
* Schedule inspection after installation
* Validate installation meets warranty requirements

### Common Installation Errors to Avoid
* Insufficient overlap at seams (most common failure point)
* Using standard adhesive in coastal applications
* Applying second layer before primary membrane has fully cured
* Improper trowel technique leading to uneven application""",
            "classification": {
                "category": "PROPRIETARY_INSTALLATION",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "storm-shield-manual-v2025",
                    "title": "StormShield Installation Guide (March 2025)",
                    "page": 42,
                    "confidence": 0.95,
                    "excerpt": "West-facing installations require double-layer application at seams with 6\" overlap (increased from 4\" in previous standard). Application temperature must be between 45-85°F with surface moisture reading below 12%. Primary membrane must cure for 24 hours before secondary layer application."
                },
                {
                    "id": "qa-memo-218",
                    "title": "Quality Assurance Memo #218",
                    "page": 3,
                    "confidence": 0.92,
                    "excerpt": "BlueBond-X formula is now required for any installation within 15 miles of saltwater. This update supersedes previous adhesive specifications for coastal applications."
                }
            ],
            "metadata": {
                "category": "PROPRIETARY_INSTALLATION",
                "query_type": "document",
                "special_query_details": {
                    "type": "installation_guide",
                    "product": "StormShield",
                    "installation_type": "west_facing_facade"
                },
                "context": {
                    "weather": {
                        "currentTemp": "52°F",
                        "forecast": "Rain expected in 48 hours",
                        "humidity": "68%"
                    }
                }
            }
        },
        priority=80  # High priority but below emergency responses
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

# Roll Forming Machine Troubleshooting Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(scotpanel|roll form|roll former|forming machine|roll forming|formed edge|material thickness).*(troubleshoot|issue|problem|error|inconsistent|thickness|edge|not working|help|fix)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Field Service Diagnostic Guide: Scotpanel Series
### Customer Issue: Edge Thickness Variation

## Initial Assessment 
Before opening any panels, verify with the customer:
- When the issue first appeared (after material change, maintenance, etc.)
- Whether the issue is consistent or intermittent
- If any recent maintenance has been performed on the machine
- If they have documentation of current material specifications

## Service Checklist
Perform standard safety lockout/tagout procedure before any mechanical inspection. Document all findings in the service report for warranty coverage.

## Common Root Causes
Based on recent service data from similar installations, probable causes include:
- Entry roller misalignment (42% of cases)
- Incorrect pressure settings for material gauge (31%)
- Material property inconsistencies (18%)
- Calibration drift on hydraulic sensors (9%)

## Required Service Tools
- Alignment laser kit (PN: ALK-SP100)
- Calibrated feeler gauge set
- Digital torque wrench (40-80Nm)
- Pressure test kit with digital gauge
- Service tablet with latest firmware

Follow each diagnostic path carefully. If multiple issues are found, document all findings for the customer before making adjustments.""",
            "classification": {
                "category": "ROLL_FORMING_TROUBLESHOOTING",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "rf-manual-2025",
                    "title": "Scotpanel Roll Forming Machine Technical Manual",
                    "page": 87,
                    "confidence": 0.95,
                    "excerpt": "When troubleshooting inconsistent material thickness at edges, always begin by checking entry roller alignment. Misalignment of 0.2mm or greater can cause significant thickness variation at formed edges."
                },
                {
                    "id": "tech-bulletin-42",
                    "title": "Technical Bulletin: Edge Formation Quality",
                    "page": 3,
                    "confidence": 0.92,
                    "excerpt": "For models produced between 2023-2024, verify the pressure sensors on forming stations 2-4 have been calibrated according to procedure TB-SP100-42. Uncalibrated sensors are responsible for 27% of reported edge quality issues."
                },
                {
                    "id": "service-bulletin-2025-03",
                    "title": "Field Service Bulletin: Edge Quality Issues",
                    "page": 1,
                    "confidence": 0.98,
                    "excerpt": "All field technicians must document pre-adjustment measurements for warranty claims. Use service tablet to photograph alignment readings before making any adjustments."
                }
            ],
            "metadata": {
                "category": "ROLL_FORMING_TROUBLESHOOTING",
                "query_type": "troubleshooting",
                "render_type": "roll_forming_troubleshooter",
                "machine_model": "Scotpanel",
                "issue_description": "Inconsistent material thickness at edges - Customer experiencing quality rejections",
                "service_context": {
                    "customer_machine": {
                        "model": "Scotpanel",
                        "serial": "SP-2024-1285",
                        "installation_date": "2024-01-15",
                        "warranty_status": "Active",
                        "last_service": "2025-03-12"
                    },
                    "customer_info": {
                        "company": "Precision Metals Inc.",
                        "location": "Cincinnati",
                        "issue_priority": "High - Production Impacted"
                    }
                }
            }
        },
        priority=75,  # High priority
        exact_match=False
    )
)

# Roll Forming Machine Assembly Guide - Scotpanel
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        # Expanded pattern to catch more variations of questions about assembly
        query_pattern=r"(scotpanel|roll form|roll former|forming machine|panel former|metal former).*(assembly|build|construct|assemble|put together|installation|setup|guide|manual|instructions|procedure|steps|how to|align)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Scotpanel Roll Forming Machine Assembly Guide

## Pre-Assembly Requirements
Before beginning assembly, verify all parts have been received and match the bill of materials. Check for any shipping damage. Ensure assembly area meets space requirements (minimum 10m x 6m clear floor space).

## Main Assembly Sequences
The Scotpanel roll forming machine assembly is divided into these primary subassemblies:
- Entry guide and shaft assembly
- Drive system installation
- Forming station setup (varies by profile configuration)
- Exit and cutoff mechanism assembly
- Control system wiring and setup

## Critical Assembly Points
- All shaft alignments must be within 0.05mm tolerance
- Bearing installation requires induction heating (80-120°C)
- Torque specifications must be followed exactly
- Roller gaps must be set according to profile specification drawings
- Lubrication ports must be properly aligned before final assembly

## Quality Control
Each subassembly requires quality verification before proceeding to the next step. Document all measurements, alignments, and torque values in the work order paperwork.""",
            "classification": {
                "category": "ROLL_FORMER_ASSEMBLY",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "rf-manual-2025",
                    "title": "Scotpanel Roll Forming Machine Assembly Manual",
                    "page": 12,
                    "confidence": 0.95,
                    "excerpt": "The assembly process requires precision alignment of all components. Tolerances tighter than 0.05mm must be maintained for proper machine operation. Always follow torque specifications to prevent premature component failure."
                },
                {
                    "id": "rf-training-guide",
                    "title": "Roll Former Assembly Training Guidelines",
                    "page": 5,
                    "confidence": 0.92,
                    "excerpt": "Bearing installation is critical to machine performance. Use the induction heater to expand bearings for installation. Never exceed 120°C to avoid damaging bearing seals. Always measure bearing temperature before installation."
                },
                {
                    "id": "rf-qc-procedures",
                    "title": "Quality Control Procedures for Roll Former Assembly",
                    "page": 8,
                    "confidence": 0.94,
                    "excerpt": "Document all critical measurements during assembly, including shaft alignments, bearing clearances, and roller gaps. These measurements must be recorded on the quality control checklist and included with the machine documentation package."
                }
            ],
            "metadata": {
                "category": "ROLL_FORMER_ASSEMBLY",
                "query_type": "assembly_guide",
                "render_type": "roll_former_assembly_navigator",
                "machine_model": "Scotpanel",
                "work_order": "WO-2025-1854",
                "customer_name": "Precision Metals Inc.",
                "assembly_station": "Station #4 - Drive Assembly",
                "assembly_context": {
                    "machine_details": {
                        "model": "Scotpanel",
                        "type": "Commercial Grade Roll Former",
                        "production_date": "2025-04-15"
                    }
                }
            }
        },
        priority=75,  # High priority
        exact_match=False  # Allow for partial matches to increase flexibility
    )
)

# Specific entry for bearing alignment questions - added for more detailed matching
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(align|alignment|installing|setup|procedure|how|what).*(bearing|bearings|shaft|entry shaft|drive).*(scotpanel|roll form|roll former|panel form)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Entry Shaft Bearing Alignment Procedure for Scotpanel

This comprehensive guide provides step-by-step instructions for aligning entry shaft bearings on Scotpanel roll formers, ensuring proper tolerance and optimal performance.

## Overview
Proper bearing alignment is critical to achieving consistent material forming. This procedure includes preparation, installation, verification, and final testing to ensure your roll former operates within specifications.

## Key Points
- Complete procedure includes 10 detailed steps
- Critical measurements must be documented
- Special tools required from calibrated tools cabinet
- Recent updates to bearing specifications (March 2025)""",
            "classification": {
                "category": "ROLL_FORMER_ASSEMBLY",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "16638744-42de-4393-bdf6-60e7f68762df",
                    "title": "Scotpanel Bearing Alignment Procedure",
                    "page": 12,
                    "confidence": 0.98,
                    "excerpt": "Entry shaft bearing alignment is critical to machine performance. Follow this procedure carefully to ensure proper alignment within specified tolerances."
                },
                {
                    "id": "1f7e29e3-a8ee-40a4-b481-4407dc38a649",
                    "title": "Scotpanel Troubleshooting Guide",
                    "page": 25,
                    "confidence": 0.85,
                    "excerpt": "Many material forming issues can be traced to improper bearing alignment. Verify alignment per procedure in maintenance manual before addressing other potential causes."
                }
            ],
            "metadata": {
                "category": "ROLL_FORMER_ASSEMBLY",
                "query_type": "assembly_guide",
                "render_type": "roll_former_assembly_navigator",
                "machine_model": "Scotpanel",
                "work_order": "WO-2025-1854",
                "customer_name": "Precision Metals Inc.",
                "assembly_station": "Station #4 - Drive Assembly",
                "assembly_context": {
                    "machine_details": {
                        "model": "Scotpanel",
                        "type": "Commercial Grade Roll Former",
                        "production_date": "2025-04-15"
                    }
                }
            }
        },
        priority=85,  # Higher priority than general assembly questions
        exact_match=False  # Allow for partial matches to increase flexibility
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
