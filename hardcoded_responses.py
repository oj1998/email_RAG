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
        exact_match=True
    )
)

# Replace your existing Construction Aggregates Alternative with this more flexible version

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
        priority=60  # Adjust priority as needed
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
                        "id": "document-005",
                        "date": "2025-04-07T08:45:00",
                        "sender": "System",
                        "recipient": "All Users",
                        "subject": "Westside Project - Daily Report",
                        "content": "HVAC work scheduled to begin next week. All materials should be on site by Thursday according to contractor confirmation.",
                        "system": "Procore",
                        "tags": ["daily-report", "schedule"],
                        "relevance": 0.75
                    },
                    {
                        "id": "email-006",
                        "date": "2025-04-09T16:03:42",
                        "sender": "Maria Chen",
                        "recipient": "Project Team",
                        "subject": "Updates on Various Vendors",
                        "content": "Several updates from today's vendor calls: ... [Multiple updates about different aspects] ... HVAC team mentioned they're proceeding with preparations and will be ready to start Monday.",
                        "system": "Outlook",
                        "tags": ["update", "multiple-topics"],
                        "relevance": 0.68
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

# Add this to your HARDCODED_RESPONSES list in hardcoded_responses.py

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

HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(safety|precaution|procedure|protocol|how|what|guide|help|advice|best|practice|method|instruction|way|emergency|standard|should|handle|manage|do|work|repair).*(splic|fiber|optic|cable|line|network|communication).*(wet|rain|storm|aerial|emergency|damage|outdoor|weather|humid|moisture|water|damp)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# EMERGENCY: Fiber Optic Splicing - Wet Weather Protocol

## Immediate Safety Requirements
1. **De-energize all equipment:** Ensure all power sources are disconnected before beginning work
2. **Verify atmospheric conditions:** Do not work during active lightning - wait minimum 30 minutes after last strike
3. **Personal protective equipment:** Waterproof gloves, insulated tools, arc-rated rain gear required

## Wet Weather Splicing Procedure
1. **Establish dry work area:** Deploy portable canopy or splice enclosure tent over work zone
2. **Cable preparation:** Thoroughly dry cable ends using lint-free wipes and isopropyl alcohol
3. **Moisture prevention:** Apply temporary water-blocking gel to exposed fiber buffer tubes
4. **Fusion splicing parameters:** Increase arc power by 10% and fusion time by 2 seconds for humid conditions
5. **Splice protection:** Use heat-shrink sleeves rated for outdoor use with adhesive lining
6. **Closure sealing:** Apply extra butyl tape around all entry points in splice enclosure

## Aerial Line Specific Precautions
* Use bucket truck with proper grounding straps
* Maintain 10-foot clearance from energized power lines
* Install temporary cable guards if working near electrical
* Double-check strand bonds and grounding before touching cable

## Critical Quality Checks
- Test each splice with OTDR before closing enclosure
- Document moisture readings inside closure (must be <30% RH)
- Verify proper drip loops at all aerial entry points
- Photograph completed work for storm damage documentation

## Emergency Contact Protocol
- Notify NOC immediately upon arrival at damage site
- Update outage management system every 30 minutes
- Contact local power company if poles are compromised
- Document all safety hazards in field report""",
            "classification": {
                "category": "FIBER_OPTIC_EMERGENCY",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "ec5caefb-f00b-4c6e-8250-4d0050792ba4",
                    "title": "Fiber Optic Safety Manual - Wet Weather Operations",
                    "page": 47,
                    "confidence": 0.98,
                    "excerpt": "When splicing fiber optic cables in wet conditions, arc parameters must be adjusted to compensate for atmospheric moisture. Increase arc power by 10% and fusion time by 2 seconds to ensure proper glass flow."
                },
                {
                    "id": "storm-response-procedures",
                    "title": "Telecommunications Storm Response Procedures",
                    "page": 12,
                    "confidence": 0.95,
                    "excerpt": "De-energize all equipment before beginning emergency repairs. Wait minimum 30 minutes after last lightning strike before ascending aerial structures. Document all damage with photographs for insurance and FEMA claims."
                }
            ],
            "metadata": {
                "category": "FIBER_OPTIC_EMERGENCY",
                "query_type": "document",
                "render_type": "fiber_optic_safety",
                "is_emergency_response": True,
                "emergency_type": "storm_damage",
                "context": {
                    "weather": {
                        "conditions": "Post-storm, wet conditions",
                        "wind_speed": "15-20 mph",
                        "precipitation": "Light rain"
                    },
                    "outage_impact": {
                        "customers_affected": "1,200",
                        "critical_facilities": "Rural hospital, 2 schools"
                    }
                }
            }
        },
        priority=95,  # High priority for emergency responses
        exact_match=False
    )
)

# Directional Drilling Specifications Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(specification|spec|standard|requirement|procedure|how|guide|what|install|method|process|technique|approach|system|way|work|do|perform|execute|practice|protocol).*(directional|horizontal|HDD|drill|bor|dig|tunnel|trench|cross|under).*(highway|road|utilities|utility|underground|street|path|infrastructure|pavement|asphalt|crossing|thoroughfare|passage|conduit|pipe|cable|wire|line|duct)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Directional Drilling Under Highways - Technical Specifications

## Pre-Drilling Requirements
1. **Utility Locates:** Obtain current utility locates (valid within 48 hours)
2. **Soil Analysis:** Conduct geotechnical survey to determine soil composition  
3. **Permits:** Secure DOT permits and right-of-way authorizations
4. **Traffic Control:** Implement approved traffic management plan

## Bore Path Specifications
* **Minimum Depth:** 48 inches below roadway surface (60 inches for state highways)
* **Entry Angle:** 8-12 degrees (never exceed 20 degrees)
* **Exit Angle:** 5-10 degrees for smooth cable/conduit pullback
* **Bore Diameter:** Minimum 1.5x bundle diameter (2x for rocky conditions)
* **Bend Radius:** Minimum 1200 feet for fiber optic installations

## Critical Clearances from Utilities
- Gas Lines: 24-inch minimum horizontal/vertical separation
- Water Mains: 12-inch minimum separation
- Electric Lines: 24-inch from primary, 12-inch from secondary
- Sewer Lines: 24-inch minimum (48-inch preferred)
- Storm Drains: 12-inch minimum separation

## Drilling Fluid Management
- Use only approved biodegradable drilling fluids
- Maintain 30-50 psi fluid pressure during pilot bore
- Monitor for inadvertent returns (frac-outs)
- Contain and properly dispose of all drilling fluids

## Safety Requirements
- Call 811 utility notification 72 hours before drilling
- Maintain pothole excavations every 50 feet in congested areas
- Use ground penetrating radar when utilities are within 36 inches
- Install tracer wire with all non-metallic conduits

## Quality Control Checkpoints
- Verify bore path alignment every 20 feet
- Document actual vs planned path deviation
- Conduct mandrel test before cable installation
- Pressure test conduit at 15 psi for 15 minutes

## Highway Crossing Specific Requirements
When crossing under highways, additional specifications apply:
- Submit engineered drawings to DOT for approval
- Maintain minimum 10-foot offset from bridge abutments
- Install warning tape 12 inches above conduit
- Place permanent markers at entry/exit points""",
            "classification": {
                "category": "DIRECTIONAL_DRILLING",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "1b2e9216-1334-4fa7-92d7-dec270712958",  # REPLACE WITH ACTUAL ID
                    "title": "Directional Drilling Technical Manual",
                    "page": 23,
                    "confidence": 0.97,
                    "excerpt": "When crossing under highways, minimum depth requirements increase to 48 inches below roadway surface, with 60 inches required for state highways. Entry angles must be maintained between 8-12 degrees for optimal bore path stability."
                },
                {
                    "id": "YOUR-DOT-REQUIREMENTS-DOC-ID",  # REPLACE WITH ACTUAL ID
                    "title": "DOT Highway Crossing Requirements",
                    "page": 15,
                    "confidence": 0.94,
                    "excerpt": "All underground utility crossings beneath state highways require engineered drawings and DOT approval. Maintain minimum 10-foot offset from bridge abutments and install permanent markers at bore entry/exit points."
                }
            ],
            "metadata": {
                "category": "DIRECTIONAL_DRILLING",
                "query_type": "document",
                "render_type": "directional_drilling_specs",
                "project_context": {
                    "crossing_type": "highway",
                    "soil_conditions": "mixed clay/rock",
                    "existing_utilities": ["gas", "water", "electric", "telecom"],
                    "permit_status": "pending_approval"
                },
                "safety_alerts": [
                    "High-pressure gas line crosses at station 2+50",
                    "Fiber optic trunk line at 36 inches depth",
                    "Rock layer expected at 8-10 feet"
                ]
            }
        },
        priority=90,  # High priority
        exact_match=False
    )
)

HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(consideration|requirement|challenge|procedure|guideline|what|how|install|deploy|run|hang|string|extend|place|position|construct|build|plan|design|strategy|method|approach|technique|issue|problem|concern|obstacle|difficulty|best|practice).*(aerial|fiber|optic|cable|line|wire|network|communication|overhead|pole|utility|outside|external|above|suspension).*(rural|remote|mountain|difficult|challenging|rough|rugged|uneven|harsh|wild|isolated|hard|tough|country|wilderness|backwoods|hill|elevated|terrain|landscape|topography|area|region|location|zone|environment)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Aerial Fiber Installation in Mountainous Rural Areas

## Terrain-Specific Considerations
1. **Access Road Limitations:** Many rural mountain areas lack proper roads - use tracked vehicles, ATVs, or helicopter support for equipment transport
2. **Slope Stability:** Assess soil conditions and erosion potential before setting poles on steep grades
3. **Weather Windows:** Mountain weather changes rapidly - monitor forecasts and plan for sudden storms
4. **Wildlife Corridors:** Avoid disrupting migration paths and nesting areas during installation

## Equipment Requirements for Remote Areas
* **Specialized Vehicles:** Tracked bucket trucks, all-terrain digger derricks
* **Portable Power:** Generator sets (minimum 25kW) for splicing equipment
* **Communication Equipment:** Satellite phones for areas without cell coverage
* **Emergency Supplies:** First aid, survival gear, extra food/water for crew

## Installation Best Practices
1. **Pole Setting on Slopes:**
   - Increase embedment depth by 10% for every 10 degrees of slope
   - Use crushed rock backfill for drainage on uphill side
   - Install guy wires at 45-degree angles perpendicular to slope
   - Consider rock anchors in lieu of traditional anchors

2. **Span Lengths in Mountainous Terrain:**
   - Reduce standard spans by 20% in areas with high wind exposure
   - Maximum 200-foot spans across valleys (vs. 300-foot standard)
   - Use heavy-duty suspension clamps at all angle points
   - Install vibration dampers on spans exceeding 150 feet

3. **Clearance Requirements:**
   - Maintain 25-foot minimum clearance over mountain roads (vs. 18-foot standard)
   - Allow for 5-foot snow accumulation in clearance calculations
   - Increase tree trimming zone to 15 feet each side in fire-prone areas
   - Consider wildlife flight paths when determining cable height

## Weather-Related Considerations
* **Winter Installation Challenges:**
  - Fiber becomes brittle below -20°C; warm cable before pulling
  - Ice loading can triple cable weight; use high-strength messenger wire
  - Snow accumulation affects access roads; maintain alternate routes
  - Shortened daylight hours require efficient crew scheduling

* **Wind Factors:**
  - Mountain ridges experience 2-3x valley wind speeds
  - Install wind dampeners on all spans exposed to prevailing winds
  - Use armor-wrapped cable in high-wind zones
  - Consider underground installation for extremely exposed areas

## Safety Protocols for Remote Work
1. **Crew Requirements:**
   - Minimum 3-person crews for mountain work (vs. 2-person standard)
   - Wilderness first aid certification for at least one crew member
   - Daily check-in procedures with base operations
   - Emergency evacuation plan for each work site

2. **Equipment Safety:**
   - Inspect climbing gear daily in harsh conditions
   - Use fall restraint (not just fall arrest) on steep slopes
   - Carry emergency descent devices for aerial rescue
   - Maintain tire chains and recovery equipment on all vehicles

## Material Logistics
* **Supply Chain Management:**
  - Pre-position materials at staging areas before winter
  - Use weatherproof storage containers for sensitive equipment
  - Maintain 20% extra materials for weather-related delays
  - Coordinate with local suppliers for emergency needs

* **Cable Management:**
  - Store cable reels in climate-controlled environment when possible
  - Use smaller reels (2,500 feet vs. 5,000 feet) for easier transport
  - Allow cable to acclimate to ambient temperature before installation
  - Document all cable serial numbers and test results""",
            "classification": {
                "category": "RURAL_INSTALLATION",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "dce73f2d-a162-4575-bea4-248d3f76ef44",  # REPLACE WITH ACTUAL ID
                    "title": "Rural Fiber Deployment Guide - Mountainous Regions",
                    "page": 67,
                    "confidence": 0.96,
                    "excerpt": "Mountain installations require specialized equipment and procedures. Reduce standard span lengths by 20% in high-wind areas, with maximum 200-foot spans across valleys. Increase pole embedment depth by 10% for every 10 degrees of slope to ensure stability."
                },
                {
                    "id": "YOUR-TERRAIN-CHALLENGES-DOC-ID",  # REPLACE WITH ACTUAL ID
                    "title": "Challenging Terrain Installation Manual",
                    "page": 42,
                    "confidence": 0.93,
                    "excerpt": "Wildlife corridor considerations are critical in rural installations. Avoid disrupting migration paths and nesting areas. Maintain 15-foot clearance zones in fire-prone areas and consider underground installation for extremely exposed mountain ridge locations."
                }
            ],
            "metadata": {
                "category": "RURAL_INSTALLATION",
                "query_type": "document",
                "render_type": "rural_installation_guide",
                "terrain_context": {
                    "terrain_type": "mountainous",
                    "elevation_range": "2,000-8,000 feet",
                    "access_difficulty": "high",
                    "weather_severity": "extreme"
                },
                "project_insights": {
                    "similar_projects": [
                        {
                            "name": "Blue Ridge Fiber Extension",
                            "completion": "2024-11",
                            "challenges": ["45-degree slopes", "wildlife protection zones", "winter storm delays"],
                            "lessons_learned": ["Pre-position materials by October", "Use helicopter for ridge-top poles", "Partner with local contractors"]
                        },
                        {
                            "name": "Rocky Mountain Rural Connect",
                            "completion": "2024-08",
                            "challenges": ["Rock drilling required", "Limited access roads", "High altitude work"],
                            "lessons_learned": ["Track vehicles essential", "Account for altitude sickness", "Extra generators needed"]
                        }
                    ]
                }
            }
        },
        priority=85,  # High priority
        exact_match=False
    )
)

# Hospital Sheet Vinyl Installation Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(how|what|procedure|install).*(sheet vinyl|vinyl).*(hospital|operating room|OR|medical|healthcare)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Hospital Operating Room - Sheet Vinyl Installation Protocol

## Critical Compliance Requirements
* Operating room must be scheduled out of service for 72 hours minimum
* HEPA filtration system must remain operational throughout installation
* Infection Control must approve all materials before use
* Continuous monitoring of temperature and humidity required

## Environmental Controls
- **Temperature:** 68-72°F (20-22°C)
- **Humidity:** 45-55% RH
- **Air Changes:** 20+ per hour
- **Pressure:** Positive 0.03" w.g.

## Installation Procedure

### Step 1: Pre-Installation Substrate Testing
Conduct moisture testing per ASTM F1869 (calcium chloride) and ASTM F2170 (RH probe). Maximum acceptable readings: 3 lbs/1000 sq ft/24hr or 80% RH.

### Step 2: Adhesive Application
Apply healthcare-grade, low-VOC adhesive using 1/16" x 1/16" x 1/16" square-notch trowel. Allow proper flash time based on temperature and humidity conditions.

**Critical:** Adhesive must be antimicrobial and meet GREENGUARD Gold certification

### Step 3: Flash Cove Installation
Install integral flash coving with minimum 6" height. Use pre-formed inside corners. All transitions must be heat-welded to create monolithic surface.

### Step 4: Heat Welding Seams
Groove seams to 2/3 material thickness. Heat weld at 650°F using color-matched welding rod. Allow to cool completely before skiving flush with surface.

### Step 5: Final Inspection & Documentation
Complete infection control checklist. Document all seam locations with photos. Perform final cleaning with approved hospital-grade disinfectant.

**Required:** Submit installation certificate to Facilities and Infection Control""",
            "classification": {
                "category": "HEALTHCARE_INSTALLATION",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "e051d95c-016c-4b1a-9011-7cec39966804",
                    "title": "IPC Healthcare Flooring Standards 2024",
                    "page": 47,
                    "confidence": 0.98,
                    "excerpt": "Operating room flooring must meet strict infection control requirements. Sheet vinyl installation requires continuous cove base with heat-welded seams. All seams must be sealed using manufacturer-approved welding rod at 650°F. Minimum 6-inch flash coving at all wall junctions is mandatory."
                },
                {
                    "id": "vinyl-manufacturer-specs",
                    "title": "Healthcare Grade Vinyl Installation Manual",
                    "page": 112,
                    "confidence": 0.95,
                    "excerpt": "Temperature and humidity control critical: Maintain 65-75°F and 40-60% RH for 48 hours before, during, and after installation. Substrate moisture must not exceed 3 lbs/1000 sq ft per 24 hours (ASTM F1869) or 80% RH (ASTM F2170)."
                },
                {
                    "id": "joint-commission-standards",
                    "title": "Joint Commission Environment of Care Standards",
                    "page": 23,
                    "confidence": 0.92,
                    "excerpt": "Flooring in procedural areas must be seamless where possible. Where seams are necessary, they must be chemically welded to prevent moisture infiltration and microbial growth. Documentation of installation procedures required for compliance."
                }
            ],
            "metadata": {
                "category": "HEALTHCARE_INSTALLATION",
                "query_type": "document",
                "render_type": "healthcare_installation_guide",
                "is_healthcare_specific": True,
                "compliance_requirements": {
                    "standards": ["IPC", "Joint Commission", "GREENGUARD"],
                    "documentation_required": True,
                    "infection_control_approval": True
                },
                "environmental_requirements": {
                    "temperature": {
                        "min": 68,
                        "max": 72,
                        "unit": "fahrenheit"
                    },
                    "humidity": {
                        "min": 45,
                        "max": 55,
                        "unit": "percent"
                    },
                    "air_changes": 20,
                    "pressure": "positive"
                }
            }
        },
        priority=85,
        exact_match=False
    )
)


# Fixed Utilities Location Insights Hardcoded Response

# Location-Based Utilities Analysis Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        # Primary high-priority specific pattern
        query_pattern=r"(material|utility|conduit|underground).+(requirement|order|installation|consideration).+(Jacksonville|Clay Hill|Florida)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Location-Based Utility Construction Insights

## Location Analysis for Jacksonville

### Site Conditions Overview
- **Soil Composition:** Clay-heavy with moderate rock content
- **Water Table:** 4-6 feet below grade, seasonal variation of 2 feet
- **Terrain:** Gently sloping with 3-5% grade changes
- **Underground Infrastructure Density:** Moderate (8 existing utilities per 100 linear feet)
- **Ground Frost Penetration:** 16-22 inches during winter months

### Production Metrics Impact
- **Trenching Rate:** 240-280 ft/day (15% below company average)
- **Conduit Installation:** 320-350 ft/day (standard efficiency)
- **Backfill Operations:** 275-300 ft/day (10% below average due to soil composition)
- **Project Timeline Adjustment:** Additional 4 days required for this location

### Material Requirements
- **Bedding Material:** Requires 30% additional sand bedding due to clay soil conditions
- **Backfill Composition:** Need specialized mix ratio for proper compaction
- **Conduit Protection:** Additional protective measures needed due to rock content
- **Recommended Supplier:** Northeast Materials (closest to site, special contractor rates)

### Equipment Recommendations
- **Primary Excavator:** Track-mounted mini-excavator recommended over standard backhoe
- **Shoring Requirements:** Hydraulic shoring system for depths over 5 feet
- **Dewatering Equipment:** Submersible pump on standby for seasonal water table fluctuations
- **Specialized Tools:** Rock saw attachment recommended for trenching

### Regulatory Considerations
- **Local Permit Lead Time:** 20 business days (vs. standard 10 days)
- **Environmental Requirements:** Wetlands proximity documentation required
- **Utility Coordination:** Jacksonville Electric Authority coordination needed
- **Traffic Control Plan:** Modified requirements due to residential zoning

### Actionable Insight Summary
1. **Schedule Impact:** Begin permit process immediately to avoid critical path delays
2. **Material Procurement:** Order additional 12 cubic yards of bedding sand
3. **Equipment Scheduling:** Reserve track equipment 14 days before project start
4. **Crew Configuration:** Add one additional laborer for proper soil handling

### Automation Opportunities
The following processes can be automated for this project:
1. **Materials Order Preparation** - Additional bedding and specialized backfill calculation and ordering
2. **Permit Application Workflow** - Streamlined application for 20-day lead time permits
3. **Equipment Scheduling** - Reservation of track and dewatering equipment
4. **Environmental Compliance** - Generation of wetlands proximity documentation""",
            "classification": {
                "category": "LOCATION_UTILITIES_INSIGHTS",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "utility-location-guide-2025",
                    "title": "Regional Utility Installation Standards",
                    "page": 42,
                    "confidence": 0.96,
                    "excerpt": "Clay-heavy soils in the northeastern Florida region require 25-30% additional bedding material compared to sandy soils. Production rates for trenching operations should be adjusted by 15-20% to account for the increased difficulty."
                },
                {
                    "id": "jax-utilities-handbook",
                    "title": "Jacksonville Utilities Installation Handbook",
                    "page": 18,
                    "confidence": 0.94,
                    "excerpt": "Permit applications for underground utility work in Clay Hill and surrounding areas require additional environmental documentation due to wetlands proximity. Allow 15-20 business days for permit processing."
                }
            ],
            "metadata": {
                "category": "LOCATION_UTILITIES_INSIGHTS",
                "query_type": "location_analysis",
                "render_type": "utilities_location_insight",
                "location_context": {
                    "location_name": "Jacksonville, Florida",
                    "coordinates": {
                        "lat": 30.1846,
                        "lng": -81.8543
                    },
                    "region_type": "suburban",
                    "soil_type": "clay-heavy",
                    "local_jurisdiction": "Jacksonville, FL"
                },
                "production_metrics": {
                    "trenching_rate": {
                        "expected": 260,
                        "unit": "ft/day",
                        "company_average": 305,
                        "variance": -15
                    },
                    "conduit_installation": {
                        "expected": 335,
                        "unit": "ft/day",
                        "company_average": 340,
                        "variance": -1.5
                    },
                    "backfill_operations": {
                        "expected": 287,
                        "unit": "ft/day",
                        "company_average": 320,
                        "variance": -10
                    },
                    "timeline_impact": {
                        "additional_days": 4,
                        "critical_path": True
                    }
                },
                "material_requirements": {
                    "bedding_material": {
                        "type": "Sand",
                        "additional_percentage": 30,
                        "reason": "Clay soil conditions",
                        "quantity": "12 cubic yards",
                        "supplier": "Northeast Materials"
                    },
                    "backfill_composition": {
                        "specialized": True,
                        "mix_ratio": "60-30-10 (native soil-aggregate-sand)",
                        "reason": "Proper compaction in clay soil"
                    },
                    "conduit_protection": {
                        "additional": True,
                        "type": "Rockshield wrapping",
                        "reason": "High rock content in soil"
                    }
                },
                "equipment_recommendations": {
                    "primary_excavator": {
                        "recommended": "Track-mounted mini-excavator",
                        "standard": "Backhoe loader",
                        "reason": "Maneuverability and soil conditions"
                    },
                    "shoring": {
                        "type": "Hydraulic",
                        "depth_threshold": 5,
                        "unit": "feet"
                    },
                    "dewatering": {
                        "needed": "standby",
                        "equipment": "3-inch submersible pump",
                        "reason": "Seasonal water table fluctuations"
                    }
                },
                "regulatory_considerations": {
                    "local_permit": {
                        "lead_time": 20,
                        "unit": "business days",
                        "standard_lead_time": 10,
                        "authority": "Jacksonville Building Department"
                    },
                    "environmental": {
                        "required": True,
                        "type": "Wetlands proximity documentation",
                        "authority": "Florida DEP"
                    },
                    "utility_coordination": {
                        "required": True,
                        "authority": "Jacksonville Electric Authority",
                        "lead_time": 7,
                        "unit": "business days"
                    }
                },
                "automation_triggers": [
                    {
                        "id": "auto-material-1",
                        "name": "Material Order Generation",
                        "description": "Create and send material order based on requirements",
                        "time_savings": 2,
                        "time_unit": "hours",
                        "impact": "procurement",
                        "deadline_sensitive": True,
                        "critical_path": False,
                        "trigger_timing": "2_weeks_before_start",
                        "parameters": {
                            "supplier": "Northeast Materials",
                            "material_type": "Sand",
                            "additional_percentage": 30
                        }
                    },
                    {
                        "id": "permit-application",
                        "name": "Permit Application Workflow",
                        "description": "Start Jacksonville permit process with 20-day lead time",
                        "time_savings": 4,
                        "time_unit": "hours",
                        "impact": "schedule",
                        "deadline_sensitive": True,
                        "critical_path": True,
                        "trigger_timing": "immediate",
                        "parameters": {
                            "jurisdiction": "Jacksonville",
                            "permit_type": "underground utility",
                            "lead_time": 20,
                            "environmental_documentation": True,
                            "wetlands_proximity": True
                        }
                    },
                    {
                        "id": "equipment-scheduling",
                        "name": "Equipment Scheduling Agent",
                        "description": "Reserve track equipment and dewatering equipment",
                        "time_savings": 1.5,
                        "time_unit": "hours",
                        "impact": "logistics",
                        "deadline_sensitive": False,
                        "trigger_timing": "14_days_before_start",
                        "parameters": {
                            "equipment_type": ["track-mounted mini-excavator", "3-inch submersible pump"],
                            "duration": "project duration",
                            "special_requirements": "Rock saw attachment",
                            "availability_alert": "Limited track excavators in region"
                        }
                    }
                ]
            }
        },
        priority=85,  # High priority
        exact_match=False
    )
)

# Backup more general pattern for utilities in Jacksonville
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(utility|utilities|underground|conduit|pipe).+(Jacksonville|Florida)",
        is_regex=True,
        # Same response data as above
        response_data={
            "status": "success",
            "answer": """# Location-Based Utility Construction Insights

## Location Analysis for Jacksonville

### Site Conditions Overview
- **Soil Composition:** Clay-heavy with moderate rock content
- **Water Table:** 4-6 feet below grade, seasonal variation of 2 feet
- **Terrain:** Gently sloping with 3-5% grade changes
- **Underground Infrastructure Density:** Moderate (8 existing utilities per 100 linear feet)
- **Ground Frost Penetration:** 16-22 inches during winter months

### Production Metrics Impact
- **Trenching Rate:** 240-280 ft/day (15% below company average)
- **Conduit Installation:** 320-350 ft/day (standard efficiency)
- **Backfill Operations:** 275-300 ft/day (10% below average due to soil composition)
- **Project Timeline Adjustment:** Additional 4 days required for this location

### Material Requirements
- **Bedding Material:** Requires 30% additional sand bedding due to clay soil conditions
- **Backfill Composition:** Need specialized mix ratio for proper compaction
- **Conduit Protection:** Additional protective measures needed due to rock content
- **Recommended Supplier:** Northeast Materials (closest to site, special contractor rates)

### Equipment Recommendations
- **Primary Excavator:** Track-mounted mini-excavator recommended over standard backhoe
- **Shoring Requirements:** Hydraulic shoring system for depths over 5 feet
- **Dewatering Equipment:** Submersible pump on standby for seasonal water table fluctuations
- **Specialized Tools:** Rock saw attachment recommended for trenching

### Regulatory Considerations
- **Local Permit Lead Time:** 20 business days (vs. standard 10 days)
- **Environmental Requirements:** Wetlands proximity documentation required
- **Utility Coordination:** Jacksonville Electric Authority coordination needed
- **Traffic Control Plan:** Modified requirements due to residential zoning

### Actionable Insight Summary
1. **Schedule Impact:** Begin permit process immediately to avoid critical path delays
2. **Material Procurement:** Order additional 12 cubic yards of bedding sand
3. **Equipment Scheduling:** Reserve track equipment 14 days before project start
4. **Crew Configuration:** Add one additional laborer for proper soil handling

### Automation Opportunities
The following processes can be automated for this project:
1. **Materials Order Preparation** - Additional bedding and specialized backfill calculation and ordering
2. **Permit Application Workflow** - Streamlined application for 20-day lead time permits
3. **Equipment Scheduling** - Reservation of track and dewatering equipment
4. **Environmental Compliance** - Generation of wetlands proximity documentation""",
            "classification": {
                "category": "LOCATION_UTILITIES_INSIGHTS",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "utility-location-guide-2025",
                    "title": "Regional Utility Installation Standards",
                    "page": 42,
                    "confidence": 0.96,
                    "excerpt": "Clay-heavy soils in the northeastern Florida region require 25-30% additional bedding material compared to sandy soils. Production rates for trenching operations should be adjusted by 15-20% to account for the increased difficulty."
                },
                {
                    "id": "jax-utilities-handbook",
                    "title": "Jacksonville Utilities Installation Handbook",
                    "page": 18,
                    "confidence": 0.94,
                    "excerpt": "Permit applications for underground utility work in Clay Hill and surrounding areas require additional environmental documentation due to wetlands proximity. Allow 15-20 business days for permit processing."
                }
            ],
            "metadata": {
                "category": "LOCATION_UTILITIES_INSIGHTS",
                "query_type": "location_analysis",
                "render_type": "utilities_location_insight",
                "location_context": {
                    "location_name": "Jacksonville, Florida",
                    "coordinates": {
                        "lat": 30.1846,
                        "lng": -81.8543
                    },
                    "region_type": "suburban",
                    "soil_type": "clay-heavy",
                    "local_jurisdiction": "Jacksonville, FL"
                },
                "production_metrics": {
                    "trenching_rate": {
                        "expected": 260,
                        "unit": "ft/day",
                        "company_average": 305,
                        "variance": -15
                    },
                    "conduit_installation": {
                        "expected": 335,
                        "unit": "ft/day",
                        "company_average": 340,
                        "variance": -1.5
                    },
                    "backfill_operations": {
                        "expected": 287,
                        "unit": "ft/day",
                        "company_average": 320,
                        "variance": -10
                    },
                    "timeline_impact": {
                        "additional_days": 4,
                        "critical_path": True
                    }
                },
                "material_requirements": {
                    "bedding_material": {
                        "type": "Sand",
                        "additional_percentage": 30,
                        "reason": "Clay soil conditions",
                        "quantity": "12 cubic yards",
                        "supplier": "Northeast Materials"
                    },
                    "backfill_composition": {
                        "specialized": True,
                        "mix_ratio": "60-30-10 (native soil-aggregate-sand)",
                        "reason": "Proper compaction in clay soil"
                    },
                    "conduit_protection": {
                        "additional": True,
                        "type": "Rockshield wrapping",
                        "reason": "High rock content in soil"
                    }
                },
                "equipment_recommendations": {
                    "primary_excavator": {
                        "recommended": "Track-mounted mini-excavator",
                        "standard": "Backhoe loader",
                        "reason": "Maneuverability and soil conditions"
                    },
                    "shoring": {
                        "type": "Hydraulic",
                        "depth_threshold": 5,
                        "unit": "feet"
                    },
                    "dewatering": {
                        "needed": "standby",
                        "equipment": "3-inch submersible pump",
                        "reason": "Seasonal water table fluctuations"
                    }
                },
                "regulatory_considerations": {
                    "local_permit": {
                        "lead_time": 20,
                        "unit": "business days",
                        "standard_lead_time": 10,
                        "authority": "Jacksonville Building Department"
                    },
                    "environmental": {
                        "required": True,
                        "type": "Wetlands proximity documentation",
                        "authority": "Florida DEP"
                    },
                    "utility_coordination": {
                        "required": True,
                        "authority": "Jacksonville Electric Authority",
                        "lead_time": 7,
                        "unit": "business days"
                    }
                },
                "automation_triggers": [
                    {
                        "id": "auto-material-1",
                        "name": "Material Order Generation",
                        "description": "Create and send material order based on requirements",
                        "time_savings": 2,
                        "time_unit": "hours",
                        "impact": "procurement",
                        "deadline_sensitive": True,
                        "critical_path": False,
                        "trigger_timing": "2_weeks_before_start",
                        "parameters": {
                            "supplier": "Northeast Materials",
                            "material_type": "Sand",
                            "additional_percentage": 30
                        }
                    },
                    {
                        "id": "permit-application",
                        "name": "Permit Application Workflow",
                        "description": "Start Jacksonville permit process with 20-day lead time",
                        "time_savings": 4,
                        "time_unit": "hours",
                        "impact": "schedule",
                        "deadline_sensitive": True,
                        "critical_path": True,
                        "trigger_timing": "immediate",
                        "parameters": {
                            "jurisdiction": "Jacksonville",
                            "permit_type": "underground utility",
                            "lead_time": 20,
                            "environmental_documentation": True,
                            "wetlands_proximity": True
                        }
                    },
                    {
                        "id": "equipment-scheduling",
                        "name": "Equipment Scheduling Agent",
                        "description": "Reserve track equipment and dewatering equipment",
                        "time_savings": 1.5,
                        "time_unit": "hours",
                        "impact": "logistics",
                        "deadline_sensitive": False,
                        "trigger_timing": "14_days_before_start",
                        "parameters": {
                            "equipment_type": ["track-mounted mini-excavator", "3-inch submersible pump"],
                            "duration": "project duration",
                            "special_requirements": "Rock saw attachment",
                            "availability_alert": "Limited track excavators in region"
                        }
                    }
                ]
            }
        },
        priority=75,  # Lower priority than the specific pattern
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
    
    if not matching_responses:
        return None
    
    # Return the highest priority response
    return sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0].response_data
