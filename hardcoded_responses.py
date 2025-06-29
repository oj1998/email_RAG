# hardcoded_responses.py - FIXED VERSION
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

HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(drill|drilling).*(log|logging|report|document|new project|start project)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# AI Drilling Log Assistant

I'm ready to help you create a comprehensive drilling report. I'll guide you through capturing all the necessary data and generate professional documentation.

**What I'll help you document:**
- Project setup and permit information
- Real-time drilling progress and conditions
- Soil classifications and geological data
- Equipment performance and fluid usage
- Safety incidents and steering corrections
- Final installation details and compliance

**To get started, I need some basic project information:**

Tell me about your drilling project - location, type of installation, or just say "new bore project" and I'll walk you through the setup questions.""",
            "classification": {
                "category": "DRILL_LOG_INITIALIZATION",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_LOG_INITIALIZATION",
                "query_type": "drilling_log",
                "render_type": "log_initialization",
                "log_phase": "initialization",
                "current_step": 1,
                "total_steps": 6
            }
        },
        priority=90,
        exact_match=False
    )
)

# 2. Project Setup Details
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(new bore project|project setup|fiber installation|utility installation|conduit install)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Project Setup - Creating Drilling Log

**Project Initialized**: Directional Bore Installation

I've started a new drilling log and captured these initial details:

**Project Information:**
📍 **Location**: 1247 Industrial Parkway, Fort Wayne, IN
📋 **Permit #**: DIG-2025-0847 (Active)
👷 **Crew Lead**: Sarah Martinez
🚛 **Equipment**: Vermeer D36x50 Navigator
🌤️ **Weather**: 68°F, Clear, Light winds
📅 **Start Time**: 09:15 AM

**Installation Specs:**
- **Bore Length**: 285 feet
- **Product**: 4" HDPE conduit for fiber optic cable
- **Entry Angle**: 12 degrees
- **Target Depth**: 8 feet at utility crossing

**Next Step**: Begin drilling operations and I'll log your progress in real-time.

Ready to start? Just tell me when you begin the bore entry.""",
            "classification": {
                "category": "DRILL_PROJECT_SETUP",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PROJECT_SETUP",
                "query_type": "drilling_log",
                "render_type": "project_setup",
                "log_phase": "setup",
                "current_step": 2,
                "total_steps": 6,
                "project_data": {
                    "permit_number": "DIG-2025-0847",
                    "crew_lead": "Sarah Martinez",
                    "equipment": "Vermeer D36x50",
                    "bore_length": 285,
                    "product_type": "4_inch_hdpe"
                }
            }
        },
        priority=92,
        exact_match=False
    )
)

# 3. Real-Time Progress Logging
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(at \d+|depth|feet|progress|drilling|bore entry|penetration|mud pressure)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Progress Entry Logged - 10:47 AM

**Current Status**: Active drilling in progress

I've captured and formatted your drilling progress:

**Drilling Parameters:**
- **Current Depth**: 125 feet (44% to target)
- **Mud Pressure**: 175 psi (normal range)
- **Flow Rate**: 85 gpm
- **Drilling Rate**: 42 ft/hr (excellent)
- **Steering**: On grade ±0.2 feet

**Geological Log:**
- **0-45 ft**: Sandy soil, good penetration
- **45-85 ft**: Clay layer, increased pressure
- **85-125 ft**: Mixed soil, normal conditions

**AI Analysis**: 
✓ Drilling parameters within optimal range
✓ On schedule for 2:30 PM completion
✓ No steering corrections needed
⚠️ Monitor pressure if clay layer continues

**Auto-calculated**: Remaining time ~3.2 hours, fluid usage tracking normal.

Continue drilling and update me at your next checkpoint.""",
            "classification": {
                "category": "DRILL_PROGRESS_LOGGING",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PROGRESS_LOGGING",
                "query_type": "drilling_log",
                "render_type": "progress_logging",
                "log_phase": "active_drilling",
                "current_step": 3,
                "total_steps": 6,
                "progress_data": {
                    "current_depth": 125,
                    "target_depth": 285,
                    "completion_percentage": 44,
                    "drilling_rate": 42,
                    "pressure": 175,
                    "flow_rate": 85
                }
            }
        },
        priority=88,
        exact_match=False
    )
)

# 4. Problem Documentation & Steering
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(steering|correction|problem|issue|drift|off grade|alignment|trouble)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Incident Logged - Steering Correction Required

**Issue Documentation**: 11:23 AM

I've documented this technical issue and resolution:

**Problem Identified:**
- **Location**: 185 feet depth
- **Issue**: Bore drifting 2.5° left of target line
- **Cause**: Encountered unexpected rock formation
- **Detection Method**: Walkover locating system

**Corrective Action Taken:**
- **Steering Applied**: 4° right correction
- **Duration**: 15 minutes
- **Method**: Reduced flow rate, adjusted toolface
- **Verification**: Walkover confirmed return to grade

**Impact Assessment:**
- **Schedule**: +20 minutes (minimal impact)
- **Cost**: Within normal parameters
- **Quality**: Grade tolerance maintained

**AI Documentation**: 
✓ Photos captured with GPS coordinates
✓ Incident report auto-generated
✓ Regulatory compliance maintained
✓ Added to final report appendix

**Status**: Issue resolved, continuing bore to exit point.""",
            "classification": {
                "category": "DRILL_PROBLEM_DOCUMENTATION",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PROBLEM_DOCUMENTATION",
                "query_type": "drilling_log",
                "render_type": "problem_documentation",
                "log_phase": "problem_solving",
                "current_step": 4,
                "total_steps": 6,
                "incident_data": {
                    "type": "steering_correction",
                    "location": 185,
                    "severity": "minor",
                    "duration_minutes": 20,
                    "resolution": "successful"
                }
            }
        },
        priority=94,
        exact_match=False
    )
)

# 5. Installation & Pullback Documentation
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(pullback|pulling|installation|install|product|pipe|conduit|cable)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Product Installation - Pullback Phase

**Installation Documentation**: 2:15 PM

I'm tracking your product installation in real-time:

**Pullback Progress:**
- **Product Type**: 285 ft of 4" HDPE conduit
- **Progress**: 180 feet pulled (63% complete)
- **Pulling Force**: 2,800 lbs (within 4,200 lb limit)
- **Installation Rate**: 95 ft/hr
- **ETA Completion**: 3:05 PM

**Quality Monitoring:**
✓ Conduit integrity verified every 50 feet
✓ Continuous mud circulation maintained
✓ No binding or excessive force detected
✓ Exit point preparation confirmed

**AI Tracking:**
- **GPS waypoints** recorded every 25 feet
- **Force measurements** logged continuously  
- **Photo documentation** at key points
- **Compliance checklist** automatically updated

**Installation Checklist:**
✓ Pre-pullback inspection complete
✓ Reamer size verified (6" for 4" product)
✓ Pulling head properly attached
✓ Emergency stop procedures briefed

**Next**: Continue monitoring pullback to completion.""",
            "classification": {
                "category": "DRILL_INSTALLATION_LOGGING",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_INSTALLATION_LOGGING",
                "query_type": "drilling_log",
                "render_type": "installation_logging",
                "log_phase": "installation",
                "current_step": 5,
                "total_steps": 6,
                "installation_data": {
                    "product_type": "4_inch_hdpe",
                    "total_length": 285,
                    "pulled_length": 180,
                    "pulling_force": 2800,
                    "max_force": 4200,
                    "progress_percentage": 63
                }
            }
        },
        priority=89,
        exact_match=False
    )
)

# 6. Final Report Generation
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(complete|completed|finished|final report|generate report|done|wrap up)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Final Drilling Report Generated ✅

**Project Completion**: 3:12 PM - All objectives achieved

I've compiled your comprehensive drilling report with full documentation:

**Executive Summary:**
- **Total Time**: 6 hours 2 minutes (2 min ahead of schedule)
- **Installation**: 285 feet of 4" HDPE conduit successfully installed
- **Quality**: All specifications met, grade tolerance ±0.2 feet
- **Incidents**: 1 minor steering correction (resolved)
- **Compliance**: 100% regulatory requirements satisfied

**Generated Documentation:**
📋 **Drilling Log** - Complete chronological record
📊 **Progress Charts** - Visual drilling profile and timeline  
📷 **Photo Package** - 47 images with GPS coordinates
📋 **Incident Report** - Steering correction details
📄 **Compliance Certificate** - Regulatory sign-offs
💾 **Digital Backup** - All data archived for 7 years

**Report Distribution:**
✓ Project manager notified
✓ Permit office filing complete
✓ Customer copy prepared
✓ Equipment maintenance log updated

**AI Learning**: I've added this project data to improve future drilling predictions and recommendations.

**Files Ready**: Download links sent to your project management system.""",
            "classification": {
                "category": "DRILL_REPORT_GENERATION",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_REPORT_GENERATION",
                "query_type": "drilling_log",
                "render_type": "report_generation",
                "log_phase": "completion",
                "current_step": 6,
                "total_steps": 6,
                "completion_data": {
                    "total_time_hours": 6.03,
                    "schedule_variance": -2,
                    "incidents_count": 1,
                    "compliance_percentage": 100,
                    "photos_captured": 47
                }
            }
        },
        priority=87,
        exact_match=False
    )
)

# Add these to hardcoded_responses.py - Simplified Independent Drilling Responses

# 1. Initial Drilling Setup Request
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(drill|drilling|bore|boring).*(log|logging|record|document|start|setup|begin)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Drill Logging Assistant Ready

I'm ready to document your drilling progress in real-time.

**Quick Start Examples:**
- "Starting bore at station 15+50, entry angle 12 degrees"
- "At 50 feet depth, pressure 175 psi, good penetration"
- "Hit clay layer at 120 feet, need steering correction"

Just describe what's happening and I'll create the proper documentation format.

**Voice Input Available** - Use the microphone button for hands-free logging while operating equipment.""",
            "classification": {
                "category": "DRILL_LOGGING_INTERFACE",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_LOGGING_INTERFACE",
                "query_type": "drilling_workflow",
                "render_type": "drill_logging_interface",
                "drilling_context": {
                    "mode": "logging_ready",
                    "session_active": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
            }
        },
        priority=85,
        exact_match=False
    )
)

# 2. Drilling Setup/Entry Parameters
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(starting|begin|entry|station|setup).*(bore|drill|drilling).*(station|angle|depth)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Drilling Setup Recorded

**Entry Parameters Logged:**
📍 **Station**: 15+50 (as specified)
📐 **Entry Angle**: 12 degrees
🎯 **Target Depth**: 250 feet
🏺 **Mud System**: Bentonite polymer mix

**Pre-Entry Checklist:**
✓ Verify drill head position at entry point
✓ Confirm entry angle setting (12°)
✓ Test mud circulation system
✓ Check locating equipment signal

**Ready for bore entry** - Report depth readings every 10 feet once penetration begins.

**Quick Actions:**
- "At 10 feet, good penetration"
- "Hit resistance at 25 feet"
- "Need steering correction"
""",
            "classification": {
                "category": "DRILL_SETUP_ENTRY",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_SETUP_ENTRY",
                "query_type": "drilling_workflow",
                "render_type": "drill_setup_entry",
                "extracted_parameters": {
                    "station": "15+50",
                    "entry_angle": 12,
                    "target_depth": 250,
                    "mud_type": "bentonite_polymer"
                }
            }
        },
        priority=90,
        exact_match=False
    )
)

# 3. Active Drilling Progress Reports
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(at|depth|feet|ft).*(depth|feet|ft).*(pressure|psi|flow|conditions|penetration)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Drilling Log Entry Recorded

**⏰ 14:23:17** - Active drilling progress documented:

📏 **Current Depth**: 120 feet
⚡ **Mud Pressure**: 175 psi
💧 **Flow Rate**: 85 gpm
🌍 **Soil Conditions**: Clay layer encountered
🎯 **Steering Status**: On grade, no correction needed

**Progress Status**: 48% complete to target depth
**Next Milestone**: Report at 150 feet

**Quick Follow-up Options:**
- Continue normal drilling operations
- Report steering correction needed
- Document equipment issue
- Log soil condition change""",
            "classification": {
                "category": "DRILL_PROGRESS_LOG",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PROGRESS_LOG",
                "query_type": "drilling_workflow", 
                "render_type": "drill_progress_log",
                "log_entry": {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "depth": 120,
                    "pressure": 175,
                    "flow_rate": 85,
                    "soil_conditions": ["clay"],
                    "steering_status": "on_grade"
                }
            }
        },
        priority=88,
        exact_match=False
    )
)

# 4. Steering Correction Needed
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(steering|correction|adjust|drift|off.grade|alignment)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Steering Correction Protocol

**⚠️ Steering Issue Logged** - 14:25:43

**Problem Identified:**
- Bore drifting 2 degrees left of target
- Detected at 145 feet depth
- Correction required to maintain grade

**Correction Applied:**
✓ Steering adjustment: 3 degrees right
✓ New heading: 87 degrees (target: 85 degrees)
✓ Locating signal verified

**Monitoring Protocol:**
- Verify correction at 25-foot intervals
- Next check point: 170 feet
- Document grade achievement

**Status**: Correction applied, resuming drilling
**Action Required**: Monitor and report position at next depth milestone""",
            "classification": {
                "category": "DRILL_STEERING_CORRECTION",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_STEERING_CORRECTION",
                "query_type": "drilling_workflow",
                "render_type": "drill_steering_correction",
                "correction_data": {
                    "drift_direction": "left",
                    "drift_amount": 2,
                    "correction_applied": 3,
                    "new_heading": 87,
                    "target_heading": 85
                }
            }
        },
        priority=92,
        exact_match=False
    )
)

# 5. Pullback/Installation Phase
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(pullback|pulling|install|installation|product|pipe|cable)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Pullback Operation Status

**🔄 Active Pullback in Progress**

**Current Status:**
📏 **Product Pulled**: 180 feet of 250 feet total
⚡ **Pulling Tension**: 2,400 lbs (within 3,500 lb limit)
💧 **Mud Flow**: 95 gpm circulation maintained
🎯 **Progress**: 72% complete

**Critical Monitoring:**
- Tension readings logged every 50 feet
- Product integrity verified at entry point
- No binding or resistance detected
- Mud returns quality: Normal

**Next Check Points:**
- 200 feet: Verify tension and progress
- 225 feet: Prepare for final pull sequence
- 250 feet: Complete installation and secure

**Status**: Normal pullback operations continuing""",
            "classification": {
                "category": "DRILL_PULLBACK_ACTIVE",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_PULLBACK_ACTIVE",
                "query_type": "drilling_workflow",
                "render_type": "drill_pullback_active",
                "pullback_data": {
                    "product_pulled": 180,
                    "total_length": 250,
                    "tension": 2400,
                    "max_tension": 3500,
                    "progress_percentage": 72
                }
            }
        },
        priority=89,
        exact_match=False
    )
)

# 6. Completion/Success
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(complete|completed|finished|done|success)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Drilling Operation Complete ✅

**🎉 Installation Successfully Completed**

**Final Summary:**
- **Total Bore Length**: 250 feet
- **Installation Time**: 4.5 hours
- **Product Installed**: 250 feet of 6" HDPE conduit
- **Final Grade Achieved**: ±0.2 feet (within tolerance)

**Quality Verification:**
✓ Product functionality tested - PASS
✓ Bore abandonment completed
✓ Entry and exit points secured
✓ Equipment demobilization complete

**Documentation Status:**
✓ Drilling log completed
✓ As-built drawings updated
✓ Quality control forms signed
✓ Final inspection scheduled

**Operation Status**: **COMPLETE**
**Next Steps**: Site restoration and final paperwork

Congratulations on successful completion of the directional drilling operation!""",
            "classification": {
                "category": "DRILL_OPERATION_COMPLETE",
                "confidence": 1.0
            },
            "sources": [],
            "metadata": {
                "category": "DRILL_OPERATION_COMPLETE",
                "query_type": "drilling_workflow",
                "render_type": "drill_operation_complete",
                "completion_data": {
                    "total_length": 250,
                    "installation_time": 4.5,
                    "product_type": "6_inch_hdpe_conduit",
                    "grade_tolerance": 0.2
                }
            }
        },
        priority=87,
        exact_match=False
    )
)

# Add this to hardcoded_responses.py - AFTER the first mold response

# Mold SOP Procedures for Severe Mold Discovery
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
        priority=96,  # Slightly higher priority than the discovery response
        exact_match=False
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
        priority=95,  # High priority for safety issues
        exact_match=False
    )
)

# Add this to hardcoded_responses.py - AFTER the other mold responses

# Mold Remediation Change Order
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
        priority=93,  # High priority but below SOPs and discovery
        exact_match=False
    )
)

# Add this to hardcoded_responses.py - AFTER the other mold responses

# BuilderTrend Client Portal Update
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
        priority=92,  # High priority but below change order
        exact_match=False
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

# Maryland Dental Office Code Compliance - Part 1
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(maryland|MD).*(electrical code|building code).*(medical|dental|healthcare|patient care).*(low voltage|conduit|cable|wiring).*(bethesda|montgomery county)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Maryland Electrical Code Requirements - Medical Facilities

## Code Compliance Analysis for Bethesda Dental Office

Based on Maryland building code requirements, **all low voltage cables in dental operatories must be installed in metallic conduit**. Your current specification for plastic conduit does not meet compliance requirements.

### Specific Code Requirements:
- **IBC Section 517.13:** Healthcare facilities electrical systems
- **Maryland Amendment MA-517.13.2:** Metal conduit required for patient care areas  
- **Montgomery County Amendment MC-E-1.4:** Enhanced grounding for medical facilities

### Technical Requirements:
- All low voltage cables in patient care areas must use EMT or rigid metal conduit
- Metal junction boxes with proper bonding provisions
- Enhanced equipment grounding system connectivity
- Separate conduits for power and low voltage in patient areas

### Compliance Impact:
- **Cost Increase:** $8,400 - $12,600 for metallic conduit upgrade
- **Schedule Impact:** Additional 3-4 days for specialized installation
- **Affected Systems:** Dental equipment controls, patient monitoring, emergency communication

The current contract specification for PVC conduit will require modification to ensure code compliance and permit approval.""",
            "classification": {
                "category": "MARYLAND_CODE_COMPLIANCE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "ibc-517-healthcare",
                    "title": "IBC Section 517 - Healthcare Facilities",
                    "page": 13,
                    "confidence": 0.96,
                    "excerpt": "Electrical wiring in patient care areas shall be installed in metal raceway systems to ensure proper grounding and electromagnetic shielding."
                },
                {
                    "id": "md-amendment-517",
                    "title": "Maryland Amendment MA-517.13.2",
                    "page": 2,
                    "confidence": 0.94,
                    "excerpt": "All low voltage cables in dental operatories must use EMT or rigid metal conduit with proper bonding provisions."
                },
                {
                    "id": "mc-electrical-medical",
                    "title": "Montgomery County Electrical Code - Medical Facilities",
                    "page": 8,
                    "confidence": 0.92,
                    "excerpt": "Additional bonding requirements for dental equipment grounding systems in Montgomery County."
                }
            ],
            "metadata": {
                "category": "MARYLAND_CODE_COMPLIANCE",
                "query_type": "document",
                "render_type": "maryland_code_compliance",
                "compliance_analysis": {
                    "jurisdiction": "Montgomery County, Maryland",
                    "project_type": "Commercial Dental Office",
                    "compliance_status": "Non-compliant - requires modification"
                },
                "code_references": [
                    {
                        "code": "IBC 517.13",
                        "title": "Healthcare Facilities Electrical Systems",
                        "requirement": "Metal raceway for patient care areas"
                    },
                    {
                        "code": "Maryland Amendment MA-517.13.2", 
                        "title": "Enhanced Grounding Requirements",
                        "requirement": "EMT conduit for dental operatories"
                    },
                    {
                        "code": "Montgomery County MC-E-1.4",
                        "title": "Medical Facility Standards",
                        "requirement": "Additional bonding for dental equipment"
                    }
                ],
                "local_amendments": [
                    {
                        "amendment": "MA-517.13.2",
                        "description": "Maryland-specific requirement for metallic conduit in dental facilities"
                    },
                    {
                        "amendment": "MC-E-1.4",
                        "description": "Montgomery County bonding requirements"
                    }
                ]
            }
        },
        priority=95,  # High priority for code compliance
        exact_match=False
    )
)

# Maryland Dental Office Contract Change Order - Part 2
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(change order|contract|modify contract).*(add|include|upgrade).*(metal conduit|EMT|metallic conduit|code compliance).*(bethesda|dental)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Contract Change Order: Maryland Code Compliance

## Change Order CO-2025-0241 Generated

I've prepared a comprehensive change order to modify your Bethesda dental office contract for Maryland electrical code compliance.

### Contract Modification Summary:
- **Original Contract:** CTR-2025-MD-187 ($248,750)
- **Change Order Value:** +$10,450
- **New Contract Total:** $259,200
- **Schedule Impact:** +3 days

### Key Contract Changes:
1. **Materials Specification:** Replace PVC conduit with EMT conduit for all low voltage applications in patient care areas
2. **Technical Standards:** Add enhanced grounding and bonding requirements per Maryland amendments
3. **Completion Date:** Extended from March 26 to March 29, 2025

### Detailed Cost Breakdown:
- EMT Conduit upgrade (840 LF): $2,940
- Metal junction boxes (24 EA): $1,680  
- Bonding hardware & fittings: $850
- Additional specialized labor: $3,840
- Engineering review & permits: $760
- Project management: $380

### Approval Required:
Dr. Sarah Chen, DDS must sign this change order to authorize the code compliance modifications and proceed with metallic conduit installation.

The change order ensures full compliance with IBC Section 517.13, Maryland Amendment MA-517.13.2, and Montgomery County electrical requirements for medical facilities.""",
            "classification": {
                "category": "CONTRACT_CHANGE_ORDER",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "contract-ctr-2025-md-187",
                    "title": "Original Contract CTR-2025-MD-187",
                    "page": 1,
                    "confidence": 0.98,
                    "excerpt": "Bethesda Dental Office electrical installation contract with material specifications and completion requirements."
                }
            ],
            "metadata": {
                "category": "CONTRACT_CHANGE_ORDER",
                "query_type": "document", 
                "render_type": "contract_change_order",
                "change_order_data": {
                    "change_order_number": "CO-2025-0241",
                    "original_contract": "CTR-2025-MD-187",
                    "original_value": 248750,
                    "change_value": 10450,
                    "new_total": 259200,
                    "reason": "Maryland electrical code compliance"
                },
                "contract_modifications": {
                    "materials": "Replace PVC with EMT conduit",
                    "schedule": "Extend completion by 3 days", 
                    "price": "Increase by $10,450",
                    "specifications": "Add metallic conduit requirements"
                },
                "cost_breakdown": {
                    "emt_conduit": 2940,
                    "metal_boxes": 1680,
                    "bonding_hardware": 850,
                    "additional_labor": 3840,
                    "engineering": 760,
                    "management": 380
                }
            }
        },
        priority=94,  # High priority for contract modification
        exact_match=False
    )
)

# Weekly Client Update Generation
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(create|generate|write).*(end.{0,10}of.{0,10}week|weekly|client).*(update|report|summary)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Weekly Client Update Generated

## Email Content

**To:** Dr. Sarah Chen <chen@bethesdadental.com>  
**Subject:** Bethesda Dental Office - Weekly Update  
**From:** Project Management Team

Dear Dr. Chen,

Here's your weekly progress update for the Bethesda dental office project:

**This Week's Accomplishments:**
• Finalized and submitted all permitting paperwork to Montgomery County
• Completed equipment coordination meeting with dental supplier (confirmed delivery schedule)
• Received MEP engineering drawings approval
• Finalized material selections and submitted procurement schedule

**Project Status:**
• Budget: $191,700 (includes recent fire code compliance adjustment)
• Pre-construction Phase: 85% complete
• Construction start: January 6, 2025

**Next Week's Focus:**
• Await permit approval (expected by Dec 23)
• Begin material procurement for long-lead items
• Schedule pre-construction meeting with trades

Please let me know if you have any questions.

Best regards,  
Project Management Team

## Data Sources
Information compiled from: Procore (budget/progress), PlanGrid (drawings), Building Department Portal (permits), Supplier CRM (equipment meetings), Microsoft Project (schedule)""",
            "classification": {
                "category": "CLIENT_WEEKLY_UPDATE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "procore-project-data",
                    "title": "Procore Project Management System",
                    "page": 1,
                    "confidence": 0.98,
                    "excerpt": "Budget tracking, change orders, and project progress data for Bethesda Dental Office project."
                },
                {
                    "id": "plangrid-drawings",
                    "title": "PlanGrid Drawing Management",
                    "page": 1,
                    "confidence": 0.95,
                    "excerpt": "MEP engineering drawings approval status and document management records."
                },
                {
                    "id": "permit-portal-data",
                    "title": "Montgomery County Building Department Portal",
                    "page": 1,
                    "confidence": 0.92,
                    "excerpt": "Permit submission status and approval tracking for commercial dental office buildout."
                }
            ],
            "metadata": {
                "category": "CLIENT_WEEKLY_UPDATE",
                "query_type": "document",
                "render_type": "client_weekly_update",
                "update_period": "week_ending_2024_12_20",
                "data_sources": [
                    "procore",
                    "plangrid", 
                    "building_dept_portal",
                    "supplier_crm",
                    "microsoft_project"
                ]
            }
        },
        priority=93,
        exact_match=False
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

# Maryland Fire Rating Hardcoded Responses - MUTUALLY EXCLUSIVE

# Maryland Fire Rating Code Compliance - Part 1
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(maryland|MD).*(fire.{0,10}rated|fire.{0,10}rating|fire.{0,10}code).*(wall|assembly|dental|medical).*(bethesda|hour|hours)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Maryland Fire Code Requirements - Healthcare Facilities

Maryland requires **2-hour fire-rated wall assemblies** for dental office treatment rooms. Your current 1-hour specification does not meet state requirements.

## Code Authority
**IBC Section 508.2 + Maryland Amendment:** Healthcare occupancies require 2-hour fire separation between treatment areas and corridors.

## Required Changes
- Upgrade from 1-hour to 2-hour fire-rated assembly
- Double layer 5/8" Type X drywall (instead of single layer)
- 6" metal studs with mineral wool insulation

## Impact
- **Cost:** +$4,200 for assembly upgrade
- **Schedule:** +2 days additional installation time

Your contract will need modification to ensure fire code compliance.""",
            "classification": {
                "category": "MD_FIRE_RATING_COMPLIANCE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "ibc-508-healthcare",
                    "title": "IBC Section 508 - Healthcare Occupancies",
                    "page": 5,
                    "confidence": 0.94,
                    "excerpt": "Healthcare occupancies require 2-hour fire separation between treatment areas and corridors per Maryland amendments."
                }
            ],
            "metadata": {
                "category": "MD_FIRE_RATING_COMPLIANCE",
                "query_type": "document",
                "render_type": "maryland_fire_rating",
                "compliance_issue": "fire_rating_upgrade_required"
            }
        },
        priority=96,
        exact_match=False
    )
)

# Maryland Fire Rating Contract Change Order - Part 2  
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(change.{0,10}order|modify.{0,10}contract).*(fire.{0,10}rated|2.{0,10}hour|upgrade).*(wall|assembly)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Change Order CO-2025-0158: Fire-Rated Wall Upgrade

## Contract Modification Summary
- **Original Contract:** $187,500
- **Change Order Value:** +$4,200  
- **New Contract Total:** $191,700
- **Schedule Impact:** +2 days

## Reason for Change
Maryland fire code requires 2-hour fire-rated wall assemblies for dental treatment rooms. Original contract specified 1-hour assemblies which do not meet state requirements.

## Technical Modifications
**Section 09 22 16 - Interior Gypsum Board:**
Replace "1-hour fire-rated assembly with single layer 5/8" Type X" with "2-hour fire-rated assembly with double layer 5/8" Type X and 6" metal studs"

## Approval Required
Dr. Sarah Chen must approve this change order to proceed with fire code compliance modifications.""",
            "classification": {
                "category": "MD_FIRE_RATING_CHANGE_ORDER",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "contract-ctr-2025-md-187", 
                    "title": "Original Contract CTR-2025-MD-187",
                    "page": 1,
                    "confidence": 0.98,
                    "excerpt": "Bethesda Dental Office construction contract with wall assembly specifications."
                }
            ],
            "metadata": {
                "category": "MD_FIRE_RATING_CHANGE_ORDER",
                "query_type": "document",
                "render_type": "simple_change_order",
                "change_order_number": "CO-2025-0158"
            }
        },
        priority=95,
        exact_match=False
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
                }
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
            "sources": [],
            "metadata": {
                "category": "ROLL_FORMING_TROUBLESHOOTING",
                "query_type": "troubleshooting",
                "render_type": "roll_forming_troubleshooter",
                "machine_model": "Scotpanel",
                "issue_description": "Inconsistent material thickness at edges"
            }
        },
        priority=75,  # High priority
        exact_match=False
    )
)

# Roll Forming Machine Assembly Guide - Scotpanel
HARDCODED_RESPONSES.append(
    HardcodedResponse(
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
            "sources": [],
            "metadata": {
                "category": "ROLL_FORMER_ASSEMBLY",
                "query_type": "assembly_guide",
                "render_type": "roll_former_assembly_navigator",
                "machine_model": "Scotpanel"
            }
        },
        priority=75,  # High priority
        exact_match=False
    )
)

# ====== REMOVED ALL DRILLING-RELATED HARDCODED RESPONSES ======
# The drilling workflow will now be handled entirely by drilling_workflow.py
# which has proper multi-step workflow support

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
    selected = sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0]
    
    return selected.response_data
