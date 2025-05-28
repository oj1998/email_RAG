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

# Emergency Storm Damage Assessment Response - Triggered by specific phrase
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"i suspect membrane damage at the riverside medical center",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Emergency Storm Damage Assessment Protocol

## Immediate Response Required

Based on the reported conditions at Riverside Medical Center, this constitutes a **Critical Priority** emergency requiring immediate assessment and containment measures.

## Facility Risk Assessment
- **Facility Type:** Healthcare facility with critical operations
- **Roof System:** 40-year-old built-up roof showing age-related vulnerability
- **Water Intrusion Location:** OR wing - highest priority area
- **Operational Impact:** Patient safety and surgical capabilities at risk

## Priority Assessment Areas

### Critical Priority
1. **Membrane Tears & Water Intrusion**
   - Document all visible membrane damage
   - Assess interior water penetration extent
   - Verify structural deck integrity
   - Emergency containment required

### Urgent Priority  
2. **Critical Area Protection**
   - OR wing immediate isolation
   - HVAC system impact evaluation
   - Equipment protection measures
   - Temporary weather protection installation

### Standard Priority
3. **Overall System Evaluation**
   - Complete roof system inspection
   - Drainage system functionality
   - Perimeter and flashing conditions

## Emergency Containment Measures
- Deploy emergency tarping over affected areas
- Establish interior water collection systems
- Isolate affected HVAC zones
- Document all damage with photographic evidence

## Next Steps
1. Complete comprehensive damage assessment using standardized checklist
2. Generate preliminary cost estimates for insurance documentation
3. Coordinate with facility management for repair scheduling
4. Engage emergency roofing contractor for immediate containment

## Timeline Considerations
- Assessment completion: 2-4 hours
- Emergency containment: Immediate (within 1 hour)
- Repair planning: 24-48 hours depending on damage extent""",
            "classification": {
                "category": "EMERGENCY_STORM_DAMAGE",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "emergency-roofing-protocols",
                    "title": "Emergency Roofing Assessment Guidelines for Healthcare Facilities",
                    "page": 23,
                    "confidence": 0.97,
                    "excerpt": "Healthcare facilities require immediate response protocols for roof emergencies due to critical operations and patient safety concerns. OR wings receive highest priority classification."
                },
                {
                    "id": "built-up-roof-storm-damage",
                    "title": "Built-Up Roof Storm Damage Assessment Manual",
                    "page": 67,
                    "confidence": 0.94,
                    "excerpt": "Membrane tears in 40+ year old built-up roofing systems typically indicate comprehensive failure requiring immediate containment and assessment of underlying structural elements."
                },
                {
                    "id": "healthcare-facility-emergency-response",
                    "title": "Healthcare Facility Emergency Response Standards",
                    "page": 12,
                    "confidence": 0.92,
                    "excerpt": "Water intrusion in operating room areas requires immediate containment and specialized remediation protocols to maintain sterile environments and prevent operational disruption."
                }
            ],
            "metadata": {
                "category": "EMERGENCY_STORM_DAMAGE",
                "query_type": "document",
                "render_type": "storm_damage_assessment",
                "is_emergency_response": True,
                "emergency_type": "storm_damage",
                "facility_info": {
                    "name": "Riverside Medical Center",
                    "type": "Healthcare Facility", 
                    "roof_age": 40,
                    "roof_type": "Built-up Roof (BUR)",
                    "square_footage": 15000,
                    "critical_areas": ["OR Wing", "ICU", "Emergency Department"],
                    "occupancy_status": "Fully Operational"
                },
                "weather_conditions": {
                    "current": "Post-storm, clearing",
                    "forecast": "Scattered showers next 24 hours",
                    "wind_speed": "15-20 mph, gusting to 30 mph", 
                    "temperature": "52°F",
                    "alerts": ["High Wind Advisory until 6 PM", "Potential flash flooding in low areas"]
                },
                "damage_categories": [
                    {
                        "id": "membrane-damage",
                        "name": "Membrane Tears & Punctures",
                        "severity": "critical",
                        "description": "Visible tears, punctures, or complete membrane failure causing water intrusion",
                        "checklist_items": [
                            "Document size and location of each tear",
                            "Check for water penetration into building",
                            "Assess structural deck condition beneath damage",
                            "Verify HVAC equipment hasn't been affected",
                            "Test membrane adhesion around damaged areas"
                        ],
                        "photo_requirements": [
                            "Wide shot showing damage extent",
                            "Close-up of each significant tear", 
                            "Interior water damage (if any)",
                            "Overall roof condition for comparison"
                        ],
                        "estimated_cost_range": "$8,000 - $25,000"
                    },
                    {
                        "id": "flashing-issues",
                        "name": "Flashing & Edge Detail Damage",
                        "severity": "urgent",
                        "description": "Compromised flashing, coping, or edge details that could lead to water infiltration", 
                        "checklist_items": [
                            "Inspect all perimeter flashing",
                            "Check penetration seals (vents, equipment)",
                            "Examine parapet wall conditions",
                            "Verify proper drainage at low points",
                            "Test for loose or missing fasteners"
                        ],
                        "photo_requirements": [
                            "Damaged flashing details",
                            "Penetration areas showing damage",
                            "Drainage issues or standing water",
                            "Before/after comparison photos"
                        ],
                        "estimated_cost_range": "$3,500 - $12,000"
                    },
                    {
                        "id": "equipment-damage", 
                        "name": "Rooftop Equipment Impact",
                        "severity": "urgent",
                        "description": "HVAC units, electrical equipment, or other rooftop installations damaged by debris",
                        "checklist_items": [
                            "Inspect all HVAC units for impact damage",
                            "Check electrical connections and housings", 
                            "Verify equipment anchoring systems",
                            "Test operational status of critical equipment",
                            "Document debris locations and sources"
                        ],
                        "photo_requirements": [
                            "Equipment damage close-ups",
                            "Debris impact locations",
                            "Overall equipment layout", 
                            "Any electrical hazards present"
                        ],
                        "estimated_cost_range": "$5,000 - $35,000"
                    },
                    {
                        "id": "drainage-blockage",
                        "name": "Drainage System Compromise", 
                        "severity": "standard",
                        "description": "Blocked drains, damaged gutters, or compromised water management systems",
                        "checklist_items": [
                            "Clear and inspect all roof drains",
                            "Check for standing water areas",
                            "Examine gutter and downspout integrity",
                            "Verify slope toward drainage points", 
                            "Document any overflow conditions"
                        ],
                        "photo_requirements": [
                            "Blocked drains with debris",
                            "Standing water areas",
                            "Damaged gutters or downspouts",
                            "Water flow patterns"
                        ],
                        "estimated_cost_range": "$1,500 - $8,000"
                    }
                ],
                "emergency_contacts": {
                    "facility_manager": "Dr. Sarah Johnson - (555) 123-4567",
                    "insurance_agent": "Mike Thompson, Alliance Insurance - (555) 987-6543",
                    "emergency_contractor": "24/7 Roof Repair Services - (555) 555-ROOF"
                },
                "context": {
                    "time_of_day": "morning",
                    "weather_status": "post_storm",
                    "facility_status": "operational",
                    "response_urgency": "critical"
                }
            }
        },
        priority=100,  # Highest priority for emergency responses
        exact_match=False
    )
)

# Healthcare Roofing Repair Protocol Response - Triggered by weekend repair query
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"membrane damaged confirmed.*we need repairs this weekend.*what is weekend repair protocol",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Healthcare Roofing Repair Protocol - Emergency Weekend Implementation

## Executive Summary

Based on completed damage assessment at Riverside Medical Center, we have **confirmed membrane failure with structural deck remaining intact**. The facility requires emergency repairs during the weekend downtime window to restore weather integrity while maintaining full healthcare compliance standards.

## Critical Timeline Requirements
- **Work Window:** Saturday 6 PM - Monday 6 AM (58-hour maximum)
- **Weather Window:** Optimal conditions confirmed for weekend
- **Completion Deadline:** Sunday 8 PM (10-hour safety buffer)
- **Total Repair Duration:** 25 hours across 6 phases

## Healthcare-Specific Compliance Framework

### Joint Commission Requirements
All repair work must maintain strict compliance with Joint Commission environment of care standards, including continuous air quality monitoring, infection control protocols, and documentation of all containment measures affecting patient areas.

### Critical Systems Coordination
The repair protocol includes specialized procedures for healthcare facilities:
- OR HVAC system isolation and restoration protocols
- Medical gas system protection during membrane work
- Emergency power system coordination
- Patient monitoring system continuity verification

## Repair Phase Overview

### Phase 1: Healthcare Facility Isolation (2 hours)
Establish medical-grade containment protocols with Infection Control Officer coordination, HVAC zone isolation, and negative pressure barriers with HEPA filtration systems.

### Phase 2: Emergency Weather Protection (4 hours)
Deploy commercial-grade temporary protection systems with 24/7 monitoring to prevent further water intrusion during repair operations.

### Phase 3: Healthcare-Grade Surface Preparation (6 hours)
Remove damaged membrane using healthcare-approved methods, including antimicrobial surface treatment and disinfection protocols meeting hospital indoor air quality standards.

### Phase 4: Medical-Grade Membrane Installation (8 hours)
Install new membrane system using healthcare-certified materials with zero-VOC emissions, antimicrobial properties, and heat-welded seams requiring 100% water testing verification.

### Phase 5: Critical Systems Reactivation (3 hours)
Restore all building systems with comprehensive air quality testing, Joint Commission documentation, and Infection Control sign-off before patient area reoccupancy.

### Phase 6: Healthcare Compliance Final Inspection (2 hours)
Complete final inspection including water testing, air quality verification, photographic documentation, and generation of all required compliance certificates.""",
            "classification": {
                "category": "HEALTHCARE_ROOFING_REPAIR",
                "confidence": 1.0
            },
            "sources": [
                {
                    "id": "00cf74dc-c3c3-4b89-8385-4fb9b0b3655e",
                    "title": "Joint Commission Healthcare Facility Environment Standards",
                    "document_id": "JC-ENV-2024-045",
                    "filename": "Joint Commission Healthcare Facility Environment Standards.txt",
                    "page": 45,
                    "confidence": 0.98,
                    "excerpt": "Emergency roofing repairs in healthcare facilities must maintain strict air quality standards and require Infection Control Officer oversight for all containment protocols affecting patient care areas."
                },
                {
                    "id": "00cf74dc-c3c3-4b89-8385-4fb9b0b3655e",
                    "title": "Joint Commission Healthcare Facility Environment Standards",
                    "document_id": "JC-ENV-2024-045",
                    "filename": "Joint Commission Healthcare Facility Environment Standards.txt",
                    "page": 23,
                    "confidence": 0.95,
                    "excerpt": "All roofing materials used in healthcare facilities must meet GREENGUARD Gold certification for indoor air quality and include antimicrobial properties to prevent contamination of medical environments."
                },
                {
                    "id": "00cf74dc-c3c3-4b89-8385-4fb9b0b3655e",
                    "title": "Joint Commission Healthcare Facility Environment Standards",
                    "document_id": "JC-ENV-2024-045",
                    "filename": "Joint Commission Healthcare Facility Environment Standards.txt",
                    "page": 78,
                    "confidence": 0.93,
                    "excerpt": "Weekend repair windows in healthcare facilities require coordination with critical systems including medical gas, emergency power, and HVAC systems to ensure continuous patient care capabilities."
                }
            ],
            "metadata": {
                "category": "HEALTHCARE_ROOFING_REPAIR",
                "query_type": "document",
                "render_type": "healthcare_roofing_repair",
                "is_emergency_repair": True,
                "facility_info": {
                    "name": "Riverside Medical Center",
                    "type": "Healthcare Facility",
                    "downtime_window": "Weekend: Saturday 6 PM - Monday 6 AM",
                    "critical_systems": ["OR HVAC", "Emergency Power", "Medical Gas Systems", "Patient Monitoring"]
                },
                "weather_window": {
                    "optimal_start": "Saturday 6:00 PM",
                    "completion_deadline": "Sunday 8:00 PM",
                    "weather_constraints": ["No precipitation expected", "Winds below 15 mph", "Temperature above 45°F"]
                }
            }
        },
        priority=95,  # High priority but below critical emergencies
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
