
# Add this to your HARDCODED_RESPONSES list in hardcoded_responses.py

# Emergency Storm Damage Assessment Response
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(emergency|storm|damage|water intrusion|membrane tear|roof damage).+(medical center|hospital|healthcare|riverside|assessment|protocol|inspection)",
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
