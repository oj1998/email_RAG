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

# Construction Aggregates Alternative
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern="What construction aggregates can we use as an alternative to crushed stone?",
        is_regex=False,
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
                    "is_exact_match": True
                }
            }
        },
        exact_match=True
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
