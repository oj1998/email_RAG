# condensed_hardcoded_responses.py
from typing import Dict, Any, List, Optional
import re

class HardcodedResponse:
    def __init__(self, query_pattern: str, is_regex: bool, response_data: Dict[str, Any], 
                 exact_match: bool = False, priority: int = 0):
        self.query_pattern = query_pattern
        self.is_regex = is_regex
        self.response_data = response_data
        self.exact_match = exact_match
        self.priority = priority
    
    def matches(self, query: str) -> bool:
        if self.exact_match:
            return query.strip().lower() == self.query_pattern.lower()
        elif self.is_regex:
            return bool(re.search(self.query_pattern, query, re.IGNORECASE))
        else:
            return self.query_pattern.lower() in query.lower()

HARDCODED_RESPONSES: List[HardcodedResponse] = []

# 1. Location-Based Utilities Analysis (Fort Wayne)
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(material|utility|conduit|underground).+(requirement|order|installation|consideration).+(Fort Wayne|Indiana)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Location-Based Utility Construction Insights - Fort Wayne

## Site Conditions Overview
- **Soil Composition:** Clay-heavy with moderate rock content
- **Water Table:** 4-6 feet below grade, seasonal variation of 2 feet
- **Underground Infrastructure Density:** Moderate (8 existing utilities per 100 linear feet)
- **Ground Frost Penetration:** 16-22 inches during winter months

## Material Requirements
- **Bedding Material:** Requires 30% additional sand bedding due to clay soil conditions
- **Backfill Composition:** Specialized mix ratio for proper compaction
- **Conduit Protection:** Additional protective measures needed due to rock content
- **Recommended Supplier:** Northeast Materials (closest to site, special contractor rates)

## Equipment Recommendations
- **Primary Excavator:** Track-mounted mini-excavator recommended over standard backhoe
- **Shoring Requirements:** Hydraulic shoring system for depths over 5 feet
- **Dewatering Equipment:** Submersible pump on standby for seasonal water table fluctuations

## Regulatory Considerations
- **Local Permit Lead Time:** 20 business days (vs. standard 10 days)
- **Environmental Requirements:** Wetlands proximity documentation required
- **Utility Coordination:** Local electric authority coordination needed""",
            "classification": {"category": "LOCATION_UTILITIES_INSIGHTS", "confidence": 1.0},
            "metadata": {
                "category": "LOCATION_UTILITIES_INSIGHTS",
                "render_type": "utilities_location_insight",
                "location_context": {"location_name": "Fort Wayne, Indiana", "soil_type": "clay-heavy"}
            }
        },
        priority=85
    )
)

# 2. Directional Drilling Under Roads
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(how|procedure|perform).*(directional|horizontal|HDD|drill).*(road|highway|utilities)",
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

## Safety Requirements
- Call 811 utility notification 72 hours before drilling
- Maintain pothole excavations every 50 feet in congested areas
- Use ground penetrating radar when utilities are within 36 inches
- Install tracer wire with all non-metallic conduits""",
            "classification": {"category": "DIRECTIONAL_DRILLING", "confidence": 1.0},
            "metadata": {
                "category": "DIRECTIONAL_DRILLING",
                "render_type": "directional_drilling_specs",
                "project_context": {"crossing_type": "highway", "soil_conditions": "mixed clay/rock"}
            }
        },
        priority=90
    )
)

# 3. Aerial Fiber Installation in Rural Mountains
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(consideration|install).*(aerial|fiber).*(rural|mountain)",
        is_regex=True,
        response_data={
            "status": "success",
            "answer": """# Aerial Fiber Installation in Mountainous Rural Areas

## Terrain-Specific Considerations
1. **Access Road Limitations:** Many rural mountain areas lack proper roads - use tracked vehicles, ATVs, or helicopter support
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

2. **Span Lengths in Mountainous Terrain:**
   - Reduce standard spans by 20% in areas with high wind exposure
   - Maximum 200-foot spans across valleys (vs. 300-foot standard)
   - Use heavy-duty suspension clamps at all angle points

## Safety Protocols for Remote Work
- Minimum 3-person crews for mountain work (vs. 2-person standard)
- Wilderness first aid certification for at least one crew member
- Daily check-in procedures with base operations
- Emergency evacuation plan for each work site""",
            "classification": {"category": "RURAL_INSTALLATION", "confidence": 1.0},
            "metadata": {
                "category": "RURAL_INSTALLATION",
                "render_type": "rural_installation_guide",
                "terrain_context": {"terrain_type": "mountainous", "access_difficulty": "high"}
            }
        },
        priority=85
    )
)

# 4. Fiber Optic Splicing in Wet Conditions
HARDCODED_RESPONSES.append(
    HardcodedResponse(
        query_pattern=r"(safety|procedure).*(splic|fiber|optic).*(wet|rain|weather)",
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

## Emergency Contact Protocol
- Notify NOC immediately upon arrival at damage site
- Update outage management system every 30 minutes
- Contact local power company if poles are compromised
- Document all safety hazards in field report""",
            "classification": {"category": "FIBER_OPTIC_EMERGENCY", "confidence": 1.0},
            "metadata": {
                "category": "FIBER_OPTIC_EMERGENCY",
                "render_type": "fiber_optic_safety",
                "is_emergency_response": True,
                "emergency_type": "storm_damage"
            }
        },
        priority=95
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
    return sorted(matching_responses, key=lambda x: x.priority, reverse=True)[0].response_data
