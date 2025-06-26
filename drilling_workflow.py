# drilling_workflow.py - Enhanced version with proper UI rendering

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
from datetime import datetime

class DrillingWorkflowStep(Enum):
    INITIAL_SETUP = "initial_setup"
    BORE_ENTRY = "bore_entry" 
    ACTIVE_DRILLING = "active_drilling"
    STEERING_CORRECTIONS = "steering_corrections"
    PULLBACK_PREP = "pullback_prep"
    PULLBACK_EXECUTION = "pullback_execution"
    COMPLETION = "completion"

class DrillingWorkflowHandler:
    def __init__(self, pool):
        self.pool = pool
        self.workflow_sessions = {}  # In production, store in database
    
    async def handle_drilling_workflow(
        self, 
        query: str, 
        conversation_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle multi-step drilling workflows"""
        
        # Check if this is a drilling-related query
        if not self._is_drilling_query(query):
            return None
            
        # Get or create workflow session
        session = await self._get_workflow_session(conversation_id)
        
        # Determine current step and next action
        current_step = self._determine_current_step(query, session)
        
        # Generate appropriate response based on workflow step
        response = await self._generate_workflow_response(
            query, current_step, session, context
        )
        
        # Update workflow session
        await self._update_workflow_session(conversation_id, session, current_step)
        
        return response
    
    def _is_drilling_query(self, query: str) -> bool:
        """Enhanced drilling query detection"""
        drilling_patterns = [
            r"drill|drilling|bore|boring",
            r"mud|bentonite|polymer",
            r"pullback|entry|station",
            r"pressure|flow|depth|angle",
            r"steering|correction|alignment",
            r"log.*drill|drill.*log|bore.*log",  # Add logging patterns
            r"start.*drill|begin.*drill|setup.*drill"  # Add setup patterns
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in drilling_patterns)
    
    async def _get_workflow_session(self, conversation_id: str) -> Dict[str, Any]:
        """Get or create drilling workflow session"""
        
        # Check database for existing session
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT workflow_data FROM drilling_sessions 
                WHERE conversation_id = $1
            """, conversation_id)
            
            if result:
                return json.loads(result['workflow_data'])
        
        # Create new session
        return {
            "session_id": f"drill_{conversation_id}_{datetime.now().timestamp()}",
            "current_step": DrillingWorkflowStep.INITIAL_SETUP.value,
            "parameters": {},
            "log_entries": [],
            "start_time": datetime.now().isoformat(),
            "questions_asked": [],
            "workflow_progress": {}
        }
    
    def _determine_current_step(self, query: str, session: Dict) -> DrillingWorkflowStep:
        """Determine what step we're in based on query and session state"""
        
        query_lower = query.lower()
        
        # Check for logging-specific queries (always render with logging interface)
        if any(word in query_lower for word in ['log', 'logging', 'record', 'document']):
            # If they're asking to log drilling, start active drilling
            if any(word in query_lower for word in ['drill', 'drilling', 'bore', 'boring']):
                return DrillingWorkflowStep.ACTIVE_DRILLING
        
        # Check for specific workflow triggers
        if any(word in query_lower for word in ['start', 'begin', 'initial', 'setup']):
            return DrillingWorkflowStep.INITIAL_SETUP
        elif any(word in query_lower for word in ['entry', 'entering', 'bore entry']):
            return DrillingWorkflowStep.BORE_ENTRY
        elif any(word in query_lower for word in ['steering', 'correction', 'adjust']):
            return DrillingWorkflowStep.STEERING_CORRECTIONS
        elif any(word in query_lower for word in ['pullback', 'pulling', 'install']):
            return DrillingWorkflowStep.PULLBACK_PREP
        elif 'complete' in query_lower or 'finish' in query_lower:
            return DrillingWorkflowStep.COMPLETION
        else:
            # Default to active drilling for any other drilling query
            return DrillingWorkflowStep.ACTIVE_DRILLING
    
    async def _generate_workflow_response(
        self, 
        query: str, 
        step: DrillingWorkflowStep, 
        session: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Generate workflow-appropriate response"""
        
        workflow_responses = {
            DrillingWorkflowStep.INITIAL_SETUP: self._handle_initial_setup,
            DrillingWorkflowStep.BORE_ENTRY: self._handle_bore_entry,
            DrillingWorkflowStep.ACTIVE_DRILLING: self._handle_active_drilling,
            DrillingWorkflowStep.STEERING_CORRECTIONS: self._handle_steering_corrections,
            DrillingWorkflowStep.PULLBACK_PREP: self._handle_pullback_prep,
            DrillingWorkflowStep.PULLBACK_EXECUTION: self._handle_pullback_execution,
            DrillingWorkflowStep.COMPLETION: self._handle_completion
        }
        
        handler = workflow_responses.get(step, self._handle_active_drilling)
        return await handler(query, session, context)
    
    async def _handle_initial_setup(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle initial drilling setup workflow"""
        
        # Extract setup parameters from query
        extracted_params = self._extract_drilling_parameters(query)
        
        # Determine what information we still need
        required_params = ['station', 'depth_target', 'entry_angle', 'mud_type']
        missing_params = [p for p in required_params if p not in extracted_params]
        
        if missing_params:
            # Ask for missing information
            next_questions = self._generate_setup_questions(missing_params)
            
            response_content = f"""# Drilling Setup - Information Needed

I've recorded your initial setup information. To complete the drilling plan, I need a few more details:

{chr(10).join([f"• {q}" for q in next_questions[:2]])}

**Current Parameters:**
{self._format_parameters(extracted_params)}

**What to expect next:** Once we have all setup parameters, I'll walk you through the bore entry checklist."""
        else:
            # All parameters collected, move to bore entry
            response_content = f"""# Setup Complete - Ready for Bore Entry

All drilling parameters recorded successfully:

{self._format_parameters(extracted_params)}

**Next Steps:**
1. Position drill rig at station {extracted_params.get('station', 'TBD')}
2. Set entry angle to {extracted_params.get('entry_angle', 'TBD')}°
3. Prepare {extracted_params.get('mud_type', 'bentonite')} mud system

**Ready to begin bore entry?** Let me know when you're positioned and I'll guide you through the entry sequence."""
        
        return {
            "status": "success",
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_SETUP", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_SETUP",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "initial_setup",
                "extracted_parameters": extracted_params,
                "missing_parameters": missing_params,
                "next_questions": next_questions if missing_params else [],
                "workflow_progress": {
                    "current_step": "initial_setup",
                    "completion_percentage": 25,
                    "next_step": "bore_entry" if not missing_params else "initial_setup"
                }
            }
        }
    
    async def _handle_bore_entry(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle bore entry workflow"""
        
        response_content = """# Bore Entry Sequence

## Pre-Entry Checklist
✓ Verify drill head position at entry point
✓ Confirm entry angle setting
✓ Test mud circulation system
✓ Check locating equipment signal

## Entry Procedure
1. **Initial Penetration**: Start at 50% speed, monitor torque
2. **Depth Monitoring**: Call out depths every 10 feet
3. **Steering Verification**: Confirm on-grade at 50 feet

**Current Status**: Ready for initial penetration
**Next Report**: Depth reading at 10 feet

What's your current depth and any observations?"""
        
        return {
            "status": "success",
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_ENTRY", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_ENTRY",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "bore_entry",
                "workflow_progress": {
                    "current_step": "bore_entry",
                    "completion_percentage": 40,
                    "next_step": "active_drilling"
                }
            }
        }
    
    async def _handle_active_drilling(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle active drilling workflow - THE KEY FUNCTION FOR LOGGING"""
        
        # Extract current drilling data
        drilling_data = self._extract_drilling_parameters(query)
        
        # Check if this is a general drilling logging request
        if any(word in query.lower() for word in ['log', 'logging', 'record', 'document']) and \
           not drilling_data:
            # Return the drilling logging interface
            response_content = """# Drill Logging Assistant Ready

I'm ready to document your drilling progress in real-time.

**Quick Start Examples:**
- "Starting bore at station 15+50, entry angle 12 degrees"
- "At 50 feet depth, pressure 175 psi, good penetration"
- "Hit clay layer at 120 feet, need steering correction"

Just describe what's happening and I'll create the proper documentation format."""
            
            return {
                "status": "success",
                "answer": response_content,
                "classification": {"category": "DRILL_LOGGING_CONVERSATIONAL", "confidence": 1.0},
                "sources": [],
                "metadata": {
                    "category": "DRILL_LOGGING_CONVERSATIONAL",
                    "query_type": "conversational_logging",
                    "render_type": "drill_logging_interface",  # KEY: This triggers the custom UI
                    "workflow_step": "active_drilling",
                    "drilling_context": {
                        "mode": "logging_ready",
                        "session_active": True,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                }
            }
        
        # If drilling data was provided, process it as a log entry
        if drilling_data:
            # Generate log entry response
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Create confirmation message
            confirmation_parts = [f"**{timestamp}** - Log entry recorded:"]
            
            if 'station' in drilling_data:
                confirmation_parts.append(f"📍 Station: {drilling_data['station']}")
            
            if 'depth' in drilling_data:
                confirmation_parts.append(f"📏 Depth: {drilling_data['depth']} {drilling_data.get('depth_unit', 'ft')}")
            
            if 'angle' in drilling_data:
                confirmation_parts.append(f"📐 Angle: {drilling_data['angle']}°")
            
            if 'pressure' in drilling_data:
                confirmation_parts.append(f"⚡ Pressure: {drilling_data['pressure']} {drilling_data.get('pressure_unit', 'psi')}")
            
            if 'flow_rate' in drilling_data:
                confirmation_parts.append(f"💧 Flow: {drilling_data['flow_rate']} {drilling_data.get('flow_unit', 'gpm')}")
            
            if 'mud_type' in drilling_data:
                confirmation_parts.append(f"🏺 Mud: {drilling_data['mud_type']}")
            
            if 'soil_conditions' in drilling_data:
                confirmation_parts.append(f"🌍 Soil: {', '.join(drilling_data['soil_conditions'])}")
            
            # Generate follow-up questions
            follow_up = self._generate_follow_up_questions(drilling_data)
            
            confirmation_message = "\n".join(confirmation_parts)
            if follow_up:
                confirmation_message += f"\n\n**Need to capture:** {follow_up}"
            
            return {
                "status": "success",
                "answer": confirmation_message,
                "classification": {"category": "DRILL_LOGGING_ENTRY", "confidence": 1.0},
                "sources": [],
                "metadata": {
                    "category": "DRILL_LOGGING_ENTRY",
                    "query_type": "conversational_logging",
                    "render_type": "drill_logging_entry",  # Different render type for entries
                    "workflow_step": "active_drilling",
                    "log_entry": {
                        "timestamp": timestamp,
                        "operator_notes": query,
                        "extracted_parameters": drilling_data,
                        "entry_type": drilling_data.get('operation', 'general_observation')
                    },
                    "extracted_data": drilling_data,
                    "drilling_context": {
                        "session_active": True,
                        "entry_count": 1,  # In production, track actual count
                        "timestamp": timestamp
                    }
                }
            }
        
        # Default active drilling response if no specific data
        response_content = f"""# Active Drilling Monitor

I'm ready to log your drilling progress. Please provide:
- Current depth reading
- Pressure and flow readings
- Any steering corrections needed
- Soil conditions encountered

**Example**: "At 120 feet, pressure 180 psi, hitting clay layer, no steering needed" """
        
        return {
            "status": "success", 
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_ACTIVE", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_ACTIVE",
                "query_type": "workflow",
                "render_type": "drill_workflow_step", 
                "workflow_step": "active_drilling",
                "drilling_data": drilling_data,
                "workflow_progress": {
                    "current_step": "active_drilling",
                    "completion_percentage": 60,
                    "next_step": "pullback_prep"
                }
            }
        }
    
    async def _handle_steering_corrections(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle steering correction workflow"""
        
        response_content = """# Steering Correction Protocol

## Correction Applied
✓ Steering adjustment logged
✓ New heading confirmed
✓ Locating signal verified

## Monitoring Requirements
- Verify correction effectiveness at next 25-foot interval
- Monitor for overcorrection tendency
- Document final grade achievement

**Status**: Correction applied, continuing bore
**Next Check**: Verify on-grade at next depth milestone

Continue drilling and report position at next checkpoint."""
        
        return {
            "status": "success",
            "answer": response_content, 
            "classification": {"category": "DRILL_WORKFLOW_STEERING", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_STEERING",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "steering_corrections",
                "workflow_progress": {
                    "current_step": "steering_corrections",
                    "completion_percentage": 70,
                    "next_step": "active_drilling"
                }
            }
        }
    
    async def _handle_pullback_prep(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle pullback preparation"""
        
        response_content = """# Pullback Preparation Checklist

## Pre-Pullback Requirements
✓ Bore completion confirmed at exit point
✓ Product pipe/cable staged and ready
✓ Pullback equipment positioned
✓ Communication established between entry and exit

## Setup Verification
1. **Reamer Selection**: Verify proper reamer size for product
2. **Pulling Tensions**: Calculate maximum allowable tension
3. **Mud System**: Switch to pullback mud formulation
4. **Product Protection**: Install protective sleeves as needed

**Status**: Ready for pullback setup
**Next Step**: Confirm product attachment and begin pullback

Ready to proceed with pullback operation?"""
        
        return {
            "status": "success",
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_PULLBACK_PREP", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_PULLBACK_PREP",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "pullback_prep",
                "workflow_progress": {
                    "current_step": "pullback_prep",
                    "completion_percentage": 80,
                    "next_step": "pullback_execution"
                }
            }
        }
    
    async def _handle_pullback_execution(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle pullback execution"""
        
        response_content = """# Pullback Execution

## Active Pullback Status
🔄 **Status**: Pulling product through bore
📏 **Progress**: Monitor footage pulled
⚡ **Tension**: Stay within calculated limits
💧 **Mud Flow**: Maintain proper circulation

## Critical Monitoring Points
- Tension readings every 50 feet
- Product integrity at entry point
- Mud returns quality
- Any resistance or binding

**Current Operation**: Active pullback in progress
**Next Update**: Report tension and footage pulled

Continue monitoring and report any changes."""
        
        return {
            "status": "success",
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_PULLBACK", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_PULLBACK",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "pullback_execution",
                "workflow_progress": {
                    "current_step": "pullback_execution",
                    "completion_percentage": 90,
                    "next_step": "completion"
                }
            }
        }
    
    async def _handle_completion(self, query: str, session: Dict, context: Dict) -> Dict[str, Any]:
        """Handle drilling completion"""
        
        response_content = """# Drilling Operation Complete

## Final Status
✅ **Product Installation**: Successfully completed
✅ **Bore Abandonment**: Entry and exit points secured
✅ **Equipment Cleanup**: Rig demobilization underway
✅ **Documentation**: Final reports being prepared

## Next Steps
1. **Quality Verification**: Test product functionality
2. **Site Restoration**: Restore surface conditions
3. **Final Documentation**: Complete as-built drawings
4. **Project Closeout**: Submit final paperwork

**Operation Status**: COMPLETE
**Total Time**: [Calculate from session start time]

Congratulations on successful completion of the drilling operation!"""
        
        return {
            "status": "success",
            "answer": response_content,
            "classification": {"category": "DRILL_WORKFLOW_COMPLETE", "confidence": 1.0},
            "sources": [],
            "metadata": {
                "category": "DRILL_WORKFLOW_COMPLETE",
                "query_type": "workflow",
                "render_type": "drill_workflow_step",
                "workflow_step": "completion",
                "workflow_progress": {
                    "current_step": "completion",
                    "completion_percentage": 100,
                    "next_step": None
                }
            }
        }
    
    def _extract_drilling_parameters(self, text: str) -> Dict[str, Any]:
        """Extract drilling parameters from natural language"""
        
        data = {}
        text_lower = text.lower()
        
        # Station patterns
        station_match = re.search(r'station\s*(\d+\+\d+|\d+)', text_lower)
        if station_match:
            data['station'] = station_match.group(1)
        
        # Depth patterns
        depth_match = re.search(r'(\d+\.?\d*)\s*(feet|ft|foot)', text_lower)
        if depth_match:
            data['depth'] = float(depth_match.group(1))
            data['depth_unit'] = 'feet'
        
        # Angle patterns
        angle_match = re.search(r'(\d+\.?\d*)\s*degree', text_lower)
        if angle_match:
            data['entry_angle'] = float(angle_match.group(1))
            data['angle'] = float(angle_match.group(1))  # Both keys for compatibility
        
        # Pressure patterns
        pressure_match = re.search(r'(\d+\.?\d*)\s*(psi|pounds)', text_lower)
        if pressure_match:
            data['pressure'] = float(pressure_match.group(1))
            data['pressure_unit'] = 'psi'
        
        # Flow patterns
        flow_match = re.search(r'(\d+\.?\d*)\s*(gpm|gallons)', text_lower)
        if flow_match:
            data['flow_rate'] = float(flow_match.group(1))
            data['flow_unit'] = 'gpm'
        
        # Mud type
        if 'bentonite' in text_lower:
            data['mud_type'] = 'bentonite'
        elif 'polymer' in text_lower:
            data['mud_type'] = 'polymer'
        
        # Soil conditions
        soil_conditions = []
        if 'clay' in text_lower:
            soil_conditions.append('clay')
        if 'sand' in text_lower:
            soil_conditions.append('sand')
        if 'rock' in text_lower:
            soil_conditions.append('rock')
        if 'water' in text_lower:
            soil_conditions.append('groundwater')
        
        if soil_conditions:
            data['soil_conditions'] = soil_conditions
        
        # Operation type
        if any(word in text_lower for word in ['starting', 'begin', 'entry']):
            data['operation'] = 'entry'
        elif any(word in text_lower for word in ['pullback', 'pulling', 'install']):
            data['operation'] = 'pullback'
        elif any(word in text_lower for word in ['steering', 'correction', 'adjust']):
            data['operation'] = 'steering_correction'
        
        return data
    
    def _generate_setup_questions(self, missing_params: List[str]) -> List[str]:
        """Generate questions for missing setup parameters"""
        
        question_map = {
            'station': "What station are you starting the bore at?",
            'depth_target': "What's the target depth for this bore?", 
            'entry_angle': "What entry angle are you using?",
            'mud_type': "What type of drilling mud will you be using?",
            'bore_diameter': "What diameter bore are you drilling?",
            'pipe_material': "What type of pipe/cable are you installing?"
        }
        
        return [question_map.get(param, f"Please specify {param}") for param in missing_params]
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """Format parameters for display"""
        
        if not params:
            return "None recorded yet"
        
        formatted = []
        for key, value in params.items():
            if key == 'station':
                formatted.append(f"• **Station**: {value}")
            elif key == 'depth':
                unit = params.get('depth_unit', 'ft')
                formatted.append(f"• **Depth**: {value} {unit}")
            elif key in ['entry_angle', 'angle']:
                formatted.append(f"• **Entry Angle**: {value}°")
            elif key == 'mud_type':
                formatted.append(f"• **Mud Type**: {value.title()}")
            elif key == 'pressure':
                unit = params.get('pressure_unit', 'psi')
                formatted.append(f"• **Pressure**: {value} {unit}")
            elif key == 'flow_rate':
                unit = params.get('flow_unit', 'gpm')
                formatted.append(f"• **Flow Rate**: {value} {unit}")
        
        return '\n'.join(formatted)
    
    def _generate_follow_up_questions(self, extracted_data: Dict) -> str:
        """Generate helpful follow-up questions based on what's missing"""
        
        missing_items = []
        
        if 'depth' not in extracted_data:
            missing_items.append("current depth")
        
        if 'pressure' not in extracted_data and extracted_data.get('operation') != 'pullback':
            missing_items.append("mud pressure")
        
        if 'flow_rate' not in extracted_data and extracted_data.get('operation') != 'pullback':
            missing_items.append("flow rate")
        
        if extracted_data.get('operation') == 'pullback' and 'tension' not in extracted_data:
            missing_items.append("pulling tension")
        
        if missing_items:
            return f"Any {', '.join(missing_items)}?"
        
        return ""
    
    async def _update_workflow_session(
        self, 
        conversation_id: str, 
        session: Dict, 
        current_step: DrillingWorkflowStep
    ):
        """Update workflow session in database"""
        
        session['current_step'] = current_step.value
        session['last_updated'] = datetime.now().isoformat()
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO drilling_sessions (conversation_id, workflow_data, updated_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (conversation_id) 
                DO UPDATE SET workflow_data = $2, updated_at = $3
            """, conversation_id, json.dumps(session), datetime.now())

# Integration function for bubble_backend.py
async def process_drilling_workflow(
    request,
    conversation_context: Optional[Dict] = None,
    pool=None
) -> Optional[Dict[str, Any]]:
    """Process drilling workflow queries"""
    
    if not pool:
        return None
    
    # ENSURE TABLE EXISTS BEFORE PROCESSING
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS drilling_sessions (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL UNIQUE,
                    workflow_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    except Exception as e:
        print(f"❌ Error ensuring drilling_sessions table: {e}")
        return None
    
    handler = DrillingWorkflowHandler(pool)
    
    # Check if this is a drilling workflow query
    result = await handler.handle_drilling_workflow(
        query=request.query,
        conversation_id=request.conversation_id,
        context=request.context.dict() if hasattr(request.context, 'dict') else {}
    )
    
    return result
