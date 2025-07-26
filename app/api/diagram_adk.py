from fastapi import APIRouter, HTTPException
import asyncio
from app.models import DiagramRequest
from app.agents.diagram_agent_adk import DiagramAgentADK

router = APIRouter()

# Initialize the ADK-based diagram agent
diagram_agent = DiagramAgentADK()

@router.post("/generate")
async def generate_diagram(request: DiagramRequest):
    """
    Generate a diagram based on the provided description using the ADK-based agent.
    
    Args:
        request: DiagramRequest containing prompt, code_style, and output_format
        
    Returns:
        The generated diagram data including image, code, and metadata
    """
    try:
        # Call the async generate_diagram method
        result = await diagram_agent.generate_diagram(
            user_prompt=request.prompt,
            code_style=request.code_style,
            output_format=request.output_format
        )
        
        # Always return the full agentic result, including retries, succeeded_attempt, etc.
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diagram: {str(e)}")