from fastapi import APIRouter, HTTPException
from app.models import DiagramRequest
from app.agents.diagram_agent import DiagramAgent

router = APIRouter()

diagram_agent = DiagramAgent()

@router.post("/generate")
def generate_diagram(request: DiagramRequest):
    result = diagram_agent.generate_diagram(
        user_prompt=request.prompt,
        code_style=request.code_style,
        output_format=request.output_format
    )
    # Always return the full agentic result, including retries, succeeded_attempt, etc.
    return result
