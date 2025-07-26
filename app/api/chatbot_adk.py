from fastapi import APIRouter, HTTPException
import asyncio
from app.models import ChatbotRequest, ChatbotSessionRequest, ChatbotMessageRequest
from app.agents.chatbot_agent_adk import ChatbotAgentADK

router = APIRouter()

# Initialize the ADK-based chatbot agent
chatbot_agent = ChatbotAgentADK()

@router.post("/create_session")
async def create_session(request: ChatbotSessionRequest):
    """
    Create a new chat session for a user.
    
    Args:
        request: ChatbotSessionRequest containing user_id
        
    Returns:
        The session ID for the new session
    """
    try:
        # Create a new session
        session_id = await chatbot_agent.create_session(user_id=request.user_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.post("/chat")
async def chat(request: ChatbotMessageRequest):
    """
    Process a chat message and generate a response.
    
    Args:
        request: ChatbotMessageRequest containing user_id, session_id, message, syllabus, and grade_level
        
    Returns:
        The response data including answer, diagram (if any), and metadata
    """
    try:
        # Process the message
        result = await chatbot_agent.chat(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
            syllabus=request.syllabus,
            grade_level=request.grade_level
        )
        
        # Add success flag to the result
        response = {
            "success": result.get("error") is None,
            **result
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/simple_chat")
async def simple_chat(request: ChatbotRequest):
    """
    Simple chat endpoint that creates a session automatically and processes a message.
    
    Args:
        request: ChatbotRequest containing message, syllabus, and grade_level
        
    Returns:
        The response data including answer, diagram (if any), and metadata
    """
    try:
        # Create a unique user ID
        import os
        user_id = f"user_{os.urandom(4).hex()}"
        
        # Create a new session
        session_id = await chatbot_agent.create_session(user_id=user_id)
        
        # Process the message
        result = await chatbot_agent.chat(
            user_id=user_id,
            session_id=session_id,
            message=request.message,
            syllabus=request.syllabus,
            grade_level=request.grade_level
        )
        
        # Add success flag and session info to the result
        response = {
            "success": result.get("error") is None,
            "user_id": user_id,
            "session_id": session_id,
            **result
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")