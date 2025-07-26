from typing import List, Optional
from pydantic import BaseModel, Field


class DiagramRequest(BaseModel):
    prompt: str
    code_style: str = "graphviz"
    output_format: str = "pdf"

class WorksheetRequest(BaseModel):
    # Base64 encoded images
    images: List[str] = Field(..., description="List of base64 encoded images")
    grade_level: str = Field("5", description="Grade level for the worksheet")
    subject: str = Field("General", description="Subject of the worksheet")
    difficulty: str = Field("medium", description="Difficulty level (easy, medium, hard)")
    question_count: int = Field(10, description="Total number of questions")
    mcq_count: int = Field(4, description="Number of multiple choice questions")
    short_answer_count: int = Field(3, description="Number of short answer questions")
    fill_blank_count: int = Field(2, description="Number of fill-in-the-blank questions")
    true_false_count: int = Field(1, description="Number of true/false questions")

class ChatbotRequest(BaseModel):
    message: str = Field(..., description="The user's message or question")
    syllabus: str = Field("General", description="The syllabus context (e.g., 'Math', 'Science', 'History')")
    grade_level: str = Field("5", description="Target grade level for the response")
    
class ChatbotSessionRequest(BaseModel):
    user_id: str = Field(..., description="The user's ID for session management")
    
class ChatbotMessageRequest(BaseModel):
    user_id: str = Field(..., description="The user's ID for session management")
    session_id: str = Field(..., description="The session ID for context tracking")
    message: str = Field(..., description="The user's message or question")
    syllabus: str = Field("General", description="The syllabus context (e.g., 'Math', 'Science', 'History')")
    grade_level: str = Field("5", description="Target grade level for the response")

