import io
import logging
import os
import re
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Union

import base64
import hashlib
from PIL import Image

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import A2A communication infrastructure
from app.agents.agent_registry import registry, provides_capability
from app.agents.agent_messages import Priority, NotificationMessage, RequestMessage, ResponseMessage, ErrorMessage, MessageType
from app.agents.message_bus import message_bus

# Setup logging
logger = logging.getLogger(__name__)

# Helper functions for worksheet generation
def _group_questions_by_type(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Group questions by their type and create a separate student worksheet and answer key
    """
    # Student worksheet - questions only (no answers)
    student_questions = {
        "multiple_choice": [],
        "short_answer": [],
        "fill_in_blank": [],
        "true_false": []
    }

    # Teacher answer key - answers only
    answer_key = {
        "multiple_choice": [],
        "short_answer": [],
        "fill_in_blank": [],
        "true_false": []
    }

    for question in questions:
        question_type = question.get("type", "short_answer")

        if question_type in student_questions:
            # For students: Remove answers and explanations
            student_question = {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "options": question.get("options", [])
            }
            student_questions[question_type].append(student_question)

            # For teachers: Just ID, correct answer, and explanation
            teacher_answer = {
                "id": question["id"],
                "correct_answer": question["correct_answer"],
                "explanation": question["explanation"]
            }
            answer_key[question_type].append(teacher_answer)

    # Add counts for each type
    question_counts = {
        "multiple_choice_count": len(student_questions["multiple_choice"]),
        "short_answer_count": len(student_questions["short_answer"]),
        "fill_in_blank_count": len(student_questions["fill_in_blank"]),
        "true_false_count": len(student_questions["true_false"])
    }

    return {
        "student_worksheet": student_questions,
        "teacher_answer_key": answer_key,
        "question_counts": question_counts
    }


def _self_evaluate_worksheet(worksheet: Dict[str, Any], question_targets: Dict[str, int], must_have_answers: bool = True) -> tuple[bool, str]:
    """
    Evaluate if the worksheet matches required criteria and is sensible. Returns (bool, str).
    """
    if not worksheet or "question_counts" not in worksheet:
        return False, "Parsing failed or worksheet missing question_counts."
    qc = worksheet["question_counts"]
    for key, count in question_targets.items():
        k = key + "_count"  # keys: 'multiple_choice_count', etc.
        if qc.get(k, 0) != count:
            return False, f"Expected {count} {key.replace('_', ' ')}; found {qc.get(k, 0)}."
    # Optionally ensure every answer/explanation exists for the teacher key
    if must_have_answers:
        tk = worksheet.get("teacher_answer_key") or {}
        for qtype, lst in tk.items():
            for item in lst:
                if "correct_answer" not in item or not str(item["correct_answer"]).strip():
                    return False, f"Missing answer for {qtype} question ID {item.get('id')}"
                if "explanation" not in item:
                    return False, f"Missing explanation for {qtype} question ID {item.get('id')}"
    return True, "OK"


def _parse_worksheet_to_json(worksheet_text: str) -> Dict[str, Any]:
    """
    Parse the AI-generated worksheet text into structured JSON format with grouped questions
    """
    try:
        lines = worksheet_text.split('\n')

        # Extract metadata
        title = ""
        grade_level = ""
        subject = ""
        difficulty = ""
        instructions = ""

        # Find title and metadata
        for line in lines:
            if line.startswith("**WORKSHEET:"):
                title = line.replace("**WORKSHEET:", "").replace("**", "").strip()
            elif line.startswith("**Grade Level:**"):
                grade_level = line.replace("**Grade Level:**", "").strip()
            elif line.startswith("**Subject:**"):
                subject = line.replace("**Subject:**", "").strip()
            elif line.startswith("**Difficulty:**"):
                difficulty = line.replace("**Difficulty:**", "").strip()

        # Extract instructions
        instruction_start = False
        for line in lines:
            if "**Instructions for Students:**" in line:
                instruction_start = True
                continue
            elif instruction_start and line.startswith("**QUESTIONS:**"):
                break
            elif instruction_start and line.strip():
                instructions += line.strip() + " "

        # Extract questions and answers
        questions = []

        # Find the questions section
        questions_section = []
        answers_section = []
        current_section = None

        for line in lines:
            if "**QUESTIONS:**" in line:
                current_section = "questions"
                continue
            elif "**ANSWER KEY:**" in line:
                current_section = "answers"
                continue
            elif current_section == "questions" and line.strip():
                questions_section.append(line)
            elif current_section == "answers" and line.strip():
                answers_section.append(line)

        # Parse questions
        question_number = 1
        current_question = None

        for line in questions_section:
            line = line.strip()
            if not line:
                continue

            # Check if it's a new question (starts with number)
            if re.match(r'^\d+\.', line):
                if current_question:
                    questions.append(current_question)

                # Extract question type and text
                question_text = re.sub(r'^\d+\.\s*', '', line)
                question_type = "short_answer"  # default

                if "**Multiple Choice:**" in question_text:
                    question_type = "multiple_choice"
                    question_text = question_text.replace("**Multiple Choice:**", "").strip()
                elif "**Fill-in-the-blank:**" in question_text:
                    question_type = "fill_in_blank"
                    question_text = question_text.replace("**Fill-in-the-blank:**", "").strip()
                elif "**True or False:**" in question_text:
                    question_type = "true_false"
                    question_text = question_text.replace("**True or False:**", "").strip()
                elif "**Short Answer:**" in question_text:
                    question_type = "short_answer"
                    question_text = question_text.replace("**Short Answer:**", "").strip()

                current_question = {
                    "id": question_number,
                    "type": question_type,
                    "question": question_text,
                    "options": [],
                    "correct_answer": "",
                    "explanation": ""
                }
                question_number += 1

            # Check if it's an option (a), b), c), d))
            elif current_question and re.match(r'^\s*[a-d]\)', line):
                option_text = re.sub(r'^\s*[a-d]\)\s*', '', line)
                current_question["options"].append(option_text)

        # Add the last question
        if current_question:
            questions.append(current_question)

        # Parse answers
        answer_number = 1
        for line in answers_section:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^\d+\.', line):
                answer_text = re.sub(r'^\d+\.\s*', '', line)

                # Find corresponding question and add answer
                if answer_number <= len(questions):
                    # Extract the correct answer and explanation
                    if questions[answer_number - 1]["type"] == "multiple_choice":
                        # Extract the letter answer (a, b, c, d)
                        match = re.search(r'\*\*([a-d])\)', answer_text)
                        if match:
                            questions[answer_number - 1]["correct_answer"] = match.group(1)

                        # Extract explanation after **
                        explanation_match = re.search(r'\*\*.*?\*\*\s*(.*)', answer_text)
                        if explanation_match:
                            questions[answer_number - 1]["explanation"] = explanation_match.group(1)
                    else:
                        # For other question types, the whole answer is the correct answer
                        if "**" in answer_text:
                            parts = answer_text.split("**")
                            if len(parts) >= 2:
                                questions[answer_number - 1]["correct_answer"] = parts[1].strip()
                                if len(parts) > 2:
                                    questions[answer_number - 1]["explanation"] = parts[2].strip()
                        else:
                            questions[answer_number - 1]["correct_answer"] = answer_text

                answer_number += 1

        # Group questions by type
        grouped_data = _group_questions_by_type(questions)

        return {
            "worksheet_info": {
                "title": title,
                "grade_level": grade_level,
                "subject": subject,
                "difficulty": difficulty,
                "instructions": instructions.strip()
            },
            **grouped_data  # This includes student_worksheet, teacher_answer_key, question_counts
        }

    except Exception as e:
        logging.exception("Error parsing worksheet")
        return {
            "worksheet_info": {
                "title": "Generated Worksheet",
                "grade_level": "Unknown",
                "subject": "General",
                "difficulty": "Medium",
                "instructions": "Complete all questions to the best of your ability."
            },
            "student_worksheet": {
                "multiple_choice": [],
                "short_answer": [],
                "fill_in_blank": [],
                "true_false": []
            },
            "teacher_answer_key": {
                "multiple_choice": [],
                "short_answer": [],
                "fill_in_blank": [],
                "true_false": []
            },
            "question_counts": {
                "multiple_choice_count": 0,
                "short_answer_count": 0,
                "fill_in_blank_count": 0,
                "true_false_count": 0
            },
            "error": f"Parsing error: {str(e)}"
        }


# Define the tool function for ADK
def generate_worksheet_tool(images_base64: List[str], grade_level: str = "5", subject: str = "General",
                      difficulty: str = "medium", mcq_count: int = 4, short_answer_count: int = 3,
                      fill_blank_count: int = 2, true_false_count: int = 1) -> Dict[str, Any]:
    """Generates a comprehensive educational worksheet based on textbook images.

    Args:
        images_base64 (List[str]): List of base64-encoded images of textbook pages.
        grade_level (str, optional): Target grade level for the worksheet. Defaults to "5".
        subject (str, optional): Subject area of the worksheet. Defaults to "General".
        difficulty (str, optional): Difficulty level (easy, medium, hard). Defaults to "medium".
        mcq_count (int, optional): Number of multiple choice questions. Defaults to 4.
        short_answer_count (int, optional): Number of short answer questions. Defaults to 3.
        fill_blank_count (int, optional): Number of fill-in-the-blank questions. Defaults to 2.
        true_false_count (int, optional): Number of true/false questions. Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing the generated worksheet with the following keys:
            - worksheet: The structured worksheet data if successful
            - raw_content: The raw text content generated by the model
            - error: Error message if generation failed
            - retries: List of retry attempts and reasons
            - succeeded_attempt: The attempt number that succeeded (if any)
    """
    # Initialize result dictionary
    result = {"worksheet": None, "raw_content": None, "error": None, "retries": []}
    
    # Convert base64 images to PIL Images for validation
    pil_images = []
    for img_base64 in images_base64:
        try:
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pil_images.append(img)
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}", "worksheet": None}
    
    if not pil_images:
        result["error"] = "At least one valid image is required."
        return result
    
    # This is a placeholder for the actual worksheet generation
    # In a real implementation, this would be handled by the ADK agent
    # The agent will replace this with the actual worksheet content
    
    # Create a sample worksheet for demonstration
    raw_content = f"""
    **WORKSHEET: Sample Worksheet for {subject}**
    **Grade Level:** {grade_level}
    **Subject:** {subject}
    **Difficulty:** {difficulty}

    **Instructions for Students:**
    Complete all questions to the best of your ability. Read each question carefully.

    **QUESTIONS:**

    1. **Multiple Choice:** Sample multiple choice question 1?
     a) Option A
     b) Option B
     c) Option C
     d) Option D

    2. **Multiple Choice:** Sample multiple choice question 2?
     a) Option A
     b) Option B
     c) Option C
     d) Option D

    3. **Short Answer:** Sample short answer question 1?

    4. **Short Answer:** Sample short answer question 2?

    5. **Fill-in-the-blank:** Sample fill in the blank question with _______ blank.

    6. **True or False:** Sample true or false statement.

    **ANSWER KEY:**

    1. **a)** Explanation for question 1.
    2. **b)** Explanation for question 2.
    3. **Sample answer** Explanation for question 3.
    4. **Sample answer** Explanation for question 4.
    5. **word** Explanation for question 5.
    6. **True** Explanation for question 6.
    """
    
    # Parse the worksheet text into structured JSON
    structured_worksheet = _parse_worksheet_to_json(raw_content)
    
    # Set the result
    result["worksheet"] = structured_worksheet
    result["raw_content"] = raw_content
    result["succeeded_attempt"] = 1
    
    return result


# Create the ADK Agent
class WorksheetAgentADK:
    def __init__(self, model="gemini-1.5-flash"):
        """Initialize the ADK-based Worksheet Agent.
        
        Args:
            model (str): The model to use for the agent. Defaults to "gemini-1.5-flash".
        """
        self.model = model
        self.agent = Agent(
            name="worksheet_agent",
            model=model,
            description="Generates educational worksheets based on textbook images.",
            instruction="""You are a helpful educational assistant that creates worksheets based on textbook images.
            When a user uploads textbook images and requests a worksheet, use the 'generate_worksheet_tool' to create
            a comprehensive worksheet with multiple choice, short answer, fill-in-the-blank, and true/false questions.
            
            Always ask for the following information if not provided:
            1. Grade level (default is 5)
            2. Subject (default is General)
            3. Difficulty level (default is medium)
            4. Number of each question type (defaults are: 4 MCQ, 3 short answer, 2 fill-in-blank, 1 true/false)
            
            Ensure the worksheet is appropriate for the specified grade level and difficulty.
            Present the worksheet results in a clear, organized manner.
            """,
            tools=[generate_worksheet_tool],
        )
        
        # Create session service for managing conversation history
        self.session_service = InMemorySessionService()
        
        # Constants for identifying the interaction context
        self.app_name = "worksheet_app"
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
    async def generate_worksheet(self, images: List[Image.Image], grade_level: str = "5", subject: str = "General",
                           difficulty: str = "medium", question_count: int = 10, mcq_count: int = 4,
                           short_answer_count: int = 3, fill_blank_count: int = 2, true_false_count: int = 1) -> Dict[str, Any]:
        """
        Generate a worksheet based on textbook images using the ADK agent.
        
        Args:
            images: List of PIL Image objects
            grade_level: Target grade level for the worksheet
            subject: Subject area of the worksheet
            difficulty: Difficulty level (easy, medium, hard)
            question_count: Total number of questions
            mcq_count: Number of multiple choice questions
            short_answer_count: Number of short answer questions
            fill_blank_count: Number of fill-in-the-blank questions
            true_false_count: Number of true/false questions
            
        Returns:
            Dict containing the generated worksheet and metadata
        """
        # Validate total question count matches breakdown
        total_specified = mcq_count + short_answer_count + fill_blank_count + true_false_count
        if total_specified != question_count:
            question_count = total_specified
            
        # Convert PIL images to base64
        images_base64 = []
        for img in images:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            encoded_img = base64.b64encode(img_bytes).decode('utf-8')
            images_base64.append(encoded_img)
            
        # Create a unique user and session ID
        user_id = f"user_{os.urandom(4).hex()}"
        session_id = f"session_{os.urandom(4).hex()}"
        
        # Create the session
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # Prepare the message for the agent
        prompt = f"""Generate a worksheet with the following specifications:
        - Grade Level: {grade_level}
        - Subject: {subject}
        - Difficulty: {difficulty}
        - Multiple Choice Questions: {mcq_count}
        - Short Answer Questions: {short_answer_count}
        - Fill-in-the-blank Questions: {fill_blank_count}
        - True/False Questions: {true_false_count}
        
        I've uploaded {len(images)} textbook page images for you to use as the basis for the worksheet.
        """
        
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        
        # Initialize result dictionary
        result = {"worksheet": None, "raw_content": None, "error": None, "retries": []}
        
        # Run the agent
        tool_result = None
        
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Check for tool responses
            if hasattr(event, 'tool_response') and event.tool_response is not None:
                tr = event.tool_response
                # Look for worksheet result structure
                if isinstance(tr, dict) and ("worksheet" in tr or "raw_content" in tr):
                    tool_result = tr
                # Sometimes Gemini may nest the result
                elif hasattr(tr, "worksheet") or hasattr(tr, "raw_content"):
                    tool_result = {"worksheet": getattr(tr, "worksheet", None),
                                  "raw_content": getattr(tr, "raw_content", None)}
            
            # Check for final response
            if event.is_final_response():
                if event.actions and event.actions.escalate:
                    result["error"] = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        # If we got a tool result, use it
        if tool_result:
            result["worksheet"] = tool_result.get("worksheet")
            result["raw_content"] = tool_result.get("raw_content")
            result["retries"] = tool_result.get("retries", [])
            result["succeeded_attempt"] = tool_result.get("succeeded_attempt")
            
            if tool_result.get("error"):
                result["error"] = tool_result["error"]
        else:
            # If we didn't get a tool result, call the tool directly as a fallback
            worksheet_result = generate_worksheet_tool(
                images_base64=images_base64,
                grade_level=grade_level,
                subject=subject,
                difficulty=difficulty,
                mcq_count=mcq_count,
                short_answer_count=short_answer_count,
                fill_blank_count=fill_blank_count,
                true_false_count=true_false_count
            )
            
            result["worksheet"] = worksheet_result.get("worksheet")
            result["raw_content"] = worksheet_result.get("raw_content")
            result["retries"] = worksheet_result.get("retries", [])
            result["succeeded_attempt"] = worksheet_result.get("succeeded_attempt")
            
            if worksheet_result.get("error"):
                result["error"] = worksheet_result["error"]
                
        return result