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
    def __init__(self, model="gemini-1.5-flash", agent_name="worksheet_agent"):
        """Initialize the ADK-based Worksheet Agent with enhanced agentic capabilities.
        
        Args:
            model (str): The model to use for the agent. Defaults to "gemini-1.5-flash".
            agent_name (str): The name to register this agent with. Defaults to "worksheet_agent".
        """
        self.model = model
        self.agent_name = agent_name
        
        # Enhanced instruction with more agentic behavior
        instruction = """You are an advanced educational worksheet generator that creates comprehensive worksheets based on textbook images.
        
        When a user uploads textbook images and requests a worksheet:
        1. Analyze the content of the images to understand the educational material
        2. Use the 'generate_worksheet_tool' to create a comprehensive worksheet
        3. Ensure questions are appropriate for the specified grade level and difficulty
        4. Evaluate the quality of generated questions and make improvements if needed
        5. Provide clear explanations for answer keys when requested
        6. Suggest additional learning resources or related topics based on the content
        7. Adapt question complexity based on the detected content difficulty
        
        Always ask for the following information if not provided:
        1. Grade level (default is 5)
        2. Subject (default is General)
        3. Difficulty level (default is medium)
        4. Number of each question type (defaults are: 4 MCQ, 3 short answer, 2 fill-in-blank, 1 true/false)
        
        Ensure the worksheet follows educational best practices:
        - Questions are clear and unambiguous
        - Difficulty progression is appropriate
        - Content aligns with curriculum standards
        - Answer keys are accurate and complete
        - Instructions are student-friendly
        
        Present the worksheet results in a clear, organized manner with proper formatting.
        """
        
        # Create the ADK agent with enhanced tools
        self.agent = Agent(
            name=agent_name,
            model=model,
            description="Advanced worksheet generation agent with agentic capabilities",
            instruction=instruction,
            tools=[generate_worksheet_tool],
        )
        
        # Create session service for managing conversation history
        self.session_service = InMemorySessionService()
        
        # Constants for identifying the interaction context
        self.app_name = "worksheet_app"
        
        # Create the runner with enhanced configuration
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        # Initialize metrics for monitoring agent performance
        self.metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "retry_count": 0,
            "average_generation_time": 0,
            "total_generation_time": 0,
            "question_types": {},     # Track frequency of different question types
            "grade_levels": {},       # Track frequency of different grade levels
            "subjects": {},           # Track frequency of different subjects
            "error_types": {}         # Track frequency of different error types
        }
        
        # Cache for storing recently generated worksheets to avoid regeneration
        self.worksheet_cache = {}
        
        # Register with the agent registry
        logger.info(f"Registering {agent_name} with the agent registry")
        registry.register_agent(agent_name, self, [
            "generate_worksheet",
            "get_supported_question_types",
            "get_worksheet_suggestions"
        ])

        # DO NOT start async tasks here! Instead, use async_init() later in the event loop
        # asyncio.create_task(self._setup_notification_handlers())

    async def async_init(self):
        """
        Run async initialization routines after construction, in a running event loop.
        """
        await self._setup_notification_handlers()

    async def _setup_notification_handlers(self):
        """
        Set up handlers for notifications from other agents.
        This enables bidirectional communication between agents.
        """
        try:
            # Register a handler for the NotificationMessage type
            message_bus.register_handler(
                self.agent_name, 
                MessageType.NOTIFICATION,
                self._handle_notification
            )
            logger.info(f"Notification handlers set up for {self.agent_name}")
        except Exception as e:
            logger.error(f"Error setting up notification handlers: {str(e)}", exc_info=True)
    
    async def _handle_notification(self, message: NotificationMessage):
        """
        Handle notifications from other agents.
        
        Args:
            message: The notification message
        """
        try:
            notification_type = message.notification_type
            sender = message.sender
            data = message.data or {}
            
            logger.info(f"Received notification from {sender}: {notification_type}")
            
            if notification_type == "question_asked":
                await self._process_question_notification(data)
            elif notification_type == "diagram_generated":
                await self._process_diagram_notification(data)
            elif notification_type == "worksheet_request":
                await self._process_worksheet_request(data)
            else:
                logger.info(f"Unknown notification type: {notification_type}")
                
        except Exception as e:
            logger.error(f"Error handling notification: {str(e)}", exc_info=True)
    
    async def _process_question_notification(self, data: Dict[str, Any]):
        """
        Process notifications about questions asked by users.
        This could trigger worksheet generation suggestions.
        
        Args:
            data: The notification data containing question details
        """
        try:
            topic = data.get("topic", "")
            grade_level = data.get("grade_level", "5")
            subject = data.get("syllabus", "General")
            
            # Check if we should suggest worksheet generation
            if topic and len(topic.split()) >= 2:  # Multi-word topics are good candidates
                logger.info(f"Considering worksheet generation for topic: {topic}")
                # Could implement logic to suggest worksheet creation
                
        except Exception as e:
            logger.error(f"Error processing question notification: {str(e)}", exc_info=True)
    
    async def _process_diagram_notification(self, data: Dict[str, Any]):
        """
        Process notifications about diagrams generated by other agents.
        This could be used to create complementary worksheets.
        
        Args:
            data: The notification data containing diagram details
        """
        try:
            topic = data.get("prompt", "")
            diagram_type = data.get("diagram_type", "")
            
            if topic and diagram_type:
                logger.info(f"Diagram generated for {topic} ({diagram_type}) - could create complementary worksheet")
                # Could implement logic to suggest complementary worksheet
                
        except Exception as e:
            logger.error(f"Error processing diagram notification: {str(e)}", exc_info=True)
    
    async def _process_worksheet_request(self, data: Dict[str, Any]):
        """
        Process direct worksheet generation requests from other agents.
        
        Args:
            data: The notification data containing worksheet request details
        """
        try:
            # This could be used for agent-to-agent worksheet generation requests
            logger.info("Received worksheet generation request from another agent")
            
        except Exception as e:
            logger.error(f"Error processing worksheet request: {str(e)}", exc_info=True)
    
    @provides_capability("get_supported_question_types")
    async def get_supported_question_types(self) -> Dict[str, Any]:
        """
        Get the supported question types for worksheet generation.
        
        Returns:
            Dict containing supported question types and their descriptions
        """
        return {
            "multiple_choice": {
                "description": "Multiple choice questions with 4 options",
                "default_count": 4,
                "max_count": 20
            },
            "short_answer": {
                "description": "Short answer questions requiring brief responses",
                "default_count": 3,
                "max_count": 15
            },
            "fill_in_blank": {
                "description": "Fill-in-the-blank questions with missing words",
                "default_count": 2,
                "max_count": 10
            },
            "true_false": {
                "description": "True/false questions",
                "default_count": 1,
                "max_count": 10
            }
        }
    
    @provides_capability("get_worksheet_suggestions")
    async def get_worksheet_suggestions(self, topic: str, grade_level: str = "5", subject: str = "General") -> Dict[str, Any]:
        """
        Get suggestions for worksheet generation based on a topic.
        
        Args:
            topic: The topic to generate suggestions for
            grade_level: The target grade level
            subject: The subject area
            
        Returns:
            Dict containing worksheet suggestions
        """
        try:
            # Generate suggestions based on topic analysis
            suggestions = {
                "recommended_question_types": [],
                "difficulty_level": "medium",
                "estimated_time": "30-45 minutes",
                "related_topics": [],
                "learning_objectives": []
            }
            
            # Simple topic analysis for suggestions
            topic_lower = topic.lower()
            
            # Recommend question types based on topic
            if any(word in topic_lower for word in ["math", "calculation", "equation", "formula"]):
                suggestions["recommended_question_types"] = ["short_answer", "fill_in_blank", "multiple_choice"]
                suggestions["difficulty_level"] = "medium"
            elif any(word in topic_lower for word in ["history", "event", "date", "person"]):
                suggestions["recommended_question_types"] = ["multiple_choice", "short_answer", "true_false"]
            elif any(word in topic_lower for word in ["science", "experiment", "hypothesis", "theory"]):
                suggestions["recommended_question_types"] = ["multiple_choice", "short_answer", "fill_in_blank"]
            else:
                suggestions["recommended_question_types"] = ["multiple_choice", "short_answer", "true_false"]
            
            # Adjust difficulty based on grade level
            grade_num = int(grade_level) if grade_level.isdigit() else 5
            if grade_num <= 3:
                suggestions["difficulty_level"] = "easy"
                suggestions["estimated_time"] = "20-30 minutes"
            elif grade_num >= 8:
                suggestions["difficulty_level"] = "hard"
                suggestions["estimated_time"] = "45-60 minutes"
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating worksheet suggestions: {str(e)}", exc_info=True)
            return {
                "recommended_question_types": ["multiple_choice", "short_answer"],
                "difficulty_level": "medium",
                "estimated_time": "30-45 minutes",
                "related_topics": [],
                "learning_objectives": []
            }
    
    @provides_capability("generate_worksheet")
    async def generate_worksheet(self, images: List[Image.Image], grade_level: str = "5", subject: str = "General",
                           difficulty: str = "medium", question_count: int = 10, mcq_count: int = 4,
                           short_answer_count: int = 3, fill_blank_count: int = 2, true_false_count: int = 1,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a worksheet based on textbook images using the ADK agent with enhanced agentic capabilities.
        
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
            context: Additional context for worksheet generation
            
        Returns:
            Dict containing the generated worksheet and enhanced metadata
        """
        # Start timing for performance metrics
        start_time = time.time()
        
        # Update metrics
        self.metrics["total_requests"] += 1
        
        # Initialize enhanced result dictionary
        result = {
            "worksheet": None,
            "raw_content": None,
            "error": None,
            "retries": [],
            "succeeded_attempt": None,
            "processing_time": None,
            "confidence_score": None,
            "question_analysis": {},
            "suggestions": []
        }
        
        try:
            # Validate total question count matches breakdown
            total_specified = mcq_count + short_answer_count + fill_blank_count + true_false_count
            if total_specified != question_count:
                question_count = total_specified
            
            # Create cache key for potential reuse
            image_hashes = []
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                image_hashes.append(img_hash)
            
            cache_key = f"{'-'.join(image_hashes)}_{grade_level}_{subject}_{difficulty}_{mcq_count}_{short_answer_count}_{fill_blank_count}_{true_false_count}"
            
            # Check cache first
            if cache_key in self.worksheet_cache:
                cached_result = self.worksheet_cache[cache_key]
                logger.info(f"Returning cached worksheet for key: {cache_key[:20]}...")
                cached_result["processing_time"] = time.time() - start_time
                return cached_result
            
            # Convert PIL images to base64
            images_base64 = []
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                encoded_img = base64.b64encode(img_bytes).decode('utf-8')
                images_base64.append(encoded_img)
            
            # Update metrics tracking
            self.metrics["grade_levels"][grade_level] = self.metrics["grade_levels"].get(grade_level, 0) + 1
            self.metrics["subjects"][subject] = self.metrics["subjects"].get(subject, 0) + 1
            
            # Create a unique user and session ID
            user_id = f"user_{os.urandom(4).hex()}"
            session_id = f"session_{os.urandom(4).hex()}"
            
            # Create the session
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            # Enhanced prompt with more context
            enhanced_context = context or {}
            prompt = f"""Generate a comprehensive educational worksheet with the following specifications:
            
            WORKSHEET REQUIREMENTS:
            - Grade Level: {grade_level}
            - Subject: {subject}
            - Difficulty: {difficulty}
            - Multiple Choice Questions: {mcq_count}
            - Short Answer Questions: {short_answer_count}
            - Fill-in-the-blank Questions: {fill_blank_count}
            - True/False Questions: {true_false_count}
            
            CONTENT SOURCE:
            I've uploaded {len(images)} textbook page images for you to use as the basis for the worksheet.
            
            ADDITIONAL CONTEXT:
            {enhanced_context.get('learning_objectives', 'Focus on key concepts and practical application.')}
            
            QUALITY REQUIREMENTS:
            - Ensure questions are age-appropriate for grade {grade_level}
            - Include clear instructions for each question type
            - Provide accurate answer keys
            - Make questions engaging and educational
            - Ensure proper difficulty progression
            
            Please analyze the images thoroughly and create questions that test understanding of the key concepts presented.
            """
            
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            # Track if we need to retry due to errors
            retry_count = 0
            max_retries = 2
            tool_result = None
            
            while retry_count <= max_retries:
                try:
                    # Run the agent
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
                    
                    # If we got a tool result, break the retry loop
                    if tool_result:
                        break
                    
                    # If no result and we haven't hit max retries, try again
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"No worksheet result from agent, retrying ({retry_count}/{max_retries})...")
                        result["retries"].append(f"Retry {retry_count}: No tool result received")
                        # Add retry information to the prompt
                        prompt += f"\n\nThis is retry attempt {retry_count}. Please ensure you use the generate_worksheet_tool to create the worksheet."
                        content = types.Content(role='user', parts=[types.Part(text=prompt)])
                    else:
                        result["error"] = "Failed to get worksheet result after multiple attempts"
                        
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    result["retries"].append(f"Retry {retry_count}: Exception - {error_msg}")
                    
                    if retry_count <= max_retries:
                        logger.warning(f"Error running agent, retrying ({retry_count}/{max_retries}): {error_msg}")
                    else:
                        raise  # Re-raise if we've exhausted retries
            
            # Process the result
            if tool_result:
                result["worksheet"] = tool_result.get("worksheet")
                result["raw_content"] = tool_result.get("raw_content")
                result["retries"].extend(tool_result.get("retries", []))
                result["succeeded_attempt"] = tool_result.get("succeeded_attempt")
                
                if tool_result.get("error"):
                    result["error"] = tool_result["error"]
                    self.metrics["failed_generations"] += 1
                    self.metrics["error_types"][tool_result["error"][:50]] = self.metrics["error_types"].get(tool_result["error"][:50], 0) + 1
                else:
                    self.metrics["successful_generations"] += 1
                    
                    # Perform quality evaluation if worksheet was generated successfully
                    if result["worksheet"]:
                        confidence_score = self._evaluate_worksheet_quality(result["worksheet"], grade_level, subject)
                        result["confidence_score"] = confidence_score
                        
                        # Analyze question distribution
                        result["question_analysis"] = self._analyze_question_distribution(result["worksheet"])
                        
                        # Generate improvement suggestions
                        result["suggestions"] = self._generate_improvement_suggestions(result["worksheet"], grade_level, difficulty)
                        
                        # Cache the successful result
                        self.worksheet_cache[cache_key] = result.copy()
                        
                        # Notify other agents about worksheet generation
                        try:
                            asyncio.create_task(self._notify_agents_about_worksheet(
                                subject, grade_level, difficulty, result
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to notify agents about worksheet: {str(e)}")
            else:
                # If we didn't get a tool result, call the tool directly as a fallback
                logger.info("Falling back to direct tool call")
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
                result["retries"].extend(worksheet_result.get("retries", []))
                result["succeeded_attempt"] = worksheet_result.get("succeeded_attempt")
                
                if worksheet_result.get("error"):
                    result["error"] = worksheet_result["error"]
                    self.metrics["failed_generations"] += 1
                else:
                    self.metrics["successful_generations"] += 1
            
            # Update timing metrics
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            self.metrics["total_generation_time"] += processing_time
            self.metrics["average_generation_time"] = self.metrics["total_generation_time"] / self.metrics["total_requests"]
            
            # Update retry metrics
            if result["retries"]:
                self.metrics["retry_count"] += len(result["retries"])
            
            return result
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = str(e)
            logger.error(f"Unexpected error in generate_worksheet: {error_msg}", exc_info=True)
            
            result["error"] = f"Unexpected error: {error_msg}"
            result["processing_time"] = time.time() - start_time
            self.metrics["failed_generations"] += 1
            self.metrics["error_types"]["unexpected_error"] = self.metrics["error_types"].get("unexpected_error", 0) + 1
            
            return result
    
    def _evaluate_worksheet_quality(self, worksheet: Dict[str, Any], grade_level: str, subject: str) -> float:
        """
        Evaluate the quality of a generated worksheet.
        
        Args:
            worksheet: The generated worksheet
            grade_level: Target grade level
            subject: Subject area
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not worksheet or not isinstance(worksheet, dict):
                return 0.0
            
            score = 0.0
            max_score = 5.0
            
            # Check if worksheet has questions
            questions = worksheet.get("questions", [])
            if questions and len(questions) > 0:
                score += 1.0
            
            # Check question variety
            question_types = set()
            for q in questions:
                if isinstance(q, dict) and "type" in q:
                    question_types.add(q["type"])
            if len(question_types) >= 2:
                score += 1.0
            
            # Check if answers are provided
            has_answers = any(isinstance(q, dict) and "answer" in q for q in questions)
            if has_answers:
                score += 1.0
            
            # Check question clarity (simple heuristic)
            clear_questions = 0
            for q in questions:
                if isinstance(q, dict) and "question" in q:
                    question_text = q["question"]
                    if len(question_text) > 10 and "?" in question_text:
                        clear_questions += 1
            if clear_questions >= len(questions) * 0.8:  # 80% of questions are clear
                score += 1.0
            
            # Check appropriate length
            if 5 <= len(questions) <= 20:  # Reasonable number of questions
                score += 1.0
            
            return min(1.0, score / max_score)
            
        except Exception as e:
            logger.error(f"Error evaluating worksheet quality: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def _analyze_question_distribution(self, worksheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the distribution of question types in the worksheet.
        
        Args:
            worksheet: The generated worksheet
            
        Returns:
            Analysis of question distribution
        """
        try:
            analysis = {
                "total_questions": 0,
                "question_types": {},
                "average_question_length": 0,
                "has_answer_key": False
            }
            
            questions = worksheet.get("questions", [])
            analysis["total_questions"] = len(questions)
            
            total_length = 0
            for q in questions:
                if isinstance(q, dict):
                    # Count question types
                    q_type = q.get("type", "unknown")
                    analysis["question_types"][q_type] = analysis["question_types"].get(q_type, 0) + 1
                    
                    # Calculate average length
                    question_text = q.get("question", "")
                    total_length += len(question_text)
                    
                    # Check for answers
                    if "answer" in q:
                        analysis["has_answer_key"] = True
            
            if questions:
                analysis["average_question_length"] = total_length / len(questions)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing question distribution: {str(e)}")
            return {"error": str(e)}
    
    def _generate_improvement_suggestions(self, worksheet: Dict[str, Any], grade_level: str, difficulty: str) -> List[str]:
        """
        Generate suggestions for improving the worksheet.
        
        Args:
            worksheet: The generated worksheet
            grade_level: Target grade level
            difficulty: Difficulty level
            
        Returns:
            List of improvement suggestions
        """
        try:
            suggestions = []
            
            questions = worksheet.get("questions", [])
            if not questions:
                suggestions.append("Add more questions to make the worksheet more comprehensive")
                return suggestions
            
            # Analyze question types
            question_types = {}
            for q in questions:
                if isinstance(q, dict):
                    q_type = q.get("type", "unknown")
                    question_types[q_type] = question_types.get(q_type, 0) + 1
            
            # Suggest variety if lacking
            if len(question_types) < 2:
                suggestions.append("Consider adding more variety in question types (multiple choice, short answer, etc.)")
            
            # Check for answer keys
            has_answers = any(isinstance(q, dict) and "answer" in q for q in questions)
            if not has_answers:
                suggestions.append("Include answer keys to help with grading and self-assessment")
            
            # Grade level specific suggestions
            grade_num = int(grade_level) if grade_level.isdigit() else 5
            if grade_num <= 3:
                suggestions.append("Consider adding more visual elements or simpler language for younger students")
            elif grade_num >= 8:
                suggestions.append("Consider adding more analytical or critical thinking questions")
            
            # Difficulty specific suggestions
            if difficulty == "easy":
                suggestions.append("Consider adding some recall-based questions for foundational knowledge")
            elif difficulty == "hard":
                suggestions.append("Consider adding application or synthesis questions for deeper learning")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}")
            return ["Review worksheet for clarity and completeness"]
    
    async def _notify_agents_about_worksheet(self, subject: str, grade_level: str, difficulty: str, result: Dict[str, Any]):
        """
        Notify other agents about worksheet generation.
        
        Args:
            subject: Subject area
            grade_level: Grade level
            difficulty: Difficulty level
            result: Generation result
        """
        try:
            notification_data = {
                "subject": subject,
                "grade_level": grade_level,
                "difficulty": difficulty,
                "question_count": result.get("question_analysis", {}).get("total_questions", 0),
                "confidence_score": result.get("confidence_score", 0),
                "processing_time": result.get("processing_time", 0)
            }
            
            notification = NotificationMessage(
                sender=self.agent_name,
                recipient="*",  # Broadcast to all agents
                notification_type="worksheet_generated",
                message=f"Generated worksheet for {subject} (Grade {grade_level}, {difficulty})",
                data=notification_data,
                priority=Priority.LOW
            )
            
            message_bus.send_message(notification)
            logger.info(f"Notified agents about worksheet generation: {subject}")
            
        except Exception as e:
            logger.error(f"Error notifying agents about worksheet: {str(e)}", exc_info=True)