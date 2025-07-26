import os
import re
from typing import Dict, Any, Optional, List, Tuple

import dotenv
import requests
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants
UNSAFE_KEYWORDS = [
    "porn", "sex", "nude", "naked", "xxx", "adult", "18+", "nsfw",
    "violence", "gore", "blood", "kill", "murder", "suicide",
    "drug", "cocaine", "heroin", "marijuana", "weed", "alcohol",
    "gambling", "betting", "casino"
]

QUESTION_WORDS = ["what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "explain"]

# Utility functions
def call_gemini_api(prompt: str, model: str = "gemini-1.5-flash", timeout: int = 10) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Call the Gemini API with a prompt.
    
    Args:
        prompt (str): The prompt to send to the API
        model (str, optional): The model to use. Defaults to "gemini-1.5-flash".
        timeout (int, optional): Timeout in seconds. Defaults to 10.
        
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: 
            - Success flag
            - Response text if successful, None otherwise
            - Error message if unsuccessful, None otherwise
    """
    # Get the API key from the environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return False, None, "GEMINI_API_KEY environment variable not set"
    
    try:
        # Call Gemini API
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            f"?key={gemini_api_key}"
        )
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(
            gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout
        )
        
        if response.status_code != 200:
            return False, None, f"API call failed with status code {response.status_code}: {response.text}"
            
        data = response.json()
        result = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        return True, result, None
        
    except Exception as e:
        return False, None, f"Error calling Gemini API: {str(e)}"

def simple_topic_extraction(message: str) -> str:
    """
    Simple fallback method to extract a topic from a message.
    
    Args:
        message (str): The user's message
        
    Returns:
        str: A simple topic extracted from the message
    """
    words = message.split()
    if len(words) <= 3:
        return message.lower()
    else:
        return " ".join(words[:3]).lower()

def simple_educational_check(message: str) -> bool:
    """
    Simple fallback method to check if a message is educational.
    
    Args:
        message (str): The user's message
        
    Returns:
        bool: True if the message appears to be educational, False otherwise
    """
    contains_question_word = any(word in message.lower() for word in QUESTION_WORDS)
    ends_with_question_mark = message.strip().endswith("?")
    return contains_question_word or ends_with_question_mark

def simple_safety_check(message: str) -> bool:
    """
    Simple fallback method to check if content is safe.
    
    Args:
        message (str): The user's message
        
    Returns:
        bool: True if the content appears to be safe, False otherwise
    """
    return not any(keyword in message.lower() for keyword in UNSAFE_KEYWORDS)

def simple_topic_similarity(topic1: str, topic2: str) -> bool:
    """
    Simple fallback method to check if two topics are similar.
    
    Args:
        topic1 (str): The first topic
        topic2 (str): The second topic
        
    Returns:
        bool: True if the topics appear to be similar, False otherwise
    """
    return topic1 == topic2 or topic1 in topic2 or topic2 in topic1

# Helper functions for the chatbot agent
def _extract_topic(message: str) -> str:
    """
    Extract the main educational topic from a user message using AI.
    
    Args:
        message (str): The user's message/question
        
    Returns:
        str: The main educational topic of the message
    """
    prompt = f"""
    Extract the main educational topic from this message. 
    Focus on identifying the core subject or concept being discussed in the context of a school curriculum.
    Return only the topic as a short phrase (1-5 words), with no additional text or explanation.
    
    Message: {message}
    
    Topic:
    """
    
    success, result, _ = call_gemini_api(prompt)
    
    if success and result:
        # Clean up the topic (remove any extra text, punctuation, etc.)
        topic = result.split('\n')[0].strip().lower()
        topic = re.sub(r'[^\w\s]', '', topic)
        return topic
    else:
        # Fallback to simple extraction if the API call fails
        return simple_topic_extraction(message)

def _is_educational_question(message: str) -> bool:
    """
    Check if a message is an educational question within the context of a school curriculum using AI.
    
    Args:
        message (str): The user's message/question
        
    Returns:
        bool: True if the message is an educational question, False otherwise
    """
    prompt = f"""
    Analyze the following message and determine if it is an educational question relevant to a school student's curriculum.
    
    An educational question is one that:
    1. Seeks knowledge or understanding about academic subjects (math, science, history, literature, etc.)
    2. Is appropriate for a school setting
    3. Could reasonably be part of a K-12 curriculum
    4. Is not seeking non-educational information (like entertainment, personal advice, etc.)
    5. Is not inappropriate for students
    
    Message: {message}
    
    Respond with only "Yes" if it is an educational question, or "No" if it is not.
    """
    
    success, result, _ = call_gemini_api(prompt)
    
    if success and result:
        # Check if the result contains "yes"
        return "yes" in result.lower()
    else:
        # Fallback to a simple heuristic if the API call fails
        return simple_educational_check(message)

def _are_topics_similar(topic1: str, topic2: str) -> bool:
    """
    Determine if two topics are semantically similar using AI.
    
    Args:
        topic1 (str): The first topic
        topic2 (str): The second topic
        
    Returns:
        bool: True if the topics are semantically similar, False otherwise
    """
    # If the topics are exactly the same, they are similar
    if topic1 == topic2:
        return True
    
    prompt = f"""
    Determine if these two educational topics are semantically similar or related.
    
    Topic 1: {topic1}
    Topic 2: {topic2}
    
    Consider them similar if:
    1. They refer to the same concept or subject
    2. One is a subtopic of the other
    3. They are closely related in an educational context
    4. They would be taught together in a curriculum
    
    Respond with only "Similar" if the topics are semantically similar or related, or "Different" if they are distinct topics.
    """
    
    success, result, _ = call_gemini_api(prompt)
    
    if success and result:
        # Check if the result contains "similar"
        return "similar" in result.lower()
    else:
        # Fallback to simple string comparison if the API call fails
        return simple_topic_similarity(topic1, topic2)

def _is_safe_content(message: str) -> bool:
    """
    Check if a message contains safe content appropriate for students (18-) using AI.
    
    Args:
        message (str): The user's message/question
        
    Returns:
        bool: True if the message contains safe content, False otherwise
    """
    prompt = f"""
    Analyze the following message and determine if it contains content that is safe and appropriate for students under 18 years old.
    
    Unsafe content includes:
    1. Sexual or adult content
    2. Graphic violence or gore
    3. Promotion of harmful substances (drugs, alcohol, etc.)
    4. Gambling or betting
    5. Hate speech, discrimination, or bullying
    6. Self-harm or suicide
    7. Any other content inappropriate for a school setting
    
    Message: {message}
    
    Respond with only "Safe" if the content is safe and appropriate for students, or "Unsafe" if it contains any inappropriate content.
    """
    
    success, result, _ = call_gemini_api(prompt)
    
    if success and result:
        # Check if the result contains "safe"
        return "safe" in result.lower() and "unsafe" not in result.lower()
    else:
        # Fallback to a simple keyword-based approach if the API call fails
        return simple_safety_check(message)

# Define the tool function for educational Q&A
def answer_educational_question(message: str, syllabus: str = "General", grade_level: str = "5") -> Dict[str, Any]:
    """Answers educational questions within the specified syllabus and grade level.
    
    Args:
        message (str): The user's question
        syllabus (str, optional): The syllabus context (e.g., "Math", "Science", "History"). Defaults to "General".
        Grade_level (str, optional): The target grade level. Defaults to "5".
        
    Returns:
        Dict[str, Any]: A dictionary containing the answer and metadata:
            - answer: The response to the user's question
            - topic: The detected topic of the question
            - is_educational: Whether the question was deemed educational
            - is_safe: Whether the content was deemed safe
            - error: Error message if any
    """
    print(f"--- Tool: answer_educational_question called with message: {message[:50]}... ---")
    
    # Initialize result dictionary
    result = {
        "answer": None,
        "topic": None,
        "is_educational": False,
        "is_safe": False,
        "error": None
    }
    
    # Extract the topic
    topic = _extract_topic(message)
    result["topic"] = topic
    
    # Check if the message is an educational question
    is_educational = _is_educational_question(message)
    result["is_educational"] = is_educational
    
    # Check if the content is safe
    is_safe = _is_safe_content(message)
    result["is_safe"] = is_safe
    
    # If the message is not educational or not safe, return an appropriate response
    if not is_educational:
        result["error"] = "I can only answer educational questions related to the syllabus."
        result["answer"] = "I'm sorry, but I can only answer educational questions related to the syllabus. Please ask a question about your studies."
        return result
    
    if not is_safe:
        result["error"] = "I cannot provide information on this topic as it may not be appropriate for educational purposes."
        result["answer"] = "I'm sorry, but I cannot provide information on this topic as it may not be appropriate for educational purposes. Please ask a different question."
        return result
    
    # Prepare the prompt for Gemini
    prompt = f"""
    You are an educational assistant helping a grade {grade_level} student with a question about {syllabus}.
    
    The student's question is: {message}
    
    Provide a clear, accurate, and educational response appropriate for a grade {grade_level} student.
    Make sure your answer is:
    1. Educational and informative
    2. Age-appropriate for grade {grade_level}
    3. Related to the {syllabus} syllabus
    4. Factually correct
    5. Easy to understand
    
    Your response should be helpful and encourage further learning.
    """
    
    success, answer, error = call_gemini_api(prompt, timeout=30)
    
    if success and answer:
        result["answer"] = answer
    else:
        result["error"] = error or "Unknown error generating answer"
        result["answer"] = "I'm sorry, but I'm having trouble processing your question right now. Please try again later."
    
    return result

# Define the tool function for generating diagrams via A2A
def generate_diagram_for_topic(topic: str) -> Dict[str, Any]:
    """Generates a diagram for a specific topic by calling the diagram agent.
    
    Args:
        topic (str): The topic to generate a diagram for
        
    Returns:
        Dict[str, Any]: A dictionary containing the diagram data:
            - image: Base64-encoded image data URI if successful
            - diagram_code: The generated diagram code
            - error: Error message if generation failed
    """
    print(f"--- Tool: generate_diagram_for_topic called for topic: {topic} ---")
    
    # Initialize result dictionary
    result = {
        "image": None,
        "diagram_code": None,
        "error": None
    }
    
    # Get the API key from the environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        result["error"] = "GEMINI_API_KEY environment variable not set"
        return result
    
    # Import the diagram agent here to avoid circular imports
    from app.agents.diagram_agent_adk import generate_diagram_tool
    
    # Prepare the prompt for the diagram
    prompt = f"Create an educational diagram explaining the concept of '{topic}' for students. The diagram should be clear, informative, and help visualize the key aspects of {topic}."
    
    # Call the diagram tool directly
    diagram_result = generate_diagram_tool(
        user_prompt=prompt,
        code_style="graphviz",
        output_format="png"
    )
    
    # Copy the relevant fields from the diagram result
    result["image"] = diagram_result.get("image")
    result["diagram_code"] = diagram_result.get("diagram_code")
    
    if diagram_result.get("error"):
        result["error"] = diagram_result["error"]
    
    return result

# Create the ADK Agent
class ChatbotAgentADK:
    def __init__(self, model="gemini-1.5-flash"):
        """Initialize the ADK-based Chatbot Agent.
        
        Args:
            model (str): The model to use for the agent. Defaults to "gemini-1.5-flash".
        """
        self.model = model
        self.agent = Agent(
            name="chatbot_agent",
            model=model,
            description="Educational chatbot that answers questions within the syllabus.",
            instruction="""You are a helpful educational assistant that answers questions within the syllabus.
            
            When a user asks a question:
            1. Use the 'answer_educational_question' tool to provide an educational response.
            2. Only answer questions that are educational and within the syllabus.
            3. Ensure all content is safe and appropriate for students (18-).
            4. If you detect that the user has asked 3 consecutive questions about the same topic,
               use the 'generate_diagram_for_topic' tool to create a visual explanation.
            
            Always be helpful, educational, and encouraging. If a question is not educational or contains
            inappropriate content, politely explain that you can only answer educational questions within the syllabus.
            """,
            tools=[answer_educational_question, generate_diagram_for_topic],
        )
        
        # Create a session service for managing conversation history
        self.session_service = InMemorySessionService()
        
        # Constants for identifying the interaction context
        self.app_name = "chatbot_app"
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
    async def create_session(self, user_id: str):
        """
        Create a new session for a user.
        
        Args:
            user_id (str): The user's ID
            
        Returns:
            str: The session ID
        """
        # Create a unique session ID
        session_id = f"session_{os.urandom(4).hex()}"
        
        # Create the session
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # Initialize session state
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        if session is not None:
            session.state = {
                "topic_counter": {},  # Counter for topics
                "last_topics": [],    # List of recent topics
                "question_count": 0   # Total question count
            }
        
        return session_id
    
    async def get_session_state(self, user_id: str, session_id: str):
        """
        Get the state of a session.
        
        Args:
            user_id (str): The user's ID
            session_id (str): The session ID
            
        Returns:
            Dict: The session state
        """
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        return session.state if session else {}
    
    async def update_session_state(self, user_id: str, session_id: str, topic: str):
        """
        Update the session state with a new topic.
        
        Args:
            user_id (str): The user's ID
            session_id (str): The session ID
            topic (str): The topic of the user's question
            
        Returns:
            Dict: The updated session state
        """
        # Get the current state
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        if not session:
            return {}
            
        state = session.state
        
        # Update the topic counter
        topic_counter = state.get("topic_counter", {})
        topic_counter[topic] = topic_counter.get(topic, 0) + 1
        
        # Update the last topics list (keep the last 5)
        last_topics = state.get("last_topics", [])
        last_topics.append(topic)
        if len(last_topics) > 5:
            last_topics = last_topics[-5:]
        
        # Update the question count
        question_count = state.get("question_count", 0) + 1
        
        # Update the session state
        updated_state = {
            "topic_counter": topic_counter,
            "last_topics": last_topics,
            "question_count": question_count
        }
        
        session.state.update(updated_state)
        
        return session.state
    
    def should_generate_diagram(self, state: Dict[str, Any], topic: str) -> bool:
        """
        Determine if a diagram should be generated based on the session state,
        using AI to detect if questions are semantically about the same topic.
        
        Args:
            state (Dict[str, Any]): The session state
            topic (str): The current topic
            
        Returns:
            bool: True if a diagram should be generated, False otherwise
        """
        # Check if the topic has been asked about 3 or more times
        topic_counter = state.get("topic_counter", {})
        
        # Count similar topics using AI-based similarity detection
        similar_topic_count = 0
        for existing_topic, count in topic_counter.items():
            if _are_topics_similar(topic, existing_topic):
                similar_topic_count += count
        
        # Check if the last 3 topics are semantically similar to the current topic
        last_topics = state.get("last_topics", [])
        if len(last_topics) >= 3:
            last_three = last_topics[-3:]
            similar_topics_count = sum(1 for t in last_three if _are_topics_similar(topic, t))
            if similar_topics_count >= 2:  # If at least 2 of the last 3 topics are similar
                return True
        
        # If similar topics have been asked about 3 or more times, generate a diagram
        return similar_topic_count >= 3
    
    async def chat(self, user_id: str, session_id: str, message: str, syllabus: str = "General", grade_level: str = "5"):
        """
        Process a chat message and generate a response.
        
        Args:
            user_id (str): The user's ID
            session_id (str): The session ID
            message (str): The user's message
            syllabus (str, optional): The syllabus context. Defaults to "General".
            Grade_level (str, optional): The target grade level. Defaults to "5".
            
        Returns:
            Dict[str, Any]: The response data
        """
        # Initialize result dictionary
        result = {
            "answer": None,
            "diagram": None,
            "topic": None,
            "error": None
        }
        
        try:
            # Extract the topic
            topic = _extract_topic(message)
            result["topic"] = topic
            
            # Update the session state
            state = await self.update_session_state(user_id, session_id, topic)
            
            # Check if we should generate a diagram
            should_diagram = self.should_generate_diagram(state, topic)
            
            # Prepare the message for the agent
            prompt = f"""
            User question: {message}
            
            Syllabus: {syllabus}
            Grade level: {grade_level}
            
            {f'I notice you have asked multiple questions about topics related to "{topic}". Please generate a diagram to help explain this concept more clearly.' if should_diagram else ''}
            """
            
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            # Run the agent
            final_response = None
            async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        result["error"] = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
            
            # If we got a final response, it means the agent successfully processed the message
            if final_response:
                result["answer"] = final_response
                
                # If we should generate a diagram, do it
                if should_diagram:
                    # Call the diagram tool directly
                    diagram_result = generate_diagram_for_topic(topic)
                    result["diagram"] = diagram_result.get("image")
            
        except Exception as e:
            result["error"] = f"Error processing message: {str(e)}"
        
        return result

# Async helper function to call the agent
async def call_chatbot_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and returns the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Iterate through events to find the final answer
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"

    return final_response_text