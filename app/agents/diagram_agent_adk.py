import os
import requests
import base64
import ast
import re
import logging
import asyncio
from typing import Dict, Any, Optional

import dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Load environment variables from .env file
dotenv.load_dotenv()

# Helper functions from the original implementation
def _clean_generated_code(raw_code):
    """Clean Gemini output by handling escapes, removing code block markers, etc."""
    # Replace escaped newlines
    code = raw_code.replace("\\n", "\n")

    # Optional: decode full escape sequences safely
    try:
        code = ast.literal_eval(f"'''{code}'''")
    except Exception as e:
        logging.warning(f"Failed to decode escape sequences: {e}")

    # Remove fenced code block markers if present
    fenced_block_match = re.search(r"```(?:[a-zA-Z0-9]*)?\n(.*?)```", code, flags=re.DOTALL)
    if fenced_block_match:
        code = fenced_block_match.group(1).strip()

    logging.debug(f"Cleaned diagram code:\n{code}")
    return code


def _render_diagram(code, code_style="graphviz", output_format="pdf"):
    """Render diagram code using Kroki service."""
    if not code or not code.strip():
        raise RuntimeError("No diagram code provided to render.")
    
    kroki_base_url = "https://kroki.io"
    url = f"{kroki_base_url}/{code_style}/{output_format}"
    headers = {"Content-Type": "text/plain", "Accept": f"image/{output_format}"}
    response = requests.post(url, data=code.encode("utf-8"), headers=headers, timeout=30)
    
    if response.status_code != 200:
        raise RuntimeError(
            f"Kroki rendering failed with status {response.status_code}: {response.content[:200]!r}"
        )
    
    encoded_img = base64.b64encode(response.content).decode('utf-8')
    return f"data:image/{output_format};base64,{encoded_img}"


def _self_evaluate_diagram(code, image_uri, code_style, output_format):
    """
    Agentic check: is the code non-empty, and did rendering yield a plausible image (not blank, not error)?
    """
    if not code or not code.strip():
        return False, "Code is empty or blank."
    if not image_uri or not image_uri.startswith(f"data:image/{output_format}"):
        return False, "Rendered image URI is missing or incorrect."
    if len(image_uri) < 100:
        return False, "Image data URI is suspiciously short."
    # You could decode base64 and check for certain error-strings or patterns for more robustness.
    return True, "OK"


# Define the tool function for ADK
def generate_diagram_tool(user_prompt: str, code_style: str = "graphviz", 
                         output_format: str = "pdf") -> Dict[str, Any]:
    """Generates a diagram based on a user's description.

    Args:
        user_prompt (str): The user's description of the diagram they want to create.
        code_style (str, optional): The diagram syntax to use (graphviz, mermaid, plantuml, etc.). Defaults to "graphviz".
        output_format (str, optional): The output format (pdf, png, svg, etc.). Defaults to "pdf".

    Returns:
        Dict[str, Any]: A dictionary containing the generated diagram with the following keys:
            - image: Base64-encoded image data URI if successful
            - diagram_code: The generated diagram code
            - error: Error message if generation failed
            - retries: List of retry attempts and reasons
            - succeeded_attempt: The attempt number that succeeded (if any)
    """
    print(f"--- Tool: generate_diagram_tool called with prompt: {user_prompt[:50]}... ---")
    
    # Initialize result dictionary
    result = {"image": None, "diagram_code": None, "error": None, "retries": []}
    
    if not user_prompt.strip():
        result["error"] = "Prompt cannot be empty."
        return result
    
    # Get API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        result["error"] = "GEMINI_API_KEY environment variable not set"
        return result
    
    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
        f"?key={gemini_api_key}"
    )
    
    # Maximum number of attempts
    max_attempts = 3
    last_reason = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Build prompt
            gemini_prompt = (
                f"Generate a diagram in {code_style} syntax for the following description:\n{user_prompt}"
            )
            
            # Call Gemini API
            payload = {"contents": [{"parts": [{"text": gemini_prompt}]}]}
            response = requests.post(
                gemini_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30
            )
            
            if response.status_code != 200:
                last_reason = f"Gemini API failed: {response.status_code}: {response.text}"
                result["retries"].append(f"Attempt {attempt}: {last_reason}")
                continue
                
            data = response.json()
            raw_code = data['candidates'][0]['content']['parts'][0]['text']
            
            # Clean the generated code
            code = _clean_generated_code(raw_code)
            if not code.strip():
                last_reason = "Generated diagram code is empty after cleaning."
                result["retries"].append(f"Attempt {attempt}: {last_reason}")
                continue
                
            # Try rendering
            try:
                image_data_uri = _render_diagram(code, code_style, output_format)
            except Exception as render_exc:
                last_reason = f"Render failed: {render_exc}"
                result["retries"].append(f"Attempt {attempt}: {last_reason}")
                continue
                
            # Self-evaluate the result
            passed, reason = _self_evaluate_diagram(code, image_data_uri, code_style, output_format)
            if passed:
                result["diagram_code"] = code
                result["image"] = image_data_uri
                result["succeeded_attempt"] = attempt
                return result
            else:
                last_reason = reason
                result["retries"].append(f"Attempt {attempt}: {reason}")
                
        except Exception as exc:
            last_reason = str(exc)
            result["retries"].append(f"Attempt {attempt} exception: {last_reason}")
            
    # If the loop ends without success
    result["error"] = f"Diagram generation failed after {max_attempts} attempts. Last reason: {last_reason}"
    return result


# Create the ADK Agent
class DiagramAgentADK:
    def __init__(self, model="gemini-2.5-flash"):
        """Initialize the ADK-based Diagram Agent.
        
        Args:
            model (str): The model to use for the agent. Defaults to "gemini-2.5-flash".
        """
        self.model = model
        self.agent = Agent(
            name="diagram_agent",
            model=model,
            description="Generates diagrams based on user descriptions.",
            instruction="""You are a helpful diagram assistant that creates visual diagrams based on user descriptions.
            When a user requests a diagram, use the 'generate_diagram_tool' to create a diagram in the requested style.
            
            Always ask for the following information if not provided:
            1. A clear description of what the diagram should represent
            2. The preferred diagram syntax (default is graphviz)
            3. The preferred output format (default is pdf)
            
            Ensure the diagram accurately represents the user's description.
            Present the diagram results in a clear, organized manner.
            """,
            tools=[generate_diagram_tool],
        )
        
        # Create session service for managing conversation history
        self.session_service = InMemorySessionService()
        
        # Constants for identifying the interaction context
        self.app_name = "diagram_app"
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
    async def generate_diagram(self, user_prompt, code_style="graphviz", output_format="pdf"):
        """
        Generate a diagram based on a user's description using the ADK agent.
        
        Args:
            user_prompt: The user's description of the diagram they want to create
            code_style: The diagram syntax to use (graphviz, mermaid, plantuml, etc.)
            output_format: The output format (pdf, png, svg, etc.)
            
        Returns:
            Dict containing the generated diagram and metadata
        """
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
        prompt = f"""Generate a diagram with the following specifications:
        - Description: {user_prompt}
        - Diagram Syntax: {code_style}
        - Output Format: {output_format}
        """
        
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        
        # Initialize result dictionary
        result = {"image": None, "diagram_code": None, "error": None, "retries": []}
        
        # Run the agent
        final_response = None
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    result["error"] = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
                
        # If we got a final response, it means the agent successfully called the tool
        if final_response:
            # The response should contain the diagram data
            # We'll extract it from the session state
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            # For now, we'll call the tool directly to get the diagram
            # This is a workaround since we can't easily extract tool results from the session
            diagram_result = generate_diagram_tool(
                user_prompt=user_prompt,
                code_style=code_style,
                output_format=output_format
            )
            
            result["image"] = diagram_result.get("image")
            result["diagram_code"] = diagram_result.get("diagram_code")
            result["retries"] = diagram_result.get("retries", [])
            result["succeeded_attempt"] = diagram_result.get("succeeded_attempt")
            
            if diagram_result.get("error"):
                result["error"] = diagram_result["error"]
                
        return result


# Async helper function to call the agent
async def call_diagram_agent_async(query: str, runner, user_id, session_id):
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
            break

    return final_response_text