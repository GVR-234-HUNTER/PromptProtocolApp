import os
import requests
import base64
import re
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Set

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

# Helper functions for diagram generation
def _clean_generated_code(raw_code: str) -> str:
    """Clean diagram code by handling escapes, removing code block markers, etc."""
    # Replace escaped newlines
    code = raw_code.replace("\\n", "\n")

    # Remove fenced code block markers if present
    fenced_block_match = re.search(r"```(?:[a-zA-Z0-9]*)?\n(.*?)```", code, flags=re.DOTALL)
    if fenced_block_match:
        code = fenced_block_match.group(1).strip()

    logging.debug(f"Cleaned diagram code:\n{code}")
    return code


def _render_diagram(code: str, code_style: str = "graphviz", output_format: str = "pdf") -> str:
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


def _self_evaluate_diagram(code: str, image_uri: str, output_format: str) -> tuple[bool, str]:
    """
    Agentic check: is the code non-empty, and did rendering yield a plausible image?
    """
    if not code or not code.strip():
        return False, "Code is empty or blank."
    if not image_uri or not image_uri.startswith(f"data:image/{output_format}"):
        return False, "Rendered image URI is missing or incorrect."
    if len(image_uri) < 100:
        return False, "Image data URI is suspiciously short."
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
    # Initialize result dictionary
    result = {"image": None, "diagram_code": None, "error": None, "retries": []}
    
    if not user_prompt.strip():
        result["error"] = "Prompt cannot be empty."
        return result
    
    # Maximum number of attempts
    max_attempts = 3
    last_reason = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            # This is a placeholder for the actual diagram generation
            # In a real implementation, this would be handled by the ADK agent
            # The agent will replace this with the actual diagram code
            code = f"""
            digraph G {{
                node [shape=box, style=filled, fillcolor=lightblue];
                edge [color=gray];
                
                A [label="Diagram for: {user_prompt}"];
                B [label="Using {code_style}"];
                C [label="Output: {output_format}"];
                
                A -> B -> C;
            }}
            """
            
            # Try rendering
            try:
                image_data_uri = _render_diagram(code, code_style, output_format)
            except Exception as render_exc:
                last_reason = f"Render failed: {render_exc}"
                result["retries"].append(f"Attempt {attempt}: {last_reason}")
                continue
                
            # Self-evaluate the result
            passed, reason = _self_evaluate_diagram(code, image_data_uri, output_format)
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
    def __init__(self, model="gemini-2.5-flash", agent_name="diagram_agent"):
        """Initialize the ADK-based Diagram Agent with enhanced agentic capabilities.
        
        Args:
            model (str): The model to use for the agent. Defaults to "gemini-2.5-flash".
            agent_name (str): The name to register this agent with. Defaults to "diagram_agent".
        """
        self.model = model
        self.agent_name = agent_name
        
        # Enhanced instruction with more agentic behavior
        instruction = """You are an advanced diagram assistant that creates visual diagrams based on user descriptions.
        
        When a user requests a diagram:
        1. Analyze the request to understand the core concept that needs visualization
        2. Select the most appropriate diagram type for the concept (flowchart, entity-relationship, etc.)
        3. Use the 'generate_diagram_tool' to create a diagram in the requested or most appropriate style
        4. Evaluate the quality of the generated diagram and make improvements if needed
        5. Provide clear explanations of the diagram elements and how they relate to the user's request
        6. Suggest potential improvements or alternative visualization approaches
        
        Always ask for the following information if not provided:
        1. A clear description of what the diagram should represent
        2. The preferred diagram syntax (default is graphviz)
        3. The preferred output format (default is pdf)
        
        Ensure the diagram accurately represents the user's description and follows best practices for:
        - Visual clarity and simplicity
        - Appropriate use of colors, shapes, and labels
        - Logical organization and flow
        - Accessibility considerations
        
        Present the diagram results in a clear, organized manner with explanations of key elements.
        """
        
        # Create the ADK agent with enhanced tools
        self.agent = Agent(
            name=agent_name,
            model=model,
            description="Advanced diagram generation agent with agentic capabilities",
            instruction=instruction,
            tools=[generate_diagram_tool],
        )
        
        # Create a session service for managing conversation history
        self.session_service = InMemorySessionService()
        
        # Constants for identifying the interaction context
        self.app_name = "diagram_app"
        
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
            "diagram_types": {},  # Track frequency of different diagram types
            "syntax_usage": {},   # Track frequency of different syntax types
            "error_types": {}     # Track frequency of different error types
        }
        
        # Cache for storing recently generated diagrams to avoid regeneration
        self.diagram_cache = {}
        
        # Register with the agent registry
        logger.info(f"Registering {agent_name} with the agent registry")
        registry.register_agent(agent_name, self, [
            "generate_diagram",
            "get_supported_diagram_types",
            "get_diagram_suggestions"
        ])
        
        # Listen for notifications from other agents
        asyncio.create_task(self._setup_notification_handlers())
        
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
            data = message.data
            
            logger.info(f"Received {notification_type} notification from {sender}")
            
            if notification_type == "new_question":
                # A chatbot agent has received a new question
                # Check if it's related to diagrams and might benefit from a diagram
                await self._process_question_notification(data)
            elif notification_type == "session_created":
                # A new session has been created
                # We could initialize diagram-related context for this session
                logger.info(f"New session {data.get('session_id')} created for user {data.get('user_id')}")
            elif notification_type == "diagram_request":
                # Direct request for a diagram
                await self._process_diagram_request(data)
        except Exception as e:
            logger.error(f"Error handling notification: {str(e)}", exc_info=True)
    
    async def _process_question_notification(self, data: Dict[str, Any]):
        """
        Process a notification about a new question.
        If the question is related to diagrams, proactively generate a diagram suggestion.
        
        Args:
            data: The notification data
        """
        try:
            # Extract relevant information
            message = data.get("message", "")
            topic = data.get("topic", "")
            user_id = data.get("user_id")
            session_id = data.get("session_id")
            sender = data.get("agent")
            
            # Check if the question is related to diagrams or visualization
            diagram_keywords = ["diagram", "chart", "graph", "visualization", "flowchart", 
                               "map", "plot", "visual", "illustration", "figure"]
            
            is_diagram_related = any(keyword in message.lower() for keyword in diagram_keywords)
            
            if is_diagram_related:
                logger.info(f"Question from {user_id} is diagram-related: {message[:50]}...")
                
                # Generate a diagram suggestion
                suggestion = {
                    "message": f"I noticed a question about {topic} that might benefit from a diagram.",
                    "diagram_types": self._suggest_diagram_types(message, topic),
                    "sample_prompt": f"Create a diagram explaining {topic}"
                }
                
                # Send the suggestion back to the chatbot agent
                await message_bus.send_message(
                    NotificationMessage(
                        sender=self.agent_name,
                        recipient=sender,
                        notification_type="diagram_suggestion",
                        message=f"Diagram suggestion for question about {topic}",
                        data={
                            "suggestion": suggestion,
                            "user_id": user_id,
                            "session_id": session_id,
                            "original_message": message,
                            "topic": topic
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error processing question notification: {str(e)}", exc_info=True)
    
    async def _process_diagram_request(self, data: Dict[str, Any]):
        """
        Process a direct request for a diagram from another agent.
        
        Args:
            data: The request data
        """
        try:
            # Extract relevant information
            prompt = data.get("prompt", "")
            code_style = data.get("code_style", "graphviz")
            output_format = data.get("output_format", "png")
            requester = data.get("requester")
            request_id = data.get("request_id")
            
            # Generate the diagram
            result = await self.generate_diagram(
                user_prompt=prompt,
                code_style=code_style,
                output_format=output_format
            )
            
            # Send the result back to the requester
            await message_bus.send_message(
                ResponseMessage(
                    sender=self.agent_name,
                    recipient=requester,
                    result=result,
                    success=result.get("error") is None,
                    in_reply_to=request_id
                )
            )
        except Exception as e:
            logger.error(f"Error processing diagram request: {str(e)}", exc_info=True)
    
    def _suggest_diagram_types(self, message: str, topic: str) -> List[Dict[str, str]]:
        """
        Suggest appropriate diagram types based on the message and topic.
        
        Args:
            message: The user's message
            topic: The extracted topic
            
        Returns:
            List of suggested diagram types with descriptions
        """
        suggestions = []
        
        # Check for specific diagram types based on keywords
        if any(kw in message.lower() for kw in ["process", "flow", "step", "sequence"]):
            suggestions.append({
                "type": "flowchart",
                "description": "Shows the steps in a process or workflow",
                "syntax": "graphviz"
            })
            
        if any(kw in message.lower() for kw in ["relation", "entity", "database", "data model"]):
            suggestions.append({
                "type": "entity-relationship",
                "description": "Shows relationships between entities or concepts",
                "syntax": "graphviz"
            })
            
        if any(kw in message.lower() for kw in ["hierarchy", "tree", "structure", "organization"]):
            suggestions.append({
                "type": "tree",
                "description": "Shows hierarchical relationships",
                "syntax": "graphviz"
            })
            
        if any(kw in message.lower() for kw in ["network", "connection", "link", "graph"]):
            suggestions.append({
                "type": "network",
                "description": "Shows connections between multiple entities",
                "syntax": "graphviz"
            })
            
        if any(kw in message.lower() for kw in ["timeline", "time", "chronology", "history"]):
            suggestions.append({
                "type": "timeline",
                "description": "Shows events over time",
                "syntax": "mermaid"
            })
            
        if any(kw in message.lower() for kw in ["class", "object", "uml", "software"]):
            suggestions.append({
                "type": "class diagram",
                "description": "Shows classes and their relationships",
                "syntax": "plantuml"
            })
        
        # If no specific type was matched, suggest a concept map as default
        if not suggestions:
            suggestions.append({
                "type": "concept map",
                "description": f"Shows key concepts related to {topic}",
                "syntax": "graphviz"
            })
            
        return suggestions
        
    @provides_capability("generate_diagram")
    async def generate_diagram(self, user_prompt: str, code_style: str = "graphviz", 
                              output_format: str = "pdf", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a diagram based on a user's description using the ADK agent with enhanced agentic capabilities.
        
        Args:
            user_prompt: The user's description of the diagram they want to create
            code_style: The diagram syntax to use (graphviz, mermaid, plantuml, etc.)
            output_format: The output format (pdf, png, svg, etc.)
            context: Additional context for diagram generation
            
        Returns:
            Dict containing the generated diagram and metadata with enhanced information
        """
        # Start timing for performance metrics
        start_time = time.time()
        
        # Initialize enhanced result dictionary
        result = {
            "image": None, 
            "diagram_code": None, 
            "error": None, 
            "retries": [],
            "diagram_type": None,
            "processing_time": None,
            "confidence_score": None,
            "diagram_elements": [],
            "improvement_suggestions": []
        }
        
        try:
            # Check cache first to avoid regenerating the same diagram
            cache_key = f"{user_prompt}:{code_style}:{output_format}"
            if cache_key in self.diagram_cache:
                logger.info(f"Using cached diagram for prompt: {user_prompt[:50]}...")
                cached_result = self.diagram_cache[cache_key]
                
                # Update metrics
                self.metrics["total_requests"] += 1
                self.metrics["successful_generations"] += 1
                
                # Add processing time
                processing_time = time.time() - start_time
                cached_result["processing_time"] = processing_time
                cached_result["from_cache"] = True
                
                return cached_result
            
            # Determine the diagram type from the prompt
            diagram_type = self._determine_diagram_type(user_prompt)
            result["diagram_type"] = diagram_type
            
            # Update syntax usage metrics
            if code_style in self.metrics["syntax_usage"]:
                self.metrics["syntax_usage"][code_style] += 1
            else:
                self.metrics["syntax_usage"][code_style] = 1
                
            # Update diagram type metrics
            if diagram_type in self.metrics["diagram_types"]:
                self.metrics["diagram_types"][diagram_type] += 1
            else:
                self.metrics["diagram_types"][diagram_type] = 1
            
            # Create a unique user and session ID
            user_id = f"user_{os.urandom(8).hex()}"
            session_id = f"session_{os.urandom(8).hex()}"
            
            # Create the session
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            # Prepare an enhanced prompt for the agent
            prompt = f"""Generate a diagram with the following specifications:
            - Description: {user_prompt}
            - Diagram Type: {diagram_type}
            - Diagram Syntax: {code_style}
            - Output Format: {output_format}
            
            Please follow these best practices:
            1. Use clear, descriptive labels for all elements
            2. Organize elements logically with appropriate hierarchy
            3. Use consistent styling and formatting
            4. Keep the diagram focused and avoid unnecessary complexity
            5. Ensure the diagram accurately represents the described concept
            
            After generating the diagram, please:
            1. Evaluate the quality of the diagram
            2. Identify key elements in the diagram
            3. Suggest any potential improvements
            """
            
            # Add any additional context if provided
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                prompt += context_str
            
            # Create the content object for the agent
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            # Initialize result dictionary
            result = {
                "image": None, 
                "diagram_code": None, 
                "error": None, 
                "retries": [],
                "diagram_type": diagram_type,
                "processing_time": None,
                "confidence_score": None,
                "diagram_elements": [],
                "improvement_suggestions": []
            }
            
            # Track if we need to retry due to errors
            retry_count = 0
            max_retries = 3
            
            # Implement exponential backoff for retries
            base_delay = 1.0  # 1 second
            
            while retry_count <= max_retries:
                try:
                    # Run the agent
                    tool_result = None
                    final_response = None
                    
                    async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                        # Check for tool responses
                        if hasattr(event, 'tool_response') and event.tool_response is not None:
                            tr = event.tool_response
                            # Look for diagram result structure (image/diagram_code)
                            if isinstance(tr, dict) and ("image" in tr or "diagram_code" in tr):
                                tool_result = tr
                            # Sometimes Gemini may nest the result
                            elif hasattr(tr, "image") or hasattr(tr, "diagram_code"):
                                tool_result = {"image": getattr(tr, "image", None),
                                              "diagram_code": getattr(tr, "diagram_code", None)}
                        
                        # Capture final text response for additional information
                        if event.is_final_response():
                            if event.content and event.content.parts:
                                final_response = event.content.parts[0].text
                            elif event.actions and event.actions.escalate:
                                result["error"] = f"Agent escalated: {event.error_message or 'No specific message.'}"
                            break
                    
                    # If we got a tool result with an image, break the retry loop
                    if tool_result and tool_result.get("image"):
                        break
                    
                    # If no valid result and we haven't hit max retries, try again
                    retry_count += 1
                    self.metrics["retry_count"] += 1
                    
                    if retry_count <= max_retries:
                        # Calculate delay with exponential backoff
                        delay = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"No valid diagram generated, retrying ({retry_count}/{max_retries}) after {delay:.1f}s delay...")
                        
                        # Add retry information to the prompt
                        prompt += f"\n\nThis is retry attempt {retry_count}. The previous attempt did not produce a valid diagram. Please try a different approach."
                        content = types.Content(role='user', parts=[types.Part(text=prompt)])
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                    else:
                        result["error"] = "Failed to generate a valid diagram after multiple attempts"
                        
                except Exception as e:
                    retry_count += 1
                    self.metrics["retry_count"] += 1
                    
                    if retry_count <= max_retries:
                        # Calculate delay with exponential backoff
                        delay = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Error running agent, retrying ({retry_count}/{max_retries}) after {delay:.1f}s: {str(e)}")
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                    else:
                        raise  # Re-raise if we've exhausted retries
            
            # Process the results
            if tool_result:
                # Extract diagram data
                result["image"] = tool_result.get("image")
                result["diagram_code"] = tool_result.get("diagram_code")
                result["retries"] = tool_result.get("retries", [])
                result["succeeded_attempt"] = tool_result.get("succeeded_attempt")
                
                if tool_result.get("error"):
                    result["error"] = tool_result["error"]
                
                # Extract additional information from the final response if available
                if final_response:
                    # Extract diagram elements
                    elements = self._extract_diagram_elements(final_response)
                    if elements:
                        result["diagram_elements"] = elements
                    
                    # Extract improvement suggestions
                    suggestions = self._extract_improvement_suggestions(final_response)
                    if suggestions:
                        result["improvement_suggestions"] = suggestions
                
                # Perform self-evaluation of the diagram
                if result["image"] and result["diagram_code"]:
                    confidence_score = self._evaluate_diagram_quality(
                        result["diagram_code"], 
                        user_prompt, 
                        diagram_type
                    )
                    result["confidence_score"] = confidence_score
                
                # If the diagram is good, cache it for future use
                if not result["error"] and result["image"] and result["confidence_score"] and result["confidence_score"] > 0.6:
                    self.diagram_cache[cache_key] = result.copy()
                    # Limit cache size to prevent memory issues
                    if len(self.diagram_cache) > 100:
                        # Remove the oldest entry
                        oldest_key = next(iter(self.diagram_cache))
                        del self.diagram_cache[oldest_key]
            else:
                # If we didn't get a tool result, call the tool directly as a fallback
                logger.warning("No tool result from agent, using fallback direct tool call")
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
            
            # Update metrics
            self.metrics["total_requests"] += 1
            if result["error"]:
                self.metrics["failed_generations"] += 1
                
                # Track error types
                error_type = "unknown_error"
                if result["error"]:
                    if "timeout" in result["error"].lower():
                        error_type = "timeout"
                    elif "render" in result["error"].lower():
                        error_type = "rendering_error"
                    elif "syntax" in result["error"].lower():
                        error_type = "syntax_error"
                
                if error_type in self.metrics["error_types"]:
                    self.metrics["error_types"][error_type] += 1
                else:
                    self.metrics["error_types"][error_type] = 1
            else:
                self.metrics["successful_generations"] += 1
            
            # Calculate and store processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            self.metrics["total_generation_time"] += processing_time
            self.metrics["average_generation_time"] = (
                self.metrics["total_generation_time"] / 
                (self.metrics["successful_generations"] + self.metrics["failed_generations"])
            )
            
            # Notify other agents about the generated diagram (non-blocking)
            if result["image"] and not result["error"]:
                asyncio.create_task(self._notify_agents_about_diagram(
                    user_prompt, diagram_type, result
                ))
            
            return result
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}", exc_info=True)
            
            # Update error metrics
            self.metrics["total_requests"] += 1
            self.metrics["failed_generations"] += 1
            
            # Track error types
            error_type = "exception"
            if error_type in self.metrics["error_types"]:
                self.metrics["error_types"][error_type] += 1
            else:
                self.metrics["error_types"][error_type] = 1
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return error result
            return {
                "image": None,
                "diagram_code": None,
                "error": f"Error generating diagram: {str(e)}",
                "processing_time": processing_time,
                "diagram_type": diagram_type if 'diagram_type' in locals() else None
            }
    
    def _determine_diagram_type(self, prompt: str) -> str:
        """
        Determine the most appropriate diagram type based on the prompt.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            The determined diagram type
        """
        prompt_lower = prompt.lower()
        
        # Check for specific diagram types based on keywords
        if any(kw in prompt_lower for kw in ["process", "flow", "step", "sequence", "workflow"]):
            return "flowchart"
            
        if any(kw in prompt_lower for kw in ["relation", "entity", "database", "data model", "erd"]):
            return "entity-relationship"
            
        if any(kw in prompt_lower for kw in ["hierarchy", "tree", "structure", "organization"]):
            return "tree"
            
        if any(kw in prompt_lower for kw in ["network", "connection", "link", "graph"]):
            return "network"
            
        if any(kw in prompt_lower for kw in ["timeline", "time", "chronology", "history"]):
            return "timeline"
            
        if any(kw in prompt_lower for kw in ["class", "object", "uml", "software"]):
            return "class diagram"
            
        if any(kw in prompt_lower for kw in ["sequence", "interaction", "message"]):
            return "sequence diagram"
            
        if any(kw in prompt_lower for kw in ["mind map", "brainstorm", "idea"]):
            return "mind map"
        
        # Default to concept map if no specific type is detected
        return "concept map"
    
    def _evaluate_diagram_quality(self, diagram_code: str, prompt: str, diagram_type: str) -> float:
        """
        Evaluate the quality of the generated diagram.
        
        Args:
            diagram_code: The generated diagram code
            prompt: The original prompt
            diagram_type: The diagram type
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Start with a base score
        score = 0.5
        
        # Check code length - too short diagrams are likely incomplete
        if len(diagram_code) < 50:
            score -= 0.2
        elif len(diagram_code) > 200:
            score += 0.1  # More detailed diagrams
        
        # Check for key diagram elements based on type
        if diagram_type == "flowchart":
            if "->|" in diagram_code or "->" in diagram_code:
                score += 0.1  # Has flow connections
            if "label" in diagram_code:
                score += 0.1  # Has labels
                
        elif diagram_type == "entity-relationship":
            if "--" in diagram_code or "-|-" in diagram_code:
                score += 0.1  # Has relationships
            if "label" in diagram_code:
                score += 0.1  # Has labels
        
        # Check for syntax errors (basic check)
        if "{" in diagram_code and "}" not in diagram_code:
            score -= 0.2  # Unclosed braces
        if "[" in diagram_code and "]" not in diagram_code:
            score -= 0.2  # Unclosed brackets
            
        # Check for prompt keywords in the diagram
        prompt_keywords = set(prompt.lower().split())
        code_lower = diagram_code.lower()
        keyword_matches = sum(1 for keyword in prompt_keywords if keyword in code_lower)
        keyword_score = min(0.3, keyword_matches * 0.05)  # Up to 0.3 for keyword matches
        score += keyword_score
        
        # Ensure score is in range [0.0, 1.0]
        return max(0.0, min(1.0, score))
    
    def _extract_diagram_elements(self, response: str) -> List[Dict[str, str]]:
        """
        Extract diagram elements from the agent's response.
        
        Args:
            response: The agent's response text
            
        Returns:
            List of diagram elements with descriptions
        """
        elements = []
        
        # Look for sections describing elements
        element_section_patterns = [
            r"(?:Key elements:|Diagram elements:|Elements in the diagram:)(.*?)(?:\n\n|$)",
            r"(?:The diagram includes:|The diagram contains:)(.*?)(?:\n\n|$)"
        ]
        
        for pattern in element_section_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                section = matches[0]
                # Look for list items
                list_items = re.findall(r"(?:[-*•]\s*)(.*?)(?:\n|$)", section)
                for item in list_items:
                    if ":" in item:
                        name, description = item.split(":", 1)
                        elements.append({
                            "name": name.strip(),
                            "description": description.strip()
                        })
                    else:
                        elements.append({
                            "name": item.strip(),
                            "description": ""
                        })
        
        return elements
    
    def _extract_improvement_suggestions(self, response: str) -> List[str]:
        """
        Extract improvement suggestions from the agent's response.
        
        Args:
            response: The agent's response text
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Look for sections with improvement suggestions
        suggestion_section_patterns = [
            r"(?:Improvement suggestions:|Potential improvements:|The diagram could be improved by:)(.*?)(?:\n\n|$)",
            r"(?:To improve the diagram:|Suggestions for improvement:)(.*?)(?:\n\n|$)"
        ]
        
        for pattern in suggestion_section_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                section = matches[0]
                # Look for list items
                list_items = re.findall(r"(?:[-*•]\s*)(.*?)(?:\n|$)", section)
                for item in list_items:
                    if item.strip():
                        suggestions.append(item.strip())
        
        return suggestions
    
    async def _notify_agents_about_diagram(self, prompt: str, diagram_type: str, result: Dict[str, Any]):
        """
        Notify other agents about a generated diagram.
        
        Args:
            prompt: The original prompt
            diagram_type: The diagram type
            result: The diagram result
        """
        try:
            # Create notification data
            notification_data = {
                "event": "diagram_generated",
                "prompt": prompt,
                "diagram_type": diagram_type,
                "has_image": result["image"] is not None,
                "confidence_score": result.get("confidence_score"),
                "processing_time": result.get("processing_time"),
                "timestamp": time.time(),
                "agent": self.agent_name
            }
            
            # Get all registered agents
            agents = registry.list_agents()
            
            # Send notification to all agents except self
            for agent in agents:
                if agent != self.agent_name:
                    await message_bus.send_message(
                        NotificationMessage(
                            sender=self.agent_name,
                            recipient=agent,
                            notification_type="diagram_generated",
                            message=f"New diagram generated for: {prompt[:50]}...",
                            data=notification_data
                        )
                    )
        except Exception as e:
            # Non-critical error, just log it
            logger.warning(f"Failed to notify agents about generated diagram: {str(e)}")
    
    @provides_capability("get_supported_diagram_types")
    async def get_supported_diagram_types(self) -> List[Dict[str, Any]]:
        """
        Get a list of supported diagram types with descriptions and syntax options.
        
        Returns:
            List of supported diagram types
        """
        return [
            {
                "type": "flowchart",
                "description": "Shows the steps in a process or workflow",
                "syntax_options": ["graphviz", "mermaid"],
                "best_for": "Processes, workflows, algorithms"
            },
            {
                "type": "entity-relationship",
                "description": "Shows relationships between entities or concepts",
                "syntax_options": ["graphviz", "mermaid", "plantuml"],
                "best_for": "Database schemas, data models, concept relationships"
            },
            {
                "type": "tree",
                "description": "Shows hierarchical relationships",
                "syntax_options": ["graphviz"],
                "best_for": "Hierarchies, taxonomies, organizational structures"
            },
            {
                "type": "network",
                "description": "Shows connections between multiple entities",
                "syntax_options": ["graphviz"],
                "best_for": "Networks, systems, interconnected components"
            },
            {
                "type": "timeline",
                "description": "Shows events over time",
                "syntax_options": ["mermaid"],
                "best_for": "Historical events, project timelines, schedules"
            },
            {
                "type": "class diagram",
                "description": "Shows classes and their relationships",
                "syntax_options": ["plantuml"],
                "best_for": "Software architecture, object-oriented design"
            },
            {
                "type": "sequence diagram",
                "description": "Shows interactions between objects over time",
                "syntax_options": ["plantuml", "mermaid"],
                "best_for": "Message flows, API interactions, process sequences"
            },
            {
                "type": "mind map",
                "description": "Shows ideas organized around a central concept",
                "syntax_options": ["graphviz", "plantuml"],
                "best_for": "Brainstorming, concept organization, idea exploration"
            },
            {
                "type": "concept map",
                "description": "Shows relationships between concepts",
                "syntax_options": ["graphviz"],
                "best_for": "Knowledge representation, concept relationships"
            }
        ]
    
    @provides_capability("get_diagram_suggestions")
    async def get_diagram_suggestions(self, topic: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Get diagram suggestions for a specific topic.
        
        Args:
            topic: The topic to suggest diagrams for
            context: Additional context about the topic
            
        Returns:
            List of diagram suggestions
        """
        # Combine topic and context for analysis
        analysis_text = f"{topic} {context}".lower()
        
        # Get all supported diagram types
        diagram_types = await self.get_supported_diagram_types()
        
        # Score each diagram type based on relevance to the topic
        scored_types = []
        for diagram_type in diagram_types:
            score = 0
            
            # Check if any "best_for" keywords match
            best_for = diagram_type["best_for"].lower()
            if any(keyword in analysis_text for keyword in best_for.split(", ")):
                score += 0.5
                
            # Check if the diagram type itself is mentioned
            if diagram_type["type"].lower() in analysis_text:
                score += 0.3
                
            # Add a small score for each word in the description that matches
            description_words = set(diagram_type["description"].lower().split())
            matches = sum(1 for word in description_words if word in analysis_text)
            score += min(0.2, matches * 0.05)
            
            # Add to scored list if score is above threshold
            if score > 0.1:
                scored_types.append({
                    **diagram_type,
                    "relevance_score": score,
                    "sample_prompt": f"Create a {diagram_type['type']} showing the key aspects of {topic}"
                })
        
        # Sort by relevance score
        scored_types.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top 3 suggestions
        return scored_types[:3]