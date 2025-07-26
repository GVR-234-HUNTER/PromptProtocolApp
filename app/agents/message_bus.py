"""
Message Bus for agent-to-agent (A2A) communication.

This module provides a centralized message bus for routing messages between agents,
implementing industry-standard patterns for reliable and scalable agent communication.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set, Tuple
import uuid
import traceback

from app.agents.agent_messages import (
    AgentMessage, RequestMessage, ResponseMessage, ErrorMessage,
    NotificationMessage, CapabilityQueryMessage, CapabilityResponseMessage,
    MessageType, Priority, create_error
)
from app.agents.agent_registry import registry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageBus:
    """
    A centralized message bus for routing messages between agents.
    
    This class provides:
    1. Message routing and delivery
    2. Subscription-based message handling
    3. Request-response correlation
    4. Message queuing and prioritization
    5. Error handling and recovery
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one message bus exists."""
        if cls._instance is None:
            cls._instance = super(MessageBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the message bus if not already initialized."""
        if self._initialized:
            return
            
        # Message handlers by recipient and message type
        self._handlers: Dict[str, Dict[MessageType, List[Callable[[AgentMessage], Awaitable[None]]]]] = {}
        
        # Pending requests waiting for responses
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Message queues by priority
        self._queues: Dict[Priority, asyncio.Queue] = {
            Priority.LOW: asyncio.Queue(),
            Priority.MEDIUM: asyncio.Queue(),
            Priority.HIGH: asyncio.Queue(),
            Priority.CRITICAL: asyncio.Queue()
        }
        
        # Set of active conversations
        self._active_conversations: Set[str] = set()
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "dropped_messages": 0
        }
        
        # Start the message processing tasks
        self._processing_tasks = []
        self._shutdown_event = asyncio.Event()
        self._initialized = True
        logger.info("Message Bus initialized")
    
    async def start(self):
        """Start the message bus processing tasks."""
        if self._processing_tasks:
            logger.warning("Message Bus already started")
            return
        
        # Start a task for each priority queue
        for priority in Priority:
            task = asyncio.create_task(self._process_queue(priority))
            self._processing_tasks.append(task)
        
        logger.info("Message Bus started")
    
    async def stop(self):
        """Stop the message bus processing tasks."""
        if not self._processing_tasks:
            logger.warning("Message Bus not running")
            return
        
        # Signal tasks to stop
        self._shutdown_event.set()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._processing_tasks)
        self._processing_tasks = []
        
        logger.info("Message Bus stopped")
    
    async def _process_queue(self, priority: Priority):
        """
        Process messages from a priority queue.
        
        Args:
            priority: The priority queue to process
        """
        queue = self._queues[priority]
        
        while not self._shutdown_event.is_set():
            try:
                # Get the next message from the queue
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Process the message
                await self._deliver_message(message)
                
                # Mark the task as done
                queue.task_done()
            except asyncio.TimeoutError:
                # No message available, continue
                continue
            except Exception as e:
                logger.error(f"Error processing message from {priority.value} queue: {str(e)}")
                self._stats["errors"] += 1
    
    async def _deliver_message(self, message: AgentMessage):
        """
        Deliver a message to its recipient.
        
        Args:
            message: The message to deliver
        """
        recipient = message.recipient
        message_type = message.message_type
        
        # Check if there are handlers for this recipient and message type
        if recipient in self._handlers and message_type in self._handlers[recipient]:
            handlers = self._handlers[recipient][message_type]
            
            # Call all handlers for this message
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for {recipient}: {str(e)}")
                    self._stats["errors"] += 1
        else:
            logger.warning(f"No handler for message type {message_type.value} to {recipient}")
            self._stats["dropped_messages"] += 1
        
        # If this is a response or error message, resolve the pending request
        if message.in_reply_to and message.in_reply_to in self._pending_requests:
            future = self._pending_requests[message.in_reply_to]
            if not future.done():
                future.set_result(message)
            del self._pending_requests[message.in_reply_to]
    
    def register_handler(self, recipient: str, message_type: MessageType, 
                         handler: Callable[[AgentMessage], Awaitable[None]]):
        """
        Register a handler for a specific recipient and message type.
        
        Args:
            recipient: The name of the recipient agent
            message_type: The type of message to handle
            handler: The handler function to call when a message is received
        """
        if recipient not in self._handlers:
            self._handlers[recipient] = {}
        
        if message_type not in self._handlers[recipient]:
            self._handlers[recipient][message_type] = []
        
        self._handlers[recipient][message_type].append(handler)
        logger.info(f"Registered handler for {message_type.value} messages to {recipient}")
    
    def unregister_handler(self, recipient: str, message_type: MessageType, 
                           handler: Callable[[AgentMessage], Awaitable[None]]) -> bool:
        """
        Unregister a handler for a specific recipient and message type.
        
        Args:
            recipient: The name of the recipient agent
            message_type: The type of message to handle
            handler: The handler function to remove
            
        Returns:
            True if the handler was unregistered, False if it wasn't registered
        """
        if (recipient in self._handlers and 
            message_type in self._handlers[recipient] and 
            handler in self._handlers[recipient][message_type]):
            self._handlers[recipient][message_type].remove(handler)
            logger.info(f"Unregistered handler for {message_type.value} messages to {recipient}")
            return True
        
        logger.warning(f"Handler for {message_type.value} messages to {recipient} not found")
        return False
    
    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to its recipient.
        
        Args:
            message: The message to send
        """
        # Add the message to the appropriate queue
        await self._queues[message.priority].put(message)
        self._stats["messages_sent"] += 1
        
        # If this is a request that starts a new conversation, add it to active conversations
        if message.message_type == MessageType.REQUEST and message.conversation_id:
            self._active_conversations.add(message.conversation_id)
    
    async def send_request(self, request: RequestMessage, timeout: Optional[float] = None) -> Tuple[bool, AgentMessage]:
        """
        Send a request and wait for a response.
        
        Args:
            request: The request message to send
            timeout: The maximum time to wait for a response (in seconds)
            
        Returns:
            A tuple of (success, response) where success is True if a response was received
            and response is the response message (or None if no response was received)
        """
        # Create a future to wait for the response
        future = asyncio.Future()
        self._pending_requests[request.message_id] = future
        
        # Send the request
        await self.send_message(request)
        
        try:
            # Wait for the response
            response = await asyncio.wait_for(future, timeout=timeout or request.timeout_seconds)
            return True, response
        except asyncio.TimeoutError:
            # No response received within the timeout
            del self._pending_requests[request.message_id]
            logger.warning(f"Request {request.message_id} timed out")
            return False, None
        except Exception as e:
            # Error waiting for response
            del self._pending_requests[request.message_id]
            logger.error(f"Error waiting for response to {request.message_id}: {str(e)}")
            return False, None
    
    async def request_capability(self, sender: str, recipient: str, capability: str, 
                                method: str, arguments: Dict[str, Any] = None,
                                timeout: Optional[float] = None) -> Tuple[bool, Any, Optional[str]]:
        """
        Send a request for a capability and wait for a response.
        
        Args:
            sender: The name of the sending agent
            recipient: The name of the receiving agent
            capability: The capability being requested
            method: The method to call on the recipient
            arguments: Arguments for the method
            timeout: The maximum time to wait for a response (in seconds)
            
        Returns:
            A tuple of (success, result, error) where success is True if the request was successful,
            result is the result of the request (if successful), and error is an error message (if unsuccessful)
        """
        # Create a request message
        request = RequestMessage(
            sender=sender,
            recipient=recipient,
            capability=capability,
            method=method,
            arguments=arguments or {},
            conversation_id=str(uuid.uuid4()),
            timeout_seconds=timeout
        )
        
        # Send the request and wait for a response
        success, response = await self.send_request(request, timeout)
        
        if not success:
            return False, None, "Request timed out"
        
        # Check if the response is an error
        if response.message_type == MessageType.ERROR:
            error_msg = response.error_message if hasattr(response, 'error_message') else "Unknown error"
            return False, None, error_msg
        
        # Extract the result from the response
        result = response.result if hasattr(response, 'result') else None
        return True, result, None
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the message bus.
        
        Returns:
            A dictionary of statistics
        """
        return self._stats.copy()
    
    def get_active_conversations(self) -> Set[str]:
        """
        Get the set of active conversation IDs.
        
        Returns:
            A set of active conversation IDs
        """
        return self._active_conversations.copy()

# Create a singleton instance
message_bus = MessageBus()

# Helper function to register an agent with the message bus
async def register_agent_with_message_bus(agent_name: str, agent_instance: Any):
    """
    Register an agent with the message bus.
    
    This function:
    1. Registers the agent with the registry
    2. Creates message handlers for the agent
    3. Starts the message bus if it's not already running
    
    Args:
        agent_name: The name of the agent
        agent_instance: The agent instance
    """
    # Register the agent with the registry
    registry.register_agent(agent_name, agent_instance)
    
    # Create a handler for request messages
    async def handle_request(message: RequestMessage):
        """Handle a request message for the agent."""
        try:
            # Get the method to call
            method_name = message.method
            method = getattr(agent_instance, method_name, None)
            
            if not method:
                # Method not found
                error_msg = create_error(
                    request=message,
                    error_code="method_not_found",
                    error_message=f"Method '{method_name}' not found on agent '{agent_name}'",
                    recoverable=False
                )
                await message_bus.send_message(error_msg)
                return
            
            # Call the method
            start_time = time.time()
            result = await registry.call_agent_method(agent_name, method_name, **message.arguments)
            processing_time = time.time() - start_time
            
            # Create a response message
            response = ResponseMessage(
                sender=agent_name,
                recipient=message.sender,
                result=result,
                success=True,
                processing_time=processing_time,
                conversation_id=message.conversation_id,
                in_reply_to=message.message_id
            )
            
            # Send the response
            await message_bus.send_message(response)
        except Exception as e:
            # Create an error message
            error_msg = create_error(
                request=message,
                error_code="execution_error",
                error_message=str(e),
                error_details={"traceback": traceback.format_exc()},
                recoverable=False
            )
            await message_bus.send_message(error_msg)
    
    # Create a handler for capability query messages
    async def handle_capability_query(message: CapabilityQueryMessage):
        """Handle a capability query message for the agent."""
        try:
            # Get the agent's capabilities
            schema = registry.get_agent_schema(agent_name)
            capabilities = list(schema.keys())
            
            # Create a capability details dictionary
            capability_details = {}
            for method_name, params in schema.items():
                capability_details[method_name] = {
                    "parameters": params,
                    "description": getattr(getattr(agent_instance, method_name, None), "__doc__", "")
                }
            
            # Create a response message
            response = CapabilityResponseMessage(
                sender=agent_name,
                recipient=message.sender,
                capabilities=capabilities,
                capability_details=capability_details,
                conversation_id=message.conversation_id,
                in_reply_to=message.message_id
            )
            
            # Send the response
            await message_bus.send_message(response)
        except Exception as e:
            # Create an error message
            error_msg = create_error(
                request=message,
                error_code="capability_query_error",
                error_message=str(e),
                error_details={"traceback": traceback.format_exc()},
                recoverable=False
            )
            await message_bus.send_message(error_msg)
    
    # Register the handlers with the message bus
    message_bus.register_handler(agent_name, MessageType.REQUEST, handle_request)
    message_bus.register_handler(agent_name, MessageType.CAPABILITY_QUERY, handle_capability_query)
    
    # Start the message bus if it's not already running
    if not message_bus._processing_tasks:
        await message_bus.start()
    
    logger.info(f"Agent '{agent_name}' registered with message bus")

# Helper function to unregister an agent from the message bus
async def unregister_agent_from_message_bus(agent_name: str):
    """
    Unregister an agent from the message bus.
    
    Args:
        agent_name: The name of the agent to unregister
    """
    # Unregister the agent from the registry
    registry.unregister_agent(agent_name)
    
    # Remove all handlers for this agent
    for message_type in MessageType:
        if agent_name in message_bus._handlers and message_type in message_bus._handlers[agent_name]:
            del message_bus._handlers[agent_name][message_type]
    
    logger.info(f"Agent '{agent_name}' unregistered from message bus")