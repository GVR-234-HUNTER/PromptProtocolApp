"""
Structured message formats for agent-to-agent (A2A) communication.

This module defines the message formats used for communication between agents,
ensuring consistent and reliable information exchange.
"""

import uuid
import time
import json
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict

class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"

class Priority(Enum):
    """Priority levels for agent messages."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    """
    Base class for all agent-to-agent messages.
    
    This provides the common structure for all messages exchanged between agents,
    including metadata for routing, tracking, and processing.
    """
    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    
    # Routing information
    sender: str = "unknown"
    recipient: str = "unknown"
    
    # Timing information
    timestamp: float = field(default_factory=time.time)
    timeout_seconds: Optional[float] = None
    
    # Message metadata
    priority: Priority = Priority.MEDIUM
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    
    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        result = asdict(self)
        # Convert enums to their string values
        result["message_type"] = self.message_type.value
        result["priority"] = self.priority.value
        return result
    
    def to_json(self) -> str:
        """Convert the message to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create a message from a dictionary."""
        # Convert string values back to enums
        if "message_type" in data:
            data["message_type"] = MessageType(data["message_type"])
        if "priority" in data:
            data["priority"] = Priority(data["priority"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Create a message from a JSON string."""
        return cls.from_dict(json.loads(json_str))

@dataclass
class RequestMessage(AgentMessage):
    """
    A request from one agent to another.
    
    This message type is used when one agent wants another agent to perform
    an action or provide information.
    """
    message_type: MessageType = MessageType.REQUEST
    
    # The capability being requested
    capability: str = ""
    
    # The method to call on the recipient
    method: str = ""
    
    # Arguments for the method
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    # Whether to wait for a response
    require_response: bool = True

@dataclass
class ResponseMessage(AgentMessage):
    """
    A response to a request.
    
    This message type is used when an agent is responding to a request
    from another agent.
    """
    message_type: MessageType = MessageType.RESPONSE
    
    # The result of the request
    result: Any = None
    
    # Whether the request was successful
    success: bool = True
    
    # Time taken to process the request (in seconds)
    processing_time: Optional[float] = None

@dataclass
class ErrorMessage(AgentMessage):
    """
    An error response to a request.
    
    This message type is used when an agent encounters an error while
    processing a request from another agent.
    """
    message_type: MessageType = MessageType.ERROR
    
    # The error code
    error_code: str = "unknown_error"
    
    # A human-readable error message
    error_message: str = "An unknown error occurred"
    
    # Additional error details
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Whether the error is recoverable
    recoverable: bool = False
    
    # Suggested retry delay (in seconds)
    retry_after: Optional[float] = None

@dataclass
class NotificationMessage(AgentMessage):
    """
    A notification from one agent to another.
    
    This message type is used when an agent wants to inform another agent
    about an event or state change, without expecting a response.
    """
    message_type: MessageType = MessageType.NOTIFICATION
    
    # The type of notification
    notification_type: str = "info"
    
    # The notification message
    message: str = ""
    
    # Additional notification data
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapabilityQueryMessage(AgentMessage):
    """
    A query for agent capabilities.
    
    This message type is used when an agent wants to discover the
    capabilities of another agent.
    """
    message_type: MessageType = MessageType.CAPABILITY_QUERY
    
    # The capability to query for (if empty, query all capabilities)
    capability: Optional[str] = None

@dataclass
class CapabilityResponseMessage(AgentMessage):
    """
    A response to a capability query.
    
    This message type is used when an agent is responding to a capability
    query from another agent.
    """
    message_type: MessageType = MessageType.CAPABILITY_RESPONSE
    
    # The capabilities provided by the agent
    capabilities: List[str] = field(default_factory=list)
    
    # Detailed information about each capability
    capability_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# Helper functions for creating messages

def create_request(
    sender: str,
    recipient: str,
    capability: str,
    method: str,
    arguments: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM,
    conversation_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None
) -> RequestMessage:
    """
    Create a request message.
    
    Args:
        sender: The name of the sending agent
        recipient: The name of the receiving agent
        capability: The capability being requested
        method: The method to call on the recipient
        arguments: Arguments for the method
        priority: The priority of the request
        conversation_id: The ID of the conversation this message is part of
        timeout_seconds: The number of seconds to wait for a response
        
    Returns:
        A RequestMessage
    """
    if arguments is None:
        arguments = {}
    
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    
    return RequestMessage(
        sender=sender,
        recipient=recipient,
        capability=capability,
        method=method,
        arguments=arguments,
        priority=priority,
        conversation_id=conversation_id,
        timeout_seconds=timeout_seconds
    )

def create_response(
    request: RequestMessage,
    result: Any,
    success: bool = True,
    processing_time: Optional[float] = None
) -> ResponseMessage:
    """
    Create a response message.
    
    Args:
        request: The request message being responded to
        result: The result of the request
        success: Whether the request was successful
        processing_time: The time taken to process the request (in seconds)
        
    Returns:
        A ResponseMessage
    """
    return ResponseMessage(
        sender=request.recipient,
        recipient=request.sender,
        result=result,
        success=success,
        processing_time=processing_time,
        conversation_id=request.conversation_id,
        in_reply_to=request.message_id
    )

def create_error(
    request: RequestMessage,
    error_code: str,
    error_message: str,
    error_details: Dict[str, Any] = None,
    recoverable: bool = False,
    retry_after: Optional[float] = None
) -> ErrorMessage:
    """
    Create an error message.
    
    Args:
        request: The request message that resulted in an error
        error_code: The error code
        error_message: A human-readable error message
        error_details: Additional error details
        recoverable: Whether the error is recoverable
        retry_after: Suggested retry delay (in seconds)
        
    Returns:
        An ErrorMessage
    """
    if error_details is None:
        error_details = {}
    
    return ErrorMessage(
        sender=request.recipient,
        recipient=request.sender,
        error_code=error_code,
        error_message=error_message,
        error_details=error_details,
        recoverable=recoverable,
        retry_after=retry_after,
        conversation_id=request.conversation_id,
        in_reply_to=request.message_id
    )

def create_notification(
    sender: str,
    recipient: str,
    notification_type: str,
    message: str,
    data: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM,
    conversation_id: Optional[str] = None
) -> NotificationMessage:
    """
    Create a notification message.
    
    Args:
        sender: The name of the sending agent
        recipient: The name of the receiving agent
        notification_type: The type of notification
        message: The notification message
        data: Additional notification data
        priority: The priority of the notification
        conversation_id: The ID of the conversation this message is part of
        
    Returns:
        A NotificationMessage
    """
    if data is None:
        data = {}
    
    return NotificationMessage(
        sender=sender,
        recipient=recipient,
        notification_type=notification_type,
        message=message,
        data=data,
        priority=priority,
        conversation_id=conversation_id
    )

def create_capability_query(
    sender: str,
    recipient: str,
    capability: Optional[str] = None,
    priority: Priority = Priority.MEDIUM
) -> CapabilityQueryMessage:
    """
    Create a capability query message.
    
    Args:
        sender: The name of the sending agent
        recipient: The name of the receiving agent
        capability: The capability to query for (if None, query all capabilities)
        priority: The priority of the query
        
    Returns:
        A CapabilityQueryMessage
    """
    return CapabilityQueryMessage(
        sender=sender,
        recipient=recipient,
        capability=capability,
        priority=priority,
        conversation_id=str(uuid.uuid4())
    )

def create_capability_response(
    query: CapabilityQueryMessage,
    capabilities: List[str],
    capability_details: Dict[str, Dict[str, Any]] = None
) -> CapabilityResponseMessage:
    """
    Create a capability response message.
    
    Args:
        query: The capability query message being responded to
        capabilities: The capabilities provided by the agent
        capability_details: Detailed information about each capability
        
    Returns:
        A CapabilityResponseMessage
    """
    if capability_details is None:
        capability_details = {}
    
    return CapabilityResponseMessage(
        sender=query.recipient,
        recipient=query.sender,
        capabilities=capabilities,
        capability_details=capability_details,
        conversation_id=query.conversation_id,
        in_reply_to=query.message_id
    )