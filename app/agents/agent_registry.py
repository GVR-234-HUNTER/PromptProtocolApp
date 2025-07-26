"""
Agent Registry for centralized agent management and discovery.

This module provides a registry for all agents in the system, allowing them to discover
and communicate with each other in a structured way. It implements industry-standard
patterns for agent-to-agent (A2A) communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, Union
import inspect
import functools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    A central registry for all agents in the system.
    
    This class provides:
    1. Agent registration and discovery
    2. Structured A2A communication
    3. Capability advertisement
    4. Request routing
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry if not already initialized."""
        if self._initialized:
            return
            
        self._agents = {}  # name -> agent instance
        self._capabilities = {}  # capability name -> list of agent names
        self._schemas = {}  # agent name -> {method name -> parameter schema}
        self._initialized = True
        logger.info("Agent Registry initialized")
    
    def register_agent(self, name: str, agent_instance: Any, capabilities: List[str] = None) -> None:
        """
        Register an agent with the registry.
        
        Args:
            name: The unique name of the agent
            agent_instance: The agent instance
            capabilities: List of capabilities this agent provides
        """
        if name in self._agents:
            logger.warning(f"Agent '{name}' already registered. Updating registration.")
        
        self._agents[name] = agent_instance
        
        # Register capabilities
        if capabilities:
            for capability in capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = []
                if name not in self._capabilities[capability]:
                    self._capabilities[capability].append(name)
        
        # Extract method schemas
        self._schemas[name] = {}
        for method_name, method in inspect.getmembers(agent_instance, predicate=inspect.ismethod):
            if method_name.startswith('_'):
                continue  # Skip private methods
                
            # Get parameter info
            sig = inspect.signature(method)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls'):
                    continue
                param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
                param_default = param.default if param.default != inspect.Parameter.empty else None
                params[param_name] = {
                    'type': param_type,
                    'default': param_default,
                    'required': param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_POSITIONAL
                }
            
            self._schemas[name][method_name] = params
        
        logger.info(f"Registered agent '{name}' with {len(self._schemas[name])} methods")
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            name: The name of the agent to unregister
            
        Returns:
            True if the agent was unregistered, False if it wasn't registered
        """
        if name not in self._agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return False
        
        # Remove from agents dict
        del self._agents[name]
        
        # Remove from capabilities
        for capability, agents in self._capabilities.items():
            if name in agents:
                agents.remove(name)
        
        # Remove schemas
        if name in self._schemas:
            del self._schemas[name]
        
        logger.info(f"Unregistered agent '{name}'")
        return True
    
    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent by name.
        
        Args:
            name: The name of the agent to get
            
        Returns:
            The agent instance or None if not found
        """
        return self._agents.get(name)
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """
        Get all agents that provide a specific capability.
        
        Args:
            capability: The capability to look for
            
        Returns:
            List of agent names that provide the capability
        """
        return self._capabilities.get(capability, [])
    
    def get_agent_schema(self, name: str) -> Dict[str, Any]:
        """
        Get the schema for an agent's methods.
        
        Args:
            name: The name of the agent
            
        Returns:
            Dictionary of method names to parameter schemas
        """
        return self._schemas.get(name, {})
    
    def list_agents(self) -> List[str]:
        """
        List all registered agents.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def list_capabilities(self) -> List[str]:
        """
        List all registered capabilities.
        
        Returns:
            List of capability names
        """
        return list(self._capabilities.keys())
    
    async def call_agent_method(self, agent_name: str, method_name: str, **kwargs) -> Any:
        """
        Call a method on an agent.
        
        Args:
            agent_name: The name of the agent
            method_name: The name of the method to call
            **kwargs: Arguments to pass to the method
            
        Returns:
            The result of the method call
            
        Raises:
            ValueError: If the agent or method is not found
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        method = getattr(agent, method_name, None)
        if not method:
            raise ValueError(f"Method '{method_name}' not found on agent '{agent_name}'")
        
        # Check if the method is async
        if inspect.iscoroutinefunction(method):
            return await method(**kwargs)
        else:
            return method(**kwargs)
    
    async def find_and_call_by_capability(self, capability: str, method_name: str, **kwargs) -> Dict[str, Any]:
        """
        Find agents with a capability and call a method on them.
        
        Args:
            capability: The capability to look for
            method_name: The method to call
            **kwargs: Arguments to pass to the method
            
        Returns:
            Dictionary of agent name to result
        """
        agents = self.get_agents_by_capability(capability)
        if not agents:
            logger.warning(f"No agents found with capability '{capability}'")
            return {}
        
        results = {}
        for agent_name in agents:
            try:
                result = await self.call_agent_method(agent_name, method_name, **kwargs)
                results[agent_name] = result
            except Exception as e:
                logger.error(f"Error calling {method_name} on {agent_name}: {str(e)}")
                results[agent_name] = {"error": str(e)}
        
        return results

# Create a singleton instance
registry = AgentRegistry()

# Decorator for registering agent methods as capabilities
def provides_capability(capability_name: str):
    """
    Decorator to register a method as providing a capability.
    
    Args:
        capability_name: The name of the capability
        
    Returns:
        Decorated method
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        
        # Mark the method with the capability
        wrapper._capability = capability_name
        return wrapper
    
    return decorator

# Helper function to register all capabilities from an agent class
def register_agent_with_capabilities(agent_instance: Any, agent_name: str = None):
    """
    Register an agent and all its capabilities with the registry.
    
    Args:
        agent_instance: The agent instance to register
        agent_name: Optional name for the agent (defaults to class name)
    """
    if agent_name is None:
        agent_name = agent_instance.__class__.__name__
    
    # Find all methods with capability markers
    capabilities = []
    for method_name, method in inspect.getmembers(agent_instance, predicate=inspect.ismethod):
        if hasattr(method, '_capability'):
            capabilities.append(method._capability)
    
    # Register the agent
    registry.register_agent(agent_name, agent_instance, capabilities)
    return agent_instance