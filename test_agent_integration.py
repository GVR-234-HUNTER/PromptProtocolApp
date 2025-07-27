#!/usr/bin/env python3
"""
Test script to verify agent integration and A2A communication capabilities.
This script tests that all agents can be instantiated and registered properly.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_agent_integration():
    """Test the integration of all agents with the registry and message bus."""
    
    try:
        # Import the agents and infrastructure
        from agents.chatbot_agent_adk import ChatbotAgentADK
        from agents.diagram_agent_adk import DiagramAgentADK
        from agents.worksheet_agent_adk import WorksheetAgentADK
        from agents.agent_registry import registry
        from agents.message_bus import message_bus
        
        logger.info("Successfully imported all agent modules")
        
        # Start the message bus
        await message_bus.start()
        logger.info("Message bus started successfully")
        
        # Test agent instantiation
        logger.info("Testing agent instantiation...")
        
        # Create chatbot agent
        chatbot_agent = ChatbotAgentADK(agent_name="test_chatbot")
        logger.info("Chatbot agent created successfully")
        
        # Create diagram agent
        diagram_agent = DiagramAgentADK(agent_name="test_diagram")
        logger.info("Diagram agent created successfully")
        
        # Create worksheet agent
        worksheet_agent = WorksheetAgentADK(agent_name="test_worksheet")
        logger.info("Worksheet agent created successfully")
        
        # Wait a moment for async registration to complete
        await asyncio.sleep(1)
        
        # Test agent registry
        logger.info("Testing agent registry...")
        
        registered_agents = registry.list_agents()
        logger.info(f"Registered agents: {registered_agents}")
        
        # Test capabilities
        capabilities = registry.list_capabilities()
        logger.info(f"Available capabilities: {capabilities}")
        
        # Test specific agent retrieval
        chatbot = registry.get_agent("test_chatbot")
        diagram = registry.get_agent("test_diagram")
        worksheet = registry.get_agent("test_worksheet")
        
        if chatbot and diagram and worksheet:
            logger.info("All agents successfully registered and retrievable")
        else:
            logger.error("Some agents failed to register properly")
            return False
        
        # Test capability-based discovery
        chat_agents = registry.get_agents_by_capability("chat")
        diagram_agents = registry.get_agents_by_capability("generate_diagram")
        worksheet_agents = registry.get_agents_by_capability("generate_worksheet")
        
        logger.info(f"Chat capable agents: {chat_agents}")
        logger.info(f"Diagram capable agents: {diagram_agents}")
        logger.info(f"Worksheet capable agents: {worksheet_agents}")
        
        # Test message bus stats
        stats = message_bus.get_stats()
        logger.info(f"Message bus stats: {stats}")
        
        logger.info("All integration tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}", exc_info=True)
        return False
    
    finally:
        # Clean up
        try:
            await message_bus.stop()
            logger.info("Message bus stopped")
        except:
            pass

async def test_basic_agent_communication():
    """Test basic communication between agents."""
    
    try:
        from agents.agent_messages import create_notification, Priority
        from agents.message_bus import message_bus
        
        logger.info("Testing basic agent communication...")
        
        # Create a test notification
        notification = create_notification(
            sender="test_sender",
            recipient="test_recipient",
            notification_type="test_message",
            message="This is a test message",
            data={"test_key": "test_value"},
            priority=Priority.MEDIUM
        )
        
        logger.info(f"Created test notification: {notification.to_dict()}")
        
        # Send the message through the bus
        message_bus.send_message(notification)
        logger.info("Test message sent successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Communication test failed: {str(e)}", exc_info=True)
        return False

async def main():
    """Main test function."""
    
    logger.info("Starting agent integration tests...")
    
    # Test 1: Agent Integration
    integration_success = await test_agent_integration()
    
    # Test 2: Basic Communication
    communication_success = await test_basic_agent_communication()
    
    # Summary
    if integration_success and communication_success:
        logger.info("🎉 All tests passed! Agent system is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)