# Educational Chatbot Implementation Summary

## Overview

I've implemented an educational chatbot agent using Google's Agent Development Kit (ADK) that meets all the requirements specified. The chatbot:

1. Accepts questions from users and responds to them
2. Stores conversation context and uses previous interactions to refine future answers
3. Calls the diagram generator agent if the user asks 3 consecutive questions on the same topic
4. Only responds to educational questions restricted to the syllabus
5. Ensures only safe content is displayed (18-)

## Implementation Details

### Files Created/Modified

1. **Agent Implementation**:
   - `/app/agents/chatbot_agent_adk.py`: The main chatbot agent implementation

2. **API Endpoints**:
   - `/app/api/chatbot_adk.py`: FastAPI router for the chatbot agent

3. **Data Models**:
   - Updated `/app/models.py` with new models for chatbot requests

4. **Application Entry Point**:
   - Updated `/app/main_adk.py` to include the chatbot router

5. **Documentation**:
   - Updated `/README_ADK.md` with chatbot information and usage examples

### Key Features

1. **Context Retention**
   - The chatbot maintains conversation context across multiple messages using ADK's session management
   - Each user gets a unique session that stores their conversation history and state

2. **Topic Tracking**
   - The chatbot extracts topics from user messages
   - It keeps track of topics in the session state
   - It can detect when a user is asking multiple questions about the same topic

3. **Automatic Diagram Generation**
   - If a user asks 3 consecutive questions about the same topic, the chatbot automatically generates a diagram
   - This is implemented through Agent-to-Agent (A2A) communication with the diagram agent

4. **Content Safety**
   - The chatbot checks if messages contain safe content using a keyword-based filter
   - It only responds to safe, educational content appropriate for students

5. **Syllabus Restriction**
   - The chatbot only answers questions that are educational and within the specified syllabus
   - It politely declines to answer non-educational questions

### API Endpoints

The chatbot provides three endpoints:

1. `/api/chatbot/create_session`: Creates a new chat session for a user
2. `/api/chatbot/chat`: Processes a message within an existing session
3. `/api/chatbot/simple_chat`: Creates a session automatically and processes a single message

## How to Test

To test the chatbot:

1. Install the required packages:
   ```bash
   pip install google-adk litellm google-generativeai fastapi uvicorn python-multipart python-dotenv pillow requests
   ```

2. Set up your API keys in the `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Start the server:
   ```bash
   python -m app.main_adk
   ```

4. Use the Swagger UI at `http://localhost:8000/docs` to interact with the API

5. Test the following scenarios:
   - Basic Q&A functionality with educational questions
   - Context retention across multiple messages
   - Topic repetition detection and diagram generation
   - Content safety filters with inappropriate questions
   - Syllabus restriction with non-educational questions

## GENTIC Implementation

The chatbot implementation follows GENTIC principles:

1. **Multi-Agent Communication (MCP/A2A)**
   - The chatbot communicates with the diagram agent to generate visual explanations
   - This is implemented through direct function calls to maintain simplicity

2. **Context Management**
   - The chatbot maintains conversation context using ADK's session management
   - It tracks topics, question counts, and user history

3. **Safety and Appropriateness**
   - Content safety filters ensure all responses are appropriate for students
   - Syllabus restrictions keep the conversation focused on educational content

4. **Adaptive Responses**
   - The chatbot adapts its responses based on conversation history
   - It automatically generates diagrams when it detects the user needs additional help

## Conclusion

The implemented chatbot agent meets all the requirements specified in the issue description. It leverages Google's ADK for structured agent development, maintains conversation context, tracks topics, generates diagrams when needed, and ensures content safety and syllabus restrictions.

The implementation is modular, well-documented, and follows the same pattern as the existing worksheet and diagram agents, making it easy to understand and extend.