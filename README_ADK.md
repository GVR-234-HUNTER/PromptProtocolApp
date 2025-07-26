# Sahayak Teaching Platform - ADK Version

This is an enhanced version of the Sahayak Teaching Platform that uses Google's Agent Development Kit (ADK) for implementing intelligent agents. The platform provides three main services:

1. **Worksheet Generator**: Creates educational worksheets based on textbook images
2. **Diagram Generator**: Creates diagrams based on text descriptions
3. **Educational Chatbot**: Answers educational questions within the syllabus, maintains conversation context, and generates diagrams when needed

## New ADK Implementation

This version has been rewritten to use Google's Agent Development Kit (ADK), which provides a structured framework for building, managing, and deploying AI agents. The ADK implementation offers several advantages:

- Better conversation management through session services
- Structured agent definition with clear tools and instructions
- Support for multiple LLM providers through LiteLLM integration
- Improved error handling and state management
- Asynchronous execution model

## Installation

### Prerequisites

- Python 3.9+
- pip

### Required Packages

```bash
# Core dependencies
pip install fastapi uvicorn python-multipart python-dotenv pillow requests

# ADK and LLM dependencies
pip install google-adk litellm google-generativeai
```

All the required packages for the chatbot agent are already included in the above list. The chatbot uses the same core dependencies and ADK/LLM dependencies as the other agents.

### API Keys

You'll need to set up the following API keys in your `.env` file:

```
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional (for multi-model support)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

You can get these API keys from:
- Gemini API Key: [Google AI Studio](https://aistudio.google.com/app/apikey)
- OpenAI API Key: [OpenAI Platform](https://platform.openai.com/api-keys)
- Anthropic API Key: [Anthropic Console](https://console.anthropic.com/settings/keys)

## Running the Application

To run the ADK version of the application:

```bash
# From the project root directory
python -m app.main_adk
```

The server will start at `http://localhost:8000`.

## API Endpoints

### Worksheet Generator

```
POST /api/worksheet/generate
```

Parameters (form data):
- `images`: List of image files (jpg, jpeg, png)
- `grade_level`: Target grade level (default: "5")
- `difficulty`: Difficulty level (easy, medium, hard) (default: "medium")
- `mcq_count`: Number of multiple choice questions (default: 4)
- `short_answer_count`: Number of short answer questions (default: 3)
- `fill_blank_count`: Number of fill-in-the-blank questions (default: 2)
- `true_false_count`: Number of true/false questions (default: 1)

### Diagram Generator

```
POST /api/diagram/generate
```

Parameters (JSON):
- `prompt`: Description of the diagram to generate
- `code_style`: Diagram syntax to use (default: "graphviz")
- `output_format`: Output format (default: "pdf")

### Educational Chatbot

The chatbot provides three endpoints for different use cases:

```
POST /api/chatbot/simple_chat
```

Parameters (JSON):
- `message`: The user's message or question
- `syllabus`: The syllabus context (e.g., "Math", "Science", "History") (default: "General")
- `grade_level`: Target grade level for the response (default: "5")

This endpoint creates a new session automatically and processes a single message. It's useful for simple interactions but doesn't maintain conversation context across multiple API calls.

```
POST /api/chatbot/create_session
```

Parameters (JSON):
- `user_id`: The user's ID for session management

This endpoint creates a new chat session for a user and returns a session ID that can be used for subsequent messages.

```
POST /api/chatbot/chat
```

Parameters (JSON):
- `user_id`: The user's ID for session management
- `session_id`: The session ID for context tracking
- `message`: The user's message or question
- `syllabus`: The syllabus context (e.g., "Math", "Science", "History") (default: "General")
- `grade_level`: Target grade level for the response (default: "5")

This endpoint processes a message within an existing session, maintaining conversation context across multiple messages. It's useful for extended conversations where context is important.

## Architecture

The ADK implementation follows this architecture:

1. **Tools**: Python functions with clear docstrings that define the agent's capabilities
2. **Agents**: ADK Agent instances that combine LLMs with tools and instructions
3. **Session Services**: Manage conversation history and state
4. **Runners**: Orchestrate the execution of agents and handle events

### Key Components

- `WorksheetAgentADK`: Generates worksheets based on textbook images
- `DiagramAgentADK`: Generates diagrams based on text descriptions
- `ChatbotAgentADK`: Educational chatbot that answers questions within the syllabus
- `InMemorySessionService`: Manages conversation history and state
- `Runner`: Orchestrates agent execution

### Chatbot Features

The Educational Chatbot includes several advanced features:

1. **Context Retention**: The chatbot maintains conversation context across multiple messages, allowing it to provide more relevant and personalized responses.

2. **Topic Tracking**: The chatbot tracks the topics of user questions and can detect when a user is asking multiple questions about the same topic.

3. **Automatic Diagram Generation**: If a user asks three consecutive questions about the same topic, the chatbot automatically generates a diagram to help explain the concept visually.

4. **Content Safety**: The chatbot ensures that all content is safe and appropriate for students (18-), filtering out inappropriate questions and content.

5. **Syllabus Restriction**: The chatbot only answers educational questions that are within the specified syllabus, helping to keep conversations focused on educational content.

6. **Agent-to-Agent Communication**: The chatbot can communicate with the Diagram Agent to generate visual explanations when needed.

## Differences from Original Implementation

The ADK implementation differs from the original in several ways:

1. **Asynchronous API**: All endpoints are now async for better performance
2. **Session Management**: Conversations are managed through ADK's session service
3. **Tool Definition**: Agent capabilities are defined as standalone functions with clear docstrings
4. **Multi-Model Support**: Can use different LLM providers through LiteLLM integration
5. **Event-Based Execution**: Agent execution follows an event-based model

## Testing

To test the implementation:

1. Start the server: `python -m app.main_adk`
2. Use the Swagger UI at `http://localhost:8000/docs` to interact with the API
3. Try generating worksheets and diagrams with different parameters
4. Test the chatbot with educational questions

## Usage Examples

### Chatbot Example

Here's an example of how to use the chatbot API for a conversation:

1. First, create a session:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/chatbot/create_session' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user123"
}'
```

Response:
```json
{
  "success": true,
  "session_id": "session_a1b2c3d4",
  "message": "Session created successfully"
}
```

2. Then, send a message using the session:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/chatbot/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user123",
  "session_id": "session_a1b2c3d4",
  "message": "What is photosynthesis?",
  "syllabus": "Science",
  "grade_level": "5"
}'
```

Response:
```json
{
  "success": true,
  "answer": "Photosynthesis is the process plants use to make their own food. Plants take in water from the soil through their roots and carbon dioxide from the air through tiny holes in their leaves called stomata. Using energy from sunlight, plants convert water and carbon dioxide into glucose (sugar) and oxygen. The sugar is used by the plant for energy and growth, while the oxygen is released into the air for us to breathe. This process is very important because it provides food for plants and oxygen for animals and humans.",
  "diagram": null,
  "topic": "what is photosynthesis",
  "error": null
}
```

3. If you ask about the same topic multiple times, the chatbot will generate a diagram:

```bash
# After sending two more questions about photosynthesis...
curl -X 'POST' \
  'http://localhost:8000/api/chatbot/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user123",
  "session_id": "session_a1b2c3d4",
  "message": "Can you explain the steps of photosynthesis again?",
  "syllabus": "Science",
  "grade_level": "5"
}'
```

Response:
```json
{
  "success": true,
  "answer": "I notice you've asked several questions about photosynthesis. Let me explain it again with a helpful diagram.\n\nPhotosynthesis happens in these steps:\n\n1. Plants absorb water through their roots\n2. They take in carbon dioxide through tiny holes in their leaves called stomata\n3. Chlorophyll in the leaves captures energy from sunlight\n4. Using this energy, plants convert water and carbon dioxide into glucose (sugar) and oxygen\n5. The sugar is used by the plant for food and growth\n6. Oxygen is released into the air\n\nThe diagram should help you visualize this process better!",
  "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "topic": "steps of photosynthesis",
  "error": null
}
```

4. For a simple one-off interaction, you can use the simple_chat endpoint:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/chatbot/simple_chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "What is the water cycle?",
  "syllabus": "Science",
  "grade_level": "5"
}'
```

## Troubleshooting

If you encounter issues:

1. Check that all required API keys are set in your `.env` file
2. Ensure all dependencies are installed
3. Check the server logs for error messages
4. Verify that the input parameters are correct

## Future Improvements

Potential future improvements include:

1. Implementing persistent session storage (e.g., database)
2. Adding more LLM providers
3. Implementing more sophisticated error handling and retry mechanisms
4. Adding user authentication and rate limiting
5. Implementing caching for improved performance