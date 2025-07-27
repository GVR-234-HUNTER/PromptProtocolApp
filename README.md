# PromptProtocolApp - Industry-Grade Agent System

## Overview

PromptProtocolApp is a sophisticated educational platform featuring industry-grade AI agents that demonstrate advanced agentic capabilities, agent-to-agent communication, and robust error handling. The system consists of three main agents: Chatbot, Diagram, and Worksheet agents, all built using Google's Agent Development Kit (ADK).

## Key Features

### 🤖 Advanced Agentic Capabilities
- **Autonomous Decision Making**: Agents independently choose response strategies and tool usage
- **Self-Evaluation**: Built-in quality assessment with confidence scoring
- **Adaptive Behavior**: Dynamic response style adaptation based on user patterns
- **Proactive Intelligence**: Context-aware suggestions and learning path recommendations
- **Continuous Learning**: Pattern recognition and optimization from user interactions

### 🔗 Agent-to-Agent (A2A) Communication
- **Central Agent Registry**: Unified agent discovery and capability management
- **Message Bus Infrastructure**: Priority-based asynchronous messaging
- **Structured Communication**: Comprehensive message types with conversation tracking
- **Bidirectional Coordination**: Full-duplex communication enabling collaborative workflows
- **Capability Discovery**: Dynamic agent capability querying and method invocation

### 🛡️ Production-Grade Resilience
- **Circuit Breaker Pattern**: Automatic failure detection with configurable thresholds
- **Graceful Degradation**: Intelligent fallback responses maintaining user experience
- **Enhanced Error Reporting**: Detailed diagnostics with specific recovery suggestions
- **Retry Mechanisms**: Exponential backoff with comprehensive timeout management
- **System Observability**: Comprehensive logging, metrics, and performance monitoring

## Architecture

The system follows a modular, microservices-inspired architecture:

```
PromptProtocolApp/
├── app/
│   ├── agents/                 # Core agent implementations
│   │   ├── chatbot_agent_adk.py      # Advanced educational chatbot
│   │   ├── diagram_agent_adk.py      # Intelligent diagram generator
│   │   ├── worksheet_agent_adk.py    # Comprehensive worksheet creator
│   │   ├── agent_registry.py         # Central agent discovery
│   │   ├── message_bus.py            # A2A communication infrastructure
│   │   └── agent_messages.py         # Structured message formats
│   ├── api/                    # FastAPI endpoints
│   └── utils/                  # Utility functions
└── test_agent_integration.py   # Integration testing framework
```

## Agents Overview

### ChatbotAgentADK
- **Purpose**: Educational Q&A with advanced NLP capabilities
- **Key Features**: 
  - Advanced topic extraction with subject-specific weighting
  - Multi-factor question complexity analysis
  - Autonomous decision-making for response strategies
  - Proactive learning path suggestions
  - Adaptive response style based on user patterns

### DiagramAgentADK
- **Purpose**: Intelligent diagram generation for educational content
- **Key Features**:
  - Context-aware diagram type selection
  - Quality evaluation and improvement suggestions
  - Intelligent caching with hash-based keys
  - Cross-agent collaboration for enhanced learning

### WorksheetAgentADK
- **Purpose**: Comprehensive worksheet generation from textbook images
- **Key Features**:
  - AI-powered question generation and quality assessment
  - Question distribution analysis and optimization
  - Topic-based worksheet recommendations
  - Integration with other agents for collaborative learning

## Installation

### Prerequisites
- Python 3.11 or higher
- Google ADK credentials and setup
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PromptProtocolApp
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

4. **Configure Google ADK**
   - Set up your Google Cloud project
   - Enable necessary APIs (Gemini, etc.)
   - Configure authentication credentials
   - Set environment variables as required by ADK

5. **Environment Variables**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   export PROJECT_ID="your-google-cloud-project-id"
   # Add other required environment variables
   ```

## Usage

### Starting the Application

1. **Run the FastAPI server**
   ```bash
   cd app
   uvicorn main_adk:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the application**
   - Web interface: `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

### Testing the System

1. **Run integration tests**
   ```bash
   python test_agent_integration.py
   ```

2. **Test individual agents**
   ```bash
   # Test chatbot functionality
   curl -X POST "http://localhost:8000/chatbot_adk" \
        -H "Content-Type: application/json" \
        -d '{"message": "What is photosynthesis?", "grade_level": "5"}'

   # Test diagram generation
   curl -X POST "http://localhost:8000/diagram_adk" \
        -H "Content-Type: application/json" \
        -d '{"user_prompt": "Create a diagram of the water cycle"}'
   ```

## API Endpoints

### Chatbot Agent
- **POST** `/chatbot_adk` - Educational Q&A with advanced capabilities
- **POST** `/chatbot_adk/session` - Create new learning session

### Diagram Agent
- **POST** `/diagram_adk` - Generate educational diagrams
- **GET** `/diagram_adk/types` - Get supported diagram types

### Worksheet Agent
- **POST** `/worksheet_adk` - Generate worksheets from textbook images
- **GET** `/worksheet_adk/question-types` - Get supported question types

## Configuration

### Agent Configuration
Each agent can be configured with various parameters:

```python
# Example agent initialization
chatbot_agent = ChatbotAgentADK(
    model="gemini-1.5-flash",
    agent_name="educational_chatbot"
)
```

### Message Bus Configuration
The message bus supports priority-based messaging:

```python
# Priority levels: HIGH, MEDIUM, LOW
# Timeout configurations
# Retry mechanisms
```

## Development

### Adding New Agents
1. Inherit from the base agent pattern
2. Implement required capabilities
3. Register with the agent registry
4. Add A2A communication handlers
5. Implement error handling and metrics

### Extending Capabilities
1. Use the `@provides_capability` decorator
2. Follow the established message patterns
3. Implement proper error handling
4. Add comprehensive logging

## Monitoring and Observability

### Metrics Tracking
Each agent tracks comprehensive metrics:
- Request counts and success rates
- Response times and performance trends
- Error rates and types
- Circuit breaker activations
- Cache hit rates
- User engagement patterns

### Logging
- Structured logging with appropriate levels
- Request/response tracking
- Error diagnostics
- Performance monitoring

## Troubleshooting

### Common Issues

1. **Agent Registration Failures**
   - Check message bus is started
   - Verify agent initialization order
   - Review error logs for specific issues

2. **ADK Authentication Issues**
   - Verify Google Cloud credentials
   - Check project permissions
   - Ensure required APIs are enabled

3. **Performance Issues**
   - Monitor circuit breaker status
   - Check cache hit rates
   - Review timeout configurations

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the established architectural patterns
2. Implement comprehensive error handling
3. Add appropriate tests and documentation
4. Ensure agentic behavior principles are maintained
5. Follow code quality standards

## License

[Add your license information here]

## Support

For support and questions:
- Review the architecture.md for technical details
- Check implementation_summary.md for implementation insights
- Create issues for bugs or feature requests
- Follow the troubleshooting guide for common problems

---

**Note**: This is an industry-grade agent system designed to demonstrate sophisticated AI agent capabilities. The implementation follows professional software development practices and can serve as a reference for building production-ready agent systems.